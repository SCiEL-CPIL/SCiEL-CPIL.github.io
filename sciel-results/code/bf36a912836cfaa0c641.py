"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.linalg.norm(centers, axis=1) > 1.0 + 1e-9):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    best_pts = None
    best_score = -1.0
    
    for spacing in [0.24, 0.25, 0.26, 0.27, 0.28]:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        mask = radius <= 0.95
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= n:
            idx = np.argsort(radius, kind="stable")[:n]
            pts = points[idx]
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
    return _get_random_initialization(n, seed=42)

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_ring_initialization(n):
    """Generate ring-based initialization."""
    best_pts = None
    best_score = -1.0
    
    for n_rings in [3, 4, 5]:
        pts_list = [[0.0, 0.0]]
        remaining = n - 1
        per_ring = [remaining // n_rings + (1 if i < remaining % n_rings else 0) for i in range(n_rings)]
        
        for ring_idx, count in enumerate(per_ring):
            r = (ring_idx + 1) * 0.3
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for a in angles:
                pts_list.append([r * np.cos(a), r * np.sin(a)])
        
        pts = np.array(pts_list[:n])
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    if best_pts is not None:
        return best_pts
    return _get_initial_hex_lattice(n)

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # We only need upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

def _force_directed_relax(pts, n_steps=200, target_r=0.12):
    """
    Force-directed relaxation to improve packing.
    Pushes overlapping circles apart and pulls them toward the center.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    
    for step in range(n_steps):
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion for overlapping circles
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only consider pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < target_d and d > 1e-10:
                    # Repulsive force proportional to overlap
                    overlap = target_d - d
                    direction = (pts[i] - pts[j]) / d
                    force_mag = overlap * 0.5
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary constraint: push circles inward if they're outside
        dists_origin = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists_origin[i] > 1.0 - target_r:
                # Push inward
                direction = -pts[i] / dists_origin[i]
                excess = dists_origin[i] - (1.0 - target_r)
                forces[i] += excess * direction
        
        # Update positions with damping
        lr = 0.01 * (1.0 - step / n_steps) + 0.001
        pts += lr * forces
        
        # Project to unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
    
    return pts

def _polish_subgradient(centers, start_time, max_iters=2000, step_size=0.005):
    """
    Direct approach: Sub-gradient ascent on the minimum distance.
    Identifies the tightest constraint and pushes points apart.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(bound_dists)
        
        # Determine which constraint is the bottleneck
        # r = min(min_pair_dist/2, min_bound_dist)
        # We want to maximize r.
        
        step = step_size
        
        if min_pair_dist / 2.0 < min_bound_dist:
            # Bottleneck is a pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j: i, j = j, i
            d = dists[i, j]
            if d > 1e-8:
                vec = (current[i] - current[j]) / d
                # Move i and j apart
                current[i] += vec * step * 0.5
                current[j] -= vec * step * 0.5
        else:
            # Bottleneck is boundary
            i = np.argmin(bound_dists)
            norm = norms[i]
            if norm > 1e-8:
                vec = current[i] / norm
                # Move i towards center
                current[i] -= vec * step
        
        # Ensure no points go out of bounds (safety check)
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 0.99
        if np.any(mask):
            current[mask] *= 0.99 / new_norms[mask]
            
        # Check improvement
        new_score = _compute_score(current)
        if new_score > best_score:
            best_score = new_score
            best_centers = current.copy()
        elif it > 100 and (it % 100 == 0):
            # If no improvement for a while, try a smaller step or restart polish?
            # For now, just continue. The L-BFGS phase usually gets us close.
            pass
            
    return best_centers, best_score

def _optimize_from_start(centers, start_time):
    """
    Multi-stage optimization:
    1. L-BFGS-B with soft min (beta 50 -> 500)
    2. Sub-gradient polish
    """
    current = centers.copy()
    
    # Stage 1: L-BFGS-B
    betas = [50.0, 200.0, 500.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad, 
                current.flatten(), 
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            if res.success or np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                # Verify validity
                if np.all(np.isfinite(candidate)):
                    current = candidate
        except Exception:
            pass
            
    # Stage 2: Direct Sub-gradient Polish
    if time.time() - start_time < DEADLINE - 5:
        current, _ = _polish_subgradient(current, start_time, max_iters=5000, step_size=0.002)
        
    return current

def _nelder_mead_objective(params):
    """Objective for Nelder-Mead: negative of the packing score."""
    centers = params.reshape(N, 2)
    return -_compute_score(centers)

def _optimize_nelder_mead(centers, start_time):
    """
    Optimize using Nelder-Mead method.
    """
    current = centers.copy()
    
    # Ensure points are inside the circle
    norms = np.linalg.norm(current, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        current[mask] *= 0.95 / norms[mask]
    
    try:
        res = minimize(
            _nelder_mead_objective,
            current.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if np.isfinite(res.fun):
            candidate = res.x.reshape(N, 2)
            if np.all(np.isfinite(candidate)):
                # Ensure validity
                norms = np.linalg.norm(candidate, axis=1)
                if np.all(norms <= 1.0):
                    current = candidate
    except Exception:
        pass
    
    # Polish with subgradient
    if time.time() - start_time < DEADLINE - 5:
        current, _ = _polish_subgradient(current, start_time, max_iters=3000, step_size=0.002)
        
    return current

def _refine_with_high_beta(centers, start_time, max_betas=5):
    """
    Refine using very high beta soft-min to sharpen the minimum.
    This helps escape shallow local minima by focusing gradient on the tightest constraints.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    betas = [1000, 5000, 10000, 50000, 100000][:max_betas]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 5:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad, 
                current.flatten(), 
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'gtol': 1e-8}
            )
            if np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    score = _compute_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_centers = candidate.copy()
                        current = candidate
        except Exception:
            pass
            
    return best_centers, best_score

def _perturb_and_retry(centers, start_time, n_perturbations=5):
    """
    Perturb the best solution slightly and re-optimize to escape local minima.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    rng = np.random.default_rng(12345)
    
    for p in range(n_perturbations):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Create a perturbed copy
        perturbed = best_centers.copy()
        noise_scale = 0.01 * (1.0 + p * 0.5)
        noise = rng.normal(0, noise_scale, size=perturbed.shape)
        perturbed += noise
        
        # Ensure validity
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            perturbed[mask] *= 0.95 / norms[mask]
            
        # Optimize the perturbed version
        optimized = _optimize_from_start(perturbed, start_time)
        if np.all(np.isfinite(optimized)):
            score = _compute_score(optimized)
            if score > best_score:
                best_score = score
                best_centers = optimized.copy()
                
    return best_centers, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=777),
    ]
    
    # Optimize each initialization with both methods
    for init in initializations:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # Try force-directed relaxation first to improve initial config
        if time.time() - start_time < DEADLINE - 20:
            target_r = max(_compute_score(init), 0.12)
            relaxed_init = _force_directed_relax(init, n_steps=100, target_r=target_r)
            if _compute_score(relaxed_init) > _compute_score(init):
                init = relaxed_init
        
        # Try L-BFGS-B optimization
        if time.time() - start_time < DEADLINE - 20:
            optimized_lbfgs = _optimize_from_start(init, start_time)
            if np.all(np.isfinite(optimized_lbfgs)):
                r = _compute_score(optimized_lbfgs)
                if r > best_score:
                    best_score = r
                    best_centers = optimized_lbfgs.copy()
        
        # Try Nelder-Mead optimization
        if time.time() - start_time < DEADLINE - 15:
            optimized_nm = _optimize_nelder_mead(init, start_time)
            if np.all(np.isfinite(optimized_nm)):
                r = _compute_score(optimized_nm)
                if r > best_score:
                    best_score = r
                    best_centers = optimized_nm.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Final polish on the absolute best
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_subgradient(best_centers, start_time, max_iters=10000, step_size=0.001)
        
    # High-beta refinement to sharpen the minimum
    if time.time() - start_time < DEADLINE - 5:
        refined, refined_score = _refine_with_high_beta(best_centers, start_time, max_betas=5)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined
        
    # Perturbation-retry loop to escape local minima
    if time.time() - start_time < DEADLINE - 10:
        perturbed, perturbed_score = _perturb_and_retry(best_centers, start_time, n_perturbations=3)
        if perturbed_score > best_score:
            best_score = perturbed_score
            best_centers = perturbed
            
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_subgradient(best_centers, start_time, max_iters=5000, step_size=0.0005)
        
    return best_centers
# EVOLVE-BLOCK-END
