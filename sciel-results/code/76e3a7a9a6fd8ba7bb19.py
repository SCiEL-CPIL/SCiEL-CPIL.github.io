"""Pack 50 equal circles in the unit disk: soft-min L-BFGS-B + physics-based relaxation + SA."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _raw_score(centers):
    """
    Calculate the maximum radius r such that all circles fit.
    r = min( min_pair_dist/2, min_edge_dist )
    """
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    centers = centers.reshape(N, 2)
    norms = np.sqrt(np.sum(centers**2, axis=1))
    if np.any(norms > 1.0 + 1e-9):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_edge_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_edge_dist)
    return max(0.0, r)

def _soft_min_objective(centers, beta=100.0):
    """
    Objective function for L-BFGS-B: minimize -soft_min(constraints)
    """
    centers = centers.reshape(N, 2)
    norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
    edge_constraints = 1.0 - norms
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    neg_beta_x = -beta * all_constraints
    max_val = np.max(neg_beta_x)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val

def _soft_min_gradient(centers, beta=100.0):
    """
    Gradient of the soft-min objective.
    """
    centers = centers.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
    edge_constraints = 1.0 - norms
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    neg_beta_x = -beta * all_constraints
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Edge gradient contribution
    edge_w = weights[:N]
    grad += edge_w[:, np.newaxis] * (centers / norms[:, np.newaxis])
    
    # Pair gradient contribution
    n_pairs = len(pair_constraints)
    i_idx = iu[0]
    j_idx = iu[1]
    w_pair = weights[N:N+n_pairs]
    
    diff_pairs = centers[i_idx] - centers[j_idx]
    dist_pairs = dists[i_idx, j_idx]
    dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
    
    np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_vecs)
    np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_vecs)
    
    return grad.reshape(-1)

def _physics_energy(centers, k_repel=500.0, k_wall=200.0, wall_softness=5.0):
    """
    Physics-based energy: strong repulsion between circles + soft wall potential.
    """
    centers = centers.reshape(N, 2)
    
    # Repulsion energy: sum of 1/d^4 for all pairs (strong inverse power)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2) + 1e-12
    np.fill_diagonal(dists_sq, np.inf)
    # Use 1/d^4 = 1/dists_sq^2
    repulsion = np.sum(1.0 / (dists_sq**2))
    
    # Wall potential: soft penalty for going outside unit circle
    norms = np.sqrt(np.sum(centers**2, axis=1))
    # Soft wall: penalize when norm > 1 - margin
    margin = 0.05
    excess = np.maximum(0, norms - (1.0 - margin))
    wall_energy = np.sum(excess**2 * wall_softness)
    
    return k_repel * repulsion + k_wall * wall_energy

def _physics_gradient(centers, k_repel=500.0, k_wall=200.0, wall_softness=5.0):
    """
    Gradient of physics-based energy.
    """
    centers = centers.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    # Repulsion gradient: d/dx_i [sum_{j!=i} 1/|x_i - x_j|^4]
    # = sum_{j!=i} -4 * (x_i - x_j) / |x_i - x_j|^6
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2) + 1e-12
    np.fill_diagonal(dists_sq, 0)  # Avoid self-interaction
    
    # dists^6 = dists_sq^3
    dists_6 = dists_sq**3
    
    # Gradient contribution from repulsion
    # For each pair (i,j), force on i is proportional to (x_i - x_j) / dists^6
    force = -4.0 * diff / dists_6[:, :, np.newaxis]
    np.fill_diagonal(force, 0)
    repulsion_grad = np.sum(force, axis=1)
    
    grad += k_repel * repulsion_grad
    
    # Wall gradient: d/dx_i [k_wall * wall_softness * max(0, |x_i| - (1-margin))^2]
    norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
    margin = 0.05
    excess = np.maximum(0, norms - (1.0 - margin))
    wall_grad = 2.0 * wall_softness * excess[:, np.newaxis] * (centers / norms[:, np.newaxis])
    
    grad += k_wall * wall_grad
    
    return grad.reshape(-1)

def _physics_relaxation(centers, max_iter=500, initial_step=0.01, deadline=None):
    """
    Gradient descent on physics-based energy to find dense packing.
    """
    current = centers.copy().reshape(N, 2)
    step_size = initial_step
    best_score = _raw_score(current)
    best_centers = current.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy = _physics_energy(current)
        grad = _physics_gradient(current).reshape(N, 2)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Adaptive step size
        new_centers = current - step_size * grad / grad_norm
        
        # Keep inside unit circle
        norms = np.sqrt(np.sum(new_centers**2, axis=1))
        out_mask = norms > 0.9999
        if np.any(out_mask):
            new_centers[out_mask] *= 0.9999 / norms[out_mask, np.newaxis]
        
        new_score = _raw_score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
        
        # Accept if energy decreased or score improved
        new_energy = _physics_energy(new_centers)
        if new_energy < energy or new_score > _raw_score(current):
            current = new_centers
            # Slowly decrease step size
            step_size *= 0.995
        else:
            # Reject, reduce step
            step_size *= 0.8
    
    return best_centers, best_score

def _newton_polish(centers, max_iter=50, tol=1e-12, deadline=None):
    """
    Newton-Raphson contact-graph refinement: find the largest r such that
    all circles are tangent to neighbors or boundary.
    """
    centers = centers.copy().reshape(N, 2)
    r = _raw_score(centers)
    
    if r <= 0:
        return centers, r
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current minimum distances
        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_dists = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Find contact pairs (within tolerance of current r)
        contact_tol = max(r * 0.05, 1e-6)
        contact_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                if abs(pair_dists[i,j] - r) < contact_tol:
                    contact_pairs.append((i, j))
        
        # Contact with boundary
        contact_boundary = []
        for i in range(N):
            if abs(edge_dists[i] - r) < contact_tol:
                contact_boundary.append(i)
        
        # Compute subgradient of r
        grad = np.zeros_like(centers)
        
        if contact_boundary:
            for i in contact_boundary:
                if norms[i] > 1e-10:
                    grad[i] += centers[i] / norms[i]
        
        if contact_pairs:
            for i, j in contact_pairs:
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    direction = d / dist
                    grad[i] -= direction
                    grad[j] += direction
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        # Try to increase r
        step_size = 0.01
        improved = False
        for _ in range(5):
            new_centers = centers + step_size * grad
            new_norms = np.sqrt(np.sum(new_centers**2, axis=1))
            out_mask = new_norms > 0.9999
            if np.any(out_mask):
                new_centers[out_mask] *= 0.9999 / new_norms[out_mask, np.newaxis]
            
            new_r = _raw_score(new_centers)
            if new_r > r + 1e-15:
                centers = new_centers
                r = new_r
                improved = True
                break
            step_size *= 0.5
        
        if not improved:
            break
        
        if abs(r - _raw_score(centers)) < tol:
            break
    
    return centers, r

def _refine(centers, beta=300.0, maxiter=200, deadline=None):
    """
    Local refinement using L-BFGS-B with soft-min objective.
    """
    try:
        result = minimize(
            _soft_min_objective,
            centers.flatten(),
            args=(beta,),
            jac=lambda x, b: _soft_min_gradient(x, b),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        refined = result.x.reshape(N, 2)
        norms = np.sqrt(np.sum(refined**2, axis=1))
        mask = norms > 1.0
        if np.any(mask):
            refined[mask] *= (1.0 - 1e-6) / norms[mask, np.newaxis]
        return refined
    except Exception:
        return centers

def _simulated_annealing(centers, steps=1500, T_init=0.015, T_final=0.0003, deadline=None):
    """
    Simulated Annealing for global exploration.
    """
    current_centers = centers.copy()
    current_score = _raw_score(current_centers)
    best_centers = current_centers.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, 1)
    
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        T = T_init * (T_final / T_init) ** (i / steps)
        
        num_moves = np.random.randint(1, 4)
        indices = np.random.choice(N, num_moves, replace=False)
        
        new_centers = current_centers.copy()
        step_size = max(T * 0.5, 0.003)
        moves = np.random.normal(0, step_size, size=(num_moves, 2))
        new_centers[indices] += moves
        
        norms = np.sqrt(np.sum(new_centers**2, axis=1))
        out_mask = norms > 0.9995
        if np.any(out_mask):
            new_centers[out_mask] *= 0.9995 / norms[out_mask, np.newaxis]
            
        new_score = _raw_score(new_centers)
        delta = new_score - current_score
        
        if delta > 0:
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = new_centers.copy()
        else:
            clamped_delta = max(delta, -10.0 * T)
            if np.random.rand() < np.exp(clamped_delta / T):
                current_centers = new_centers
                current_score = new_score
    
    return best_centers, best_score

def _make_rings(radii, counts, offsets=None):
    """Generate ring-based initial configuration."""
    centers = []
    centers.append([0.0, 0.0])
    start_idx = 1
    for r, c in zip(radii, counts):
        base_offset = offsets[start_idx-1] if offsets else 0.0
        for k in range(c):
            theta = 2 * np.pi * k / c + base_offset
            centers.append([r * np.cos(theta), r * np.sin(theta)])
        start_idx += 1
    return np.array(centers[:N])

def _hexagonal_seed(n, jitter=0.0):
    """Generate hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) >= N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        pts = points[idx]
        if jitter > 0:
            pts += np.random.normal(0, jitter, pts.shape)
            norms = np.sqrt(np.sum(pts**2, axis=1))
            out_mask = norms > 0.999
            if np.any(out_mask):
                pts[out_mask] *= 0.999 / norms[out_mask, np.newaxis]
        return pts
    return None

def construct():
    """Main entry point: find best packing of 50 circles in unit disk."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Ring-based initializations
    r1 = [0.0, 0.24, 0.46, 0.68, 0.88]
    c1 = [1, 6, 12, 18, 13]
    initializations.append(_make_rings(r1, c1))
    
    for seed in range(10):
        np.random.seed(seed)
        perturbed_radii = [r1[0]] + [r + np.random.uniform(-0.02, 0.02) for r in r1[1:]]
        offsets = [0.0] * len(r1)
        for s in range(1, len(r1)):
            offsets[s] = np.random.uniform(-0.15, 0.15)
        initializations.append(_make_rings(perturbed_radii, c1, offsets))
    
    ring_configs = [
        ([0.0, 0.25, 0.5, 0.75, 0.9], [1, 6, 12, 18, 13]),
        ([0.0, 0.22, 0.44, 0.66, 0.85], [1, 7, 14, 14, 14]),
        ([0.0, 0.2, 0.4, 0.6, 0.8, 0.95], [1, 5, 10, 15, 10, 9]),
        ([0.0, 0.18, 0.38, 0.58, 0.78, 0.92], [1, 6, 11, 16, 10, 6]),
        ([0.0, 0.2, 0.45, 0.7, 0.9], [1, 8, 16, 14, 11]),
    ]
    for radii, counts in ring_configs:
        total = sum(counts)
        if total < N:
            counts[-1] += N - total
        elif total > N:
            counts[-1] -= total - N
            if counts[-1] <= 0:
                continue
        cfg = _make_rings(radii, counts)
        if cfg is not None and len(cfg) == N:
            initializations.append(cfg)
    
    # 2. Hexagonal lattice initializations
    hex_pts = _hexagonal_seed(N)
    if hex_pts is not None:
        initializations.append(hex_pts)
        for j in [0.01, 0.02, 0.05]:
            hex_jit = _hexagonal_seed(N, jitter=j)
            if hex_jit is not None:
                initializations.append(hex_jit)
    
    # 3. Random initializations
    for seed in range(5):
        np.random.seed(100 + seed)
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)
    
    # Filter valid initializations
    valid_inits = []
    for init in initializations:
        if init is not None and len(init) == N and np.all(np.isfinite(init)):
            norms = np.sqrt(np.sum(init**2, axis=1))
            if np.all(norms <= 1.0):
                valid_inits.append(init)
    
    if not valid_inits:
        angles = 2 * np.pi * np.arange(N) / N
        radii = np.sqrt(np.linspace(0, 1, N))
        valid_inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Optimize each initialization
    best_points = None
    best_score = 0.0
    
    for idx, init in enumerate(valid_inits):
        if time.time() > deadline - 20.0:
            break
        
        # Physics-based relaxation phase
        phys_points, phys_score = _physics_relaxation(init, max_iter=300, initial_step=0.01, deadline=deadline)
        
        # Simulated annealing phase
        sa_time_budget = 15.0
        sa_steps = int(sa_time_budget * 80)
        sa_points, sa_score = _simulated_annealing(phys_points, steps=min(sa_steps, 1000), 
                                                    T_init=0.015, T_final=0.0003, deadline=deadline)
        
        # Refine with L-BFGS-B
        refined_points = _refine(sa_points, beta=300.0, maxiter=100, deadline=deadline)
        refined_score = _raw_score(refined_points)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
        
        # Newton polish
        if time.time() < deadline - 10.0:
            polished_points, polished_score = _newton_polish(refined_points, max_iter=30, deadline=deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points.copy()
        
        # Try another refinement with different beta
        if time.time() < deadline - 10.0:
            refined_points2 = _refine(refined_points, beta=500.0, maxiter=100, deadline=deadline)
            refined_score2 = _raw_score(refined_points2)
            
            if refined_score2 > best_score:
                best_score = refined_score2
                best_points = refined_points2.copy()
    
    # Final refinement on best
    if best_points is not None and time.time() < deadline - 5.0:
        final_points = _refine(best_points, beta=500.0, maxiter=200, deadline=deadline)
        final_score = _raw_score(final_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Final Newton polish
    if best_points is not None and time.time() < deadline - 2.0:
        final_polished, final_polished_score = _newton_polish(best_points, max_iter=50, deadline=deadline)
        if final_polished_score > best_score:
            best_score = final_polished_score
            best_points = final_polished
    
    # Ensure we return a valid result
    if best_points is None or len(best_points) != N:
        angles = 2 * np.pi * np.arange(N) / N
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Final validity check
    norms = np.sqrt(np.sum(best_points**2, axis=1))
    out_mask = norms > 1.0
    if np.any(out_mask):
        best_points[out_mask] *= (1.0 - 1e-6) / norms[out_mask, np.newaxis]
    
    return best_points
