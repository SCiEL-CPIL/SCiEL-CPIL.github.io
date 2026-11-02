import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using broadcasting (N=100 is small)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _potential_energy_objective_and_grad(centers_flat, gamma=10.0):
    """
    Minimize potential energy: sum over pairs of 1/(dist^2 + eps) + wall terms.
    This is smooth and encourages spreading out.
    """
    centers = centers_flat.reshape(-1, 3)
    n = centers.shape[0]
    eps = 1e-6
    
    # Wall penalty: sum of 1/(dist_to_wall^2 + eps)
    dists_to_walls = 0.5 - np.abs(centers)
    wall_energy = np.sum(1.0 / (dists_to_walls ** 2 + eps))
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    pair_energy = np.sum(1.0 / (dists_sq + eps))
    
    obj = wall_energy + gamma * pair_energy
    
    # Gradient
    grad = np.zeros_like(centers_flat)
    grad_flat = grad.reshape(n, 3)
    
    # Wall gradient: d/dx_k of 1/((0.5-|x_k|)^2 + eps) = 2*(0.5-|x_k|)*sign(x_k) / ((0.5-|x_k|)^2 + eps)^2
    for k in range(3):
        dw = dists_to_walls[:, k]
        signs = np.sign(centers[:, k])
        signs[signs == 0] = 1.0
        grad_flat[:, k] += 2.0 * dw * signs / (dw ** 2 + eps) ** 2
    
    # Pairwise gradient: d/dc_i of 1/(|c_i-c_j|^2 + eps) = -2*(c_i-c_j) / (|c_i-c_j|^2 + eps)^2
    for i in range(n):
        for j in range(i+1, n):
            d = diff[i, j]
            dsq = dists_sq[i, j]
            factor = -2.0 * gamma / (dsq + eps) ** 2
            grad_flat[i] += factor * d
            grad_flat[j] -= factor * d
    
    return obj, grad

def _build_fcc_init():
    """
    Build an FCC (face-centered cubic) lattice initialization.
    FCC has better packing density than BCC.
    """
    # FCC unit cell has 4 atoms: corners + face centers
    # We'll create a grid of FCC unit cells
    
    nx, ny, nz = 4, 5, 5  # 4*5*5 = 100 unit cells, each with 4 points = 400, take first 100
    a = 1.0 / nx  # lattice constant in x
    
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner
                cx = -0.5 + (i + 0.5) * a
                cy = -0.5 + (j + 0.5) * a
                cz = -0.5 + (k + 0.5) * a
                points.append([cx, cy, cz])
                # Face centers
                points.append([cx + a/2, cy, cz])
                points.append([cx, cy + a/2, cz])
                points.append([cx, cy, cz + a/2])
    
    centers = np.array(points[:N])
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _soft_min_objective_and_grad(centers_flat, beta, iu):
    """
    Objective and gradient for soft-min approximation of the radius.
    Returns (obj, grad).
    """
    centers = centers_flat.reshape(-1, 3)
    n = centers.shape[0]
    
    # Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()  # shape (3*n,)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]  # shape (n*(n-1)/2,)
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    m = len(all_vals)
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * x_i)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    log_sum = max_val + np.log(sum_exp)
    obj = -log_sum / beta
    
    # Gradient: d/dx_i of obj = exp(-beta*x_i) / sum(exp(-beta*x_j)) * (d x_i / d centers)
    # For soft min: grad_obj = (exp_vals / sum_exp) * (-beta) * (d all_vals / d centers)
    # Wait: obj = -(1/beta) * log(sum(exp(-beta * all_vals)))
    # d obj / d all_vals_k = -(1/beta) * (1/sum) * exp(-beta*all_vals_k) * (-beta) = exp_vals_k / sum_exp
    # So d obj / d centers = sum_k (exp_vals_k / sum_exp) * d all_vals_k / d centers
    
    weights = exp_vals / sum_exp  # shape (m,)
    
    grad = np.zeros_like(centers_flat)
    
    # Wall contributions: all_vals[0:3*n] = dists_to_walls.flatten()
    # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
    wall_weights = weights[:3*n].reshape(n, 3)
    grad_flat = grad.reshape(n, 3)
    for k in range(3):
        signs = np.sign(centers[:, k])
        signs[signs == 0] = 1.0
        grad_flat[:, k] -= wall_weights[:, k] * signs
    
    # Pair contributions: all_vals[3*n:] = pair_vals_unique
    # pair_vals[i,j] = |c_i - c_j| / 2
    # d pair_vals[i,j] / d c_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # d pair_vals[i,j] / d c_j = (c_j - c_i) / (2 * |c_i - c_j|) = -d pair_vals[i,j] / d c_i
    
    pair_weights = weights[3*n:]  # shape (n*(n-1)/2,)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            w = pair_weights[idx]
            idx += 1
            d = diff[i, j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                direction = d / (2.0 * dist)
                grad_flat[i] += w * direction
                grad_flat[j] -= w * direction
    
    return obj, grad

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    nx, ny, nz = 5, 5, 2
    n_cells = nx * ny * nz  # 50
    
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_sobol_init(seed=42):
    """
    Generate N points using a Sobol sequence (low discrepancy).
    """
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=3, seed=seed)
    points = sampler.random(N)
    # Map [0,1] to [-0.5, 0.5]
    centers = points - 0.5
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish_repulsion(centers, deadline, iu):
    """
    Local search using repulsion forces with KDTree for efficiency.
    Improved with adaptive step size and better force scaling.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.002
    decay = 0.98
    min_step = 1e-8
    margin = 1.02
    
    no_improve_count = 0
    max_no_improve = 150
    
    while _time_left() > 1.0 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(centers)
        if r > best_r + 1e-13:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
            # Slightly increase step if we're improving well
            step_size = min(step_size * 1.02, 0.005)
        else:
            no_improve_count += 1
            step_size *= 0.95
        
        # Compute forces using KDTree for pairwise interactions
        forces = np.zeros_like(centers)
        
        # Wall forces: stronger push when close to wall
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.005)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                # Quadratic strength for stronger push near wall
                strengths = ((limit - dists[mask]) / limit) ** 2
                forces[mask, k] += dirs * strengths * 2.0
                
        # Pairwise forces using KDTree
        tree = cKDTree(centers)
        limit = 2 * max(r_current * margin, 0.005)
        pairs = tree.query_pairs(limit, output_type='ndarray')
        
        if len(pairs) > 0:
            for i, j in pairs:
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-9:
                    strength = (limit - dist) / limit
                    if strength > 0:
                        # Inverse distance weighting for stronger push at close range
                        inv_dist = 1.0 / (dist + 0.01)
                        force_dir = diff / dist
                        forces[i] += force_dir * strength * inv_dist
                        forces[j] -= force_dir * strength * inv_dist
        
        # Normalize forces to prevent large jumps
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.49999, 0.49999)
        
        step_size *= decay
        
    return best_centers, best_r

def _polish_tight(centers, deadline, iu):
    """
    Final tight polish with very small steps for high-precision convergence.
    Uses gradient-based approach for final refinement.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    # First do a few rounds of force-directed polish
    step_size = 0.0003
    decay = 0.995
    min_step = 1e-9
    margin = 1.005
    
    no_improve_count = 0
    max_no_improve = 200
    
    while _time_left() > 0.5 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(centers)
        if r > best_r + 1e-14:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.001)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = ((limit - dists[mask]) / limit) ** 2
                forces[mask, k] += dirs * strengths * 3.0
                
        # Pairwise forces using KDTree
        tree = cKDTree(centers)
        limit = 2 * max(r_current * margin, 0.001)
        pairs = tree.query_pairs(limit, output_type='ndarray')
        
        if len(pairs) > 0:
            for i, j in pairs:
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-9:
                    strength = (limit - dist) / limit
                    if strength > 0:
                        inv_dist = 1.0 / (dist + 0.005)
                        force_dir = diff / dist
                        forces[i] += force_dir * strength * inv_dist
                        forces[j] -= force_dir * strength * inv_dist
        
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.499999, 0.499999)
        
        step_size *= decay
    
    # Final gradient-based polish using soft-min with very high beta
    if _time_left() > 1.0:
        try:
            beta = 5000.0
            res = minimize(
                _soft_min_objective_and_grad,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (N * 3),
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 10}
            )
            new_centers = np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
            r = _compute_radius(new_centers)
            if r > best_r:
                best_r = r
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_r

def _penalty_optimize(centers, r_init, deadline):
    """
    Use penalty method to optimize radius directly.
    """
    x0 = np.concatenate([centers.flatten(), [r_init]])
    bounds = [(-0.4999, 0.4999)] * (N * 3) + [(0.0, 0.5)]
    
    best_x = x0.copy()
    best_r = r_init
    
    penalty = 1.0
    for _ in range(10):
        if _time_left() < 2.0:
            break
        
        try:
            res = minimize(
                _penalty_objective,
                best_x,
                args=(None, penalty),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            new_x = res.x
            new_centers = new_x[:N*3].reshape(N, 3)
            new_r = new_x[N*3]
            
            # Check if this is valid (no violations)
            r_actual = _compute_radius(new_centers)
            
            if r_actual > best_r:
                best_r = r_actual
                best_x = new_x.copy()
            else:
                # If penalty didn't improve, increase penalty
                penalty *= 2.0
                continue
                
        except Exception:
            penalty *= 2.0
            continue
        
        # Check convergence
        if penalty > 1e6:
            break
        
        penalty *= 1.5
    
    return best_x[:N*3].reshape(N, 3), best_r

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    remaining = time_budget
    
    # Phase 1: Potential energy minimization (smooth, fast)
    if remaining > 8.0:
        try:
            res = minimize(
                _potential_energy_objective_and_grad,
                best_centers.flatten(),
                args=(10.0,),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            new_centers = np.clip(res.x.reshape(-1, 3), -0.4999, 0.4999)
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Phase 2: Soft-min optimization with increasing beta
    beta_start = 50.0
    beta_end = 2000.0
    n_beta_steps = 5
    
    if _time_left() > 5.0 and (time.time() - start_t) < remaining - 4.0:
        for step in range(n_beta_steps):
            if _time_left() < 3.0 or (time.time() - start_t) > remaining - 3.0:
                break
                
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
            
            x0 = best_centers.flatten()
            
            try:
                max_iter = 60 if step < 2 else 30
                res = minimize(
                    _soft_min_objective_and_grad,
                    x0,
                    args=(beta, iu),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (N * 3),
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 15}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
                    
            except Exception:
                pass
    
    # Phase 3: Polish with repulsion
    if _time_left() > 3.0 and (time.time() - start_t) < remaining - 2.0:
        polish_deadline = min(deadline, start_t + remaining - 2.0)
        polish_centers, polish_radius = _polish_repulsion(best_centers, polish_deadline, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    return best_centers, best_radius

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    def try_init(centers, time_budget=18.0):
        nonlocal best_centers, best_radius
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        if _time_left() > time_budget + 3.0:
            opt_centers, opt_radius = _optimize_from_initial(centers, _DEADLINE, iu, time_budget)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
        else:
            if _time_left() > 4.0:
                polish_centers, polish_radius = _polish_repulsion(centers, _DEADLINE, iu)
                if polish_radius > best_radius:
                    best_radius = polish_radius
                    best_centers = polish_centers
    
    # Initializations - prioritize diverse, good starting points
    # 1. FCC lattice (better packing density)
    if _time_left() > 35.0:
        centers = _build_fcc_init()
        try_init(centers, time_budget=32.0)
    
    # 2. BCC lattice
    if _time_left() > 30.0:
        centers = _build_bcc_init()
        try_init(centers, time_budget=25.0)
    
    # 3. Sobol sequence (low discrepancy)
    if _time_left() > 25.0:
        centers = _build_sobol_init()
        try_init(centers, time_budget=20.0)
    
    # 4. Grids with slight perturbation
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
    ]
    
    for nx, ny, nz in grid_configs:
        if _time_left() < 25.0:
            break
        centers = _build_grid_init(nx, ny, nz, 0.005, seed=nx*100+ny*10+nz)
        try_init(centers, time_budget=18.0)
    
    # 5. Halton sequence
    if _time_left() > 20.0:
        centers = _build_halton_init()
        try_init(centers, time_budget=16.0)
    
    # 6. One random initialization
    if _time_left() > 15.0:
        rng = np.random.default_rng(123)
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        try_init(centers, time_budget=12.0)
    
    # Final polish with remaining time
    if best_centers is not None and _time_left() > 8.0:
        polish_centers, polish_radius = _polish_repulsion(best_centers, _DEADLINE, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
    
    # Tight final polish
    if best_centers is not None and _time_left() > 3.0:
        tight_centers, tight_radius = _polish_tight(best_centers, _DEADLINE, iu)
        if tight_radius > best_radius:
            best_radius = tight_radius
            best_centers = tight_centers
    
    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
