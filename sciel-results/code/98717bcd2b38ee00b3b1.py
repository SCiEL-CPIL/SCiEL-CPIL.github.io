import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _exact_radius(C):
    """Compute the exact packing radius."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    if n < 2:
        return np.inf
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distance
    min_bnd = np.min(0.5 - np.abs(C))
    
    return min(min_pair, min_bnd)

def _repulsive_energy_and_grad(C, power=4.0):
    """
    Compute repulsive energy E = sum_{i<j} 1/d_{ij}^power and its gradient.
    Used to spread points apart initially.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1) + 1e-12
    np.fill_diagonal(dists_sq, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[triu_idx]
    
    p = power
    inv_d_sq_p = np.power(pair_dists_sq, -p/2.0)
    energy = np.sum(inv_d_sq_p)
    
    # Gradient: dE/dx_i = sum_j -p * (d^2)^(-p/2 - 1) * (x_i - x_j)
    coeff = -p * np.power(pair_dists_sq, -p/2.0 - 1.0)
    d_vecs = C[triu_idx[0]] - C[triu_idx[1]]
    
    grad_C = np.zeros_like(C)
    grad_contrib = coeff[:, np.newaxis] * d_vecs
    np.add.at(grad_C, triu_idx[0], grad_contrib)
    np.add.at(grad_C, triu_idx[1], -grad_contrib)
    
    return energy, grad_C.flatten()

def _soft_min_radius_and_grad(C, beta):
    """
    Compute the soft-min radius and its gradient.
    Returns (value, gradient) where value is -soft_min_radius.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    # Pairwise terms
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0
    
    # Boundary terms
    boundary_terms = 0.5 - np.abs(C)
    boundary_terms_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_terms, boundary_terms_flat])
    
    # Soft min calculation using log-sum-exp for numerical stability
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    
    # Clip to prevent underflow/overflow
    shifted = np.clip(shifted, -700, 700)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
        weights = np.zeros_like(all_terms)
        min_indices = np.where(all_terms == min_val)[0]
        weights[min_indices] = 1.0 / len(min_indices)
    else:
        weights = exp_vals / sum_exp
    
    soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
    
    grad_C = np.zeros_like(C)
    
    pair_weights = weights[:len(pair_terms)]
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    
    d_vecs = C[i_idx] - C[j_idx]
    d_norms = np.linalg.norm(d_vecs, axis=1)
    d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)
    
    # d term_k / d c_i = (c_i - c_j) / (2 * d)
    grad_pair_i = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    grad_pair_j = -pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    
    np.add.at(grad_C, i_idx, grad_pair_i)
    np.add.at(grad_C, j_idx, grad_pair_j)
    
    # Boundary gradient
    bnd_weights = weights[len(pair_terms):].reshape(n, 3)
    signs = np.sign(C)
    signs[np.abs(C) < 1e-15] = 1.0
    grad_C += -bnd_weights * signs
    
    return -soft_min_val, -grad_C.flatten()

def _hard_min_radius_and_grad(C, tol=1e-8):
    """
    Compute the hard-min radius and its gradient (approximated via subgradient).
    Returns (value, gradient) where value is -hard_min_radius.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    # Pairwise terms
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0
    
    # Boundary terms
    boundary_terms = 0.5 - np.abs(C)
    boundary_terms_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_terms, boundary_terms_flat])
    
    min_val = np.min(all_terms)
    min_indices = np.where(np.abs(all_terms - min_val) < tol)[0]
    
    if len(min_indices) == 0:
        min_indices = np.array([np.argmin(all_terms)])
    
    weights = np.zeros_like(all_terms)
    weights[min_indices] = 1.0 / len(min_indices)
    
    grad_C = np.zeros_like(C)
    
    pair_weights = weights[:len(pair_terms)]
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    
    d_vecs = C[i_idx] - C[j_idx]
    d_norms = np.linalg.norm(d_vecs, axis=1)
    d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)
    
    grad_pair_i = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    grad_pair_j = -pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    
    np.add.at(grad_C, i_idx, grad_pair_i)
    np.add.at(grad_C, j_idx, grad_pair_j)
    
    bnd_weights = weights[len(pair_terms):].reshape(n, 3)
    signs = np.sign(C)
    signs[np.abs(C) < 1e-15] = 1.0
    grad_C += -bnd_weights * signs
    
    return -min_val, -grad_C.flatten()

def _bcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # BCC: 2 points per unit cell
    # Try to fit in a grid
    n_cells = max(1, int(round(n_spheres / 2.0)))
    m = int(np.ceil(n_cells ** (1.0 / 3.0)))
    while m ** 3 < n_cells:
        m += 1
        
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    points = points[:n_spheres]
    
    span = m - 1 if m > 1 else 1.0
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    
    points += rng.uniform(-0.02, 0.02, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points

def _fcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # FCC: 4 points per unit cell
    n_cells = max(1, int(round(n_spheres / 4.0)))
    m = int(np.ceil(n_cells ** (1.0 / 3.0)))
    while m ** 3 < n_cells:
        m += 1
        
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    points = points[:n_spheres]
    
    span = m - 1 if m > 1 else 1.0
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    
    points += rng.uniform(-0.02, 0.02, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points

def _cubic_grid_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n_spheres].copy()
    grid += rng.uniform(-0.01, 0.01, size=grid.shape)
    return grid

def _random_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(n_spheres, 3))

def _jittered_grid_init(n_spheres=50, seed=42):
    """Create a grid with more aggressive jitter to break symmetry."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n_spheres].copy()
    # Larger jitter to break lattice symmetry
    grid += rng.uniform(-0.05, 0.05, size=grid.shape)
    grid = np.clip(grid, -0.45, 0.45)
    return grid

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Leave 10s margin for the 150s kill
    
    # Generate diverse initializations
    initializations = [
        _bcc_lattice_init(N, seed=42),
        _fcc_lattice_init(N, seed=42),
        _cubic_grid_init(N, seed=42),
        _jittered_grid_init(N, seed=99),
        _random_init(N, seed=7),
    ]
    
    best_C = None
    best_r = -np.inf
    
    # Evaluate initializations and sort by quality
    init_scores = []
    for init in initializations:
        r = _exact_radius(init)
        init_scores.append((r, init))
        if r > best_r:
            best_r = r
            best_C = init.copy()
    
    # Sort by radius (descending) so we process best initializations first
    init_scores.sort(key=lambda x: -x[0])
    sorted_inits = [x[1] for x in init_scores]
    
    # Optimization Strategy:
    # 1. Repulsive energy pre-pass to spread points
    # 2. Coarse-to-fine soft-min with multiple restarts
    # 3. Final high-precision polish
    
    # Beta schedule: start coarse, get finer
    betas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    
    # For each initialization, run the optimization
    for init in sorted_inits:
        if time.time() > deadline - 50.0:
            break
            
        C_curr = init.copy()
        
        # Step 0: Repulsive energy pre-pass to spread points
        if time.time() < deadline - 45.0:
            def rep_obj(x):
                C = x.reshape(-1, 3)
                val, _ = _repulsive_energy_and_grad(C, power=4.0)
                return val
                
            def rep_grad(x):
                C = x.reshape(-1, 3)
                _, grad = _repulsive_energy_and_grad(C, power=4.0)
                return grad

            try:
                res = minimize(
                    rep_obj,
                    C_curr.flatten(),
                    jac=rep_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
                )
                C_new = res.x.reshape(-1, 3)
                C_new = np.clip(C_new, -0.45, 0.45)
                r_new = _exact_radius(C_new)
                if r_new > _exact_radius(C_curr):
                    C_curr = C_new
            except Exception:
                pass
        
        # Step 1: Soft-min schedule
        for beta in betas:
            if time.time() > deadline - 35.0:
                break
                
            def soft_obj(x, b=beta):
                C = x.reshape(-1, 3)
                val, _ = _soft_min_radius_and_grad(C, b)
                return val
                
            def soft_grad(x, b=beta):
                C = x.reshape(-1, 3)
                _, grad = _soft_min_radius_and_grad(C, b)
                return grad

            try:
                # Use L-BFGS-B with careful tolerances
                res = minimize(
                    soft_obj,
                    C_curr.flatten(),
                    jac=soft_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
                )
                C_new = res.x.reshape(-1, 3)
                
                # Check if the new solution is better
                r_new = _exact_radius(C_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
                
                # Update current for next beta step
                C_curr = C_new
            except Exception:
                pass
        
        # Step 2: Hard-min polishing on the best found for this init
        if time.time() > deadline - 20.0:
            continue
        
        # Use the best C found in this trajectory (or C_curr)
        for _ in range(2):  # Two rounds of hard-min polishing
            if time.time() > deadline - 15.0:
                break
            
            def hard_obj(x):
                C = x.reshape(-1, 3)
                val, _ = _hard_min_radius_and_grad(C)
                return val
                
            def hard_grad(x):
                C = x.reshape(-1, 3)
                _, grad = _hard_min_radius_and_grad(C)
                return grad

            try:
                res = minimize(
                    hard_obj,
                    C_curr.flatten(),
                    jac=hard_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                C_new = res.x.reshape(-1, 3)
                r_new = _exact_radius(C_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
                
                C_curr = C_new
            except Exception:
                pass
    
    # Final high-precision polish on global best
    if time.time() < deadline - 5.0 and best_C is not None:
        # Multi-stage final polish with increasing beta
        for beta_final in [5000.0, 10000.0]:
            if time.time() > deadline - 3.0:
                break
            
            def final_obj(x, b=beta_final):
                C = x.reshape(-1, 3)
                val, _ = _soft_min_radius_and_grad(C, b)
                return val
                
            def final_grad(x, b=beta_final):
                C = x.reshape(-1, 3)
                _, grad = _soft_min_radius_and_grad(C, b)
                return grad

            try:
                res = minimize(
                    final_obj,
                    best_C.flatten(),
                    jac=final_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
                )
                C_new = res.x.reshape(-1, 3)
                r_new = _exact_radius(C_new)
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
            except Exception:
                pass
        
        # Another round of hard-min polishing
        if time.time() < deadline - 2.0:
            def final_hard_obj(x):
                C = x.reshape(-1, 3)
                val, _ = _hard_min_radius_and_grad(C, tol=1e-10)
                return val
                
            def final_hard_grad(x):
                C = x.reshape(-1, 3)
                _, grad = _hard_min_radius_and_grad(C, tol=1e-10)
                return grad

            try:
                res = minimize(
                    final_hard_obj,
                    best_C.flatten(),
                    jac=final_hard_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
                )
                C_new = res.x.reshape(-1, 3)
                r_new = _exact_radius(C_new)
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
            except Exception:
                pass

    # Ensure validity: clip to cube
    if best_C is not None:
        best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    else:
        best_C = _cubic_grid_init(N)
        best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
        
    return best_C
