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
    
    # Soft min calculation using log-sum-exp trick
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
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

def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t such that all constraints are satisfied.
    We use a smooth approximation: minimize t + penalty for constraint violations.
    Actually, let's use a different approach: we minimize the maximum violation.
    
    Let's use: min over C of max(0, t - min_constraint(C)) + t
    where t is the radius. This is equivalent to minimizing t subject to constraints.
    
    Simpler: just minimize -soft_min_radius as before, but with better schedule.
    """
    C = x.reshape(-1, 3)
    val, _ = _soft_min_radius_and_grad(C, beta)
    return val

def _epigraph_gradient(x, beta):
    C = x.reshape(-1, 3)
    _, grad = _soft_min_radius_and_grad(C, beta)
    return grad

def _bcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # BCC: 2 points per unit cell
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

def _optimize_from_init(C_init, deadline, current_best_r):
    """
    Optimize starting from C_init, using a coarse-to-fine soft-min schedule.
    Returns (best_C, best_r) found from this initialization.
    """
    C_curr = C_init.copy()
    best_C = C_init.copy()
    best_r = _exact_radius(C_init)
    
    # Soft-min beta schedule: start coarse, get fine
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        def obj(x, b=beta):
            C = x.reshape(-1, 3)
            val, _ = _soft_min_radius_and_grad(C, b)
            return val
            
        def grad(x, b=beta):
            C = x.reshape(-1, 3)
            _, g = _soft_min_radius_and_grad(C, b)
            return g

        try:
            # Use L-BFGS-B with reasonable settings
            res = minimize(
                obj,
                C_curr.flatten(),
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            C_new = res.x.reshape(-1, 3)
            
            # Clip to valid range
            C_new = np.clip(C_new, -0.5 + 1e-15, 0.5 - 1e-15)
            
            r_new = _exact_radius(C_new)
            
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
            
            # Update C_curr for next beta step
            # Only update if it improved or is comparable
            if r_new >= best_r - 1e-6:
                C_curr = C_new
            else:
                # Keep previous if it got worse
                pass
                
        except Exception:
            pass
    
    return best_C, best_r

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Leave 15s margin for the 150s kill
    
    # Generate multiple initializations
    initializations = []
    
    # Structured lattices with different seeds
    for seed in [42, 123, 456, 789]:
        initializations.append(_bcc_lattice_init(N, seed=seed))
        initializations.append(_fcc_lattice_init(N, seed=seed))
        initializations.append(_cubic_grid_init(N, seed=seed))
    
    # Random initializations
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(_random_init(N, seed=seed))
    
    # Evaluate initializations and pick the best few
    init_radii = []
    for init in initializations:
        r = _exact_radius(init)
        init_radii.append(r)
    
    # Sort by radius, take top initializations
    sorted_indices = np.argsort(init_radii)[::-1]
    top_k = min(6, len(initializations))  # Use top 6 initializations
    
    best_C = None
    best_r = -np.inf
    
    # Optimize from each of the top initializations
    for idx in sorted_indices[:top_k]:
        if time.time() > deadline - 15.0:
            break
            
        C_init = initializations[idx]
        C_opt, r_opt = _optimize_from_init(C_init, deadline, best_r)
        
        if r_opt > best_r:
            best_r = r_opt
            best_C = C_opt
    
    # Final high-precision polish on global best
    if time.time() < deadline - 5.0 and best_C is not None:
        beta_final = 5000.0
        
        def final_obj(x):
            C = x.reshape(-1, 3)
            val, _ = _soft_min_radius_and_grad(C, beta_final)
            return val
            
        def final_grad(x):
            C = x.reshape(-1, 3)
            _, g = _soft_min_radius_and_grad(C, beta_final)
            return g

        try:
            res = minimize(
                final_obj,
                best_C.flatten(),
                jac=final_grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 100
                }
            )
            C_new = res.x.reshape(-1, 3)
            C_new = np.clip(C_new, -0.5 + 1e-15, 0.5 - 1e-15)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new
        except Exception:
            pass
    
    # Ensure validity
    if best_C is None:
        best_C = _cubic_grid_init(N)
    
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_C
