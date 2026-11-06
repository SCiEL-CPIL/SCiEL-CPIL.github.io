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

def _jittered_grid_init(n_spheres=50, seed=42, jitter=0.05):
    rng = np.random.RandomState(seed)
    # Create a grid slightly larger than needed to ensure we have enough points
    # 50 points -> roughly 4x4x4 = 64. 
    m = int(np.ceil(n_spheres ** (1.0/3.0)))
    # To get better coverage, we can use a slightly denser grid and pick the best, 
    # or just use a standard grid and jitter.
    # Let's use a 4x4x4 grid = 64 points, pick 50.
    m = 4 
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select n_spheres points from the grid
    indices = rng.choice(len(grid), size=n_spheres, replace=False)
    points = grid[indices].copy()
    
    # Add jitter
    points += rng.uniform(-jitter, jitter, size=points.shape)
    
    # Clip to valid range
    points = np.clip(points, -0.45, 0.45)
    return points

def _fcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # FCC: 4 points per unit cell
    # 50 spheres. 
    # Let's try to fit them in a box.
    # Approx 3x3x3 = 27 cells * 4 = 108 points. Too many.
    # 2x3x3 = 18 cells * 4 = 72 points. 
    # We need 50.
    
    # Generate a large enough FCC lattice
    m_x, m_y, m_z = 2, 3, 3
    points = []
    for ix in range(m_x):
        for iy in range(m_y):
            for iz in range(m_z):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    
    # We have 72 points. We need 50.
    # Randomly select 50? Or select the "best" 50?
    # Random selection from a good lattice is a decent heuristic.
    indices = rng.choice(len(points), size=n_spheres, replace=False)
    points = points[indices]
    
    # Normalize to fit in [-0.5, 0.5]
    # Current span: x: 0..2 (size 2), y: 0..3 (size 3), z: 0..3 (size 3)
    # We want to fit this into a cube.
    # Scale factor: min(1.0 / max_span)
    spans = points.max(axis=0) - points.min(axis=0)
    scale = 0.9 / np.max(spans)
    points = points * scale
    points -= points.mean(axis=0)
    
    # Add small jitter to break symmetry
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points

def _bcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # BCC: 2 points per unit cell
    # 50 spheres -> 25 cells. 
    # 3x3x3 = 27 cells -> 54 points.
    m = 3
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    indices = rng.choice(len(points), size=n_spheres, replace=False)
    points = points[indices]
    
    spans = points.max(axis=0) - points.min(axis=0)
    scale = 0.9 / np.max(spans)
    points = points * scale
    points -= points.mean(axis=0)
    
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points

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
    # We will run this schedule, checking time between steps
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline - 5.0:
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
            # Use L-BFGS-B
            # Max iterations should be limited to avoid hanging on one step
            # 100-200 iterations per beta step is usually enough for convergence
            res = minimize(
                obj,
                C_curr.flatten(),
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 150,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
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
                C_curr = C_new # Still update to keep moving, even if slight dip, 
                               # as the landscape changes with beta
                               
        except Exception:
            pass
    
    return best_C, best_r

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Leave 40s margin for safety and overhead
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Jittered Grids (often very effective)
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        initializations.append(_jittered_grid_init(N, seed=seed, jitter=0.03))
        
    # 2. FCC Lattices
    for seed in [10, 11, 12]:
        initializations.append(_fcc_lattice_init(N, seed=seed))
        
    # 3. BCC Lattices
    for seed in [20, 21, 22]:
        initializations.append(_bcc_lattice_init(N, seed=seed))
        
    # 4. Random
    for seed in [30, 31]:
        initializations.append(_random_init(N, seed=seed))
    
    # Evaluate initializations and pick the best few
    init_radii = []
    for init in initializations:
        r = _exact_radius(init)
        init_radii.append(r)
    
    # Sort by radius, take top initializations
    sorted_indices = np.argsort(init_radii)[::-1]
    
    # We have ~17 initializations.
    # Let's pick the top 4 to optimize fully.
    top_k = 4
    
    best_C = None
    best_r = -np.inf
    
    # Allocate time per initialization
    # Total time for optimization: deadline - t_start - 10 (margin)
    time_per_init = (deadline - t_start - 10.0) / top_k
    
    for idx in sorted_indices[:top_k]:
        t_init_start = time.time()
        init_deadline = t_init_start + time_per_init
        
        C_init = initializations[idx]
        C_opt, r_opt = _optimize_from_init(C_init, init_deadline, best_r)
        
        if r_opt > best_r:
            best_r = r_opt
            best_C = C_opt
    
    # Final high-precision polish on global best
    if time.time() < deadline - 5.0 and best_C is not None:
        # Run a few more high-beta steps on the global best
        C_curr = best_C.copy()
        best_r_polish = _exact_radius(C_curr)
        
        # High beta steps
        for beta in [2000.0, 5000.0, 10000.0]:
            if time.time() > deadline - 2.0:
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
                res = minimize(
                    obj,
                    C_curr.flatten(),
                    jac=grad,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 100
                    }
                )
                C_new = res.x.reshape(-1, 3)
                C_new = np.clip(C_new, -0.5 + 1e-15, 0.5 - 1e-15)
                r_new = _exact_radius(C_new)
                
                if r_new > best_r_polish:
                    best_r_polish = r_new
                    C_curr = C_new
                else:
                    # If no improvement, break? Or keep going?
                    # If beta is very high, no improvement might mean we are stuck.
                    pass
            except Exception:
                pass
        
        if best_r_polish > best_r:
            best_r = best_r_polish
            best_C = C_curr
    
    # Ensure validity
    if best_C is None:
        best_C = _jittered_grid_init(N, seed=1)
    
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_C
