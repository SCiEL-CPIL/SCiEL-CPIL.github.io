"""
Optimization of 100 equal spheres in a unit cube.
Uses smooth surrogate (soft-min) formulation with L-BFGS-B to maximize minimum radius.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _adam_update(params, grads, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer update. Returns updated params, m, v, t."""
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v, t + 1

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_objective_and_grad(x, temperature=0.01):
    """
    Smooth surrogate: minimize -soft_min(all distances) where distances include
    pairwise/2 and wall distances. Uses log-sum-exp for numerical stability.
    Returns (objective_value, gradient_array).
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Compute all pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(dists_sq[i_idx, j_idx] + 1e-15) / 2.0
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all distances into one vector
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ -T * log(sum(exp(-a_i/T)))
    scaled = -all_dists / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Objective: -soft_min = T * (log(sum_exp) + max_scaled)
    obj = temperature * (np.log(sum_exp) + max_scaled)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise gradient contribution
    for k in range(len(i_idx)):
        w = weights[k]
        if w < 1e-10:
            continue
        i = i_idx[k]
        j = j_idx[k]
        d = pair_dists[k] * 2  # actual distance
        if d < 1e-10:
            continue
        vec = centers[i] - centers[j]
        factor = -w / (2 * d)
        grad[i] += factor * vec
        grad[j] -= factor * vec
    
    # Wall gradient contribution
    for i in range(n):
        w = weights[len(i_idx) + i]
        if w < 1e-10:
            continue
        max_val = max_abs[i]
        for k in range(3):
            if np.isclose(abs(centers[i, k]), max_val, atol=1e-10):
                grad[i, k] += w * np.sign(centers[i, k])
                break
    
    return obj, grad.flatten()

def _polish_objective_and_grad(x, active_pairs, active_walls):
    """
    High-precision polish: minimize -min over active constraints only.
    active_pairs: list of (i, j) tuples
    active_walls: list of (i, k) tuples where k is coordinate
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Collect active distances
    dists = []
    grads_info = []  # (index_in_centers, coord_or_pair, weight_info)
    
    for (i, j) in active_pairs:
        d = np.linalg.norm(centers[i] - centers[j]) / 2.0
        dists.append(d)
        grads_info.append(('pair', i, j, d))
    
    for (i, k) in active_walls:
        d = 0.5 - abs(centers[i, k])
        dists.append(d)
        grads_info.append(('wall', i, k, d))
    
    if not dists:
        return 0.0, np.zeros_like(x)
    
    min_d = np.min(dists)
    
    # Gradient: only the minimum constraint(s) contribute
    grad = np.zeros_like(centers)
    for info in grads_info:
        if info[0] == 'pair':
            _, i, j, d = info
            if np.isclose(d, min_d, atol=1e-12):
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist > 1e-10:
                    factor = -1.0 / (2 * dist)
                    grad[i] += factor * vec
                    grad[j] -= factor * vec
        else:  # wall
            _, i, k, d = info
            if np.isclose(d, min_d, atol=1e-12):
                grad[i, k] += np.sign(centers[i, k])
    
    return -min_d, grad.flatten()

def _adam_optimize(centers, max_iters=300, initial_temp=0.05, final_temp=0.001,
                   lr=0.005, deadline=None):
    """
    Adam gradient descent on soft-min objective with annealed temperature.
    Returns optimized centers.
    """
    x = centers.flatten().copy()
    n = len(x)
    
    m = np.zeros(n)
    v = np.zeros(n)
    t = 1
    
    # Anneal temperature from initial_temp to final_temp
    num_temps = 5
    temps = np.geomspace(initial_temp, final_temp, num_temps)
    
    for temp in temps:
        if deadline is not None and time.time() > deadline:
            break
        
        iters_for_temp = max(50, max_iters // num_temps)
        
        for it in range(iters_for_temp):
            if deadline is not None and time.time() > deadline:
                break
            
            obj, grad = _soft_min_objective_and_grad(x, temperature=temp)
            
            x, m, v, t = _adam_update(x, grad, m, v, t, lr=lr)
            
            # Project back to cube bounds
            centers_tmp = x.reshape(-1, 3)
            centers_tmp = np.clip(centers_tmp, -0.49999, 0.49999)
            x = centers_tmp.flatten()
        
        # Track best
        centers_current = x.reshape(-1, 3)
    
    return centers_current

def _polish_with_active_set(centers, deadline=None):
    """
    High-precision polish using active-set formulation with L-BFGS-B.
    """
    if deadline is not None and time.time() > deadline:
        return centers
    
    centers = centers.copy()
    n = N
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall = np.min(wall_dists)
    
    r_curr = min(min_pair, min_wall)
    
    # Active pairs: those close to the minimum
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    active_pairs = [(i_idx[k], j_idx[k]) for k in range(len(i_idx)) 
                    if pair_dists[k] < r_curr * 1.01 + 1e-6]
    
    # Active walls
    active_walls = []
    for i in range(n):
        if wall_dists[i] < r_curr * 1.01 + 1e-6:
            max_val = max_abs[i]
            for k in range(3):
                if np.isclose(abs(centers[i, k]), max_val, atol=1e-10):
                    active_walls.append((i, k))
                    break
    
    if not active_pairs and not active_walls:
        return centers
    
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    def polish_obj_grad(x):
        return _polish_objective_and_grad(x, active_pairs, active_walls)
    
    try:
        res = minimize(
            polish_obj_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
        )
        centers = res.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Second polish pass with updated active set
    if deadline is not None and time.time() < deadline:
        centers = centers.copy()
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        max_abs = np.max(np.abs(centers), axis=1)
        wall_dists = 0.5 - max_abs
        min_wall = np.min(wall_dists)
        r_curr = min(min_pair, min_wall)
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        active_pairs = [(i_idx[k], j_idx[k]) for k in range(len(i_idx)) 
                        if pair_dists[k] < r_curr * 1.005 + 1e-7]
        
        active_walls = []
        for i in range(n):
            if wall_dists[i] < r_curr * 1.005 + 1e-7:
                max_val = max_abs[i]
                for k in range(3):
                    if np.isclose(abs(centers[i, k]), max_val, atol=1e-10):
                        active_walls.append((i, k))
                        break
        
        if active_pairs or active_walls:
            def polish_obj_grad2(x):
                return _polish_objective_and_grad(x, active_pairs, active_walls)
            
            try:
                res = minimize(
                    polish_obj_grad2,
                    centers.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
                )
                centers = res.x.reshape(-1, 3)
            except Exception:
                pass
    
    return centers

def _fcc_init(n_spheres=100):
    """
    Generate FCC lattice initialization for n_spheres spheres in unit cube.
    FCC has 4 atoms per conventional cell.
    """
    # FCC basis vectors in a cubic cell of side 'a'
    # Atoms at: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    
    # We need to fit ~100 spheres. Try different lattice sizes.
    # FCC: n_atoms = 4 * nx * ny * nz
    # For 100: 4*5*5*1 = 100, or 4*4*4*2=128 (too many), 4*5*4*1=80 (too few)
    # Let's use 5x5x1 = 100 exactly
    
    best_centers = None
    best_r = 0
    
    for nx, ny, nz in [(5, 5, 1), (4, 5, 2), (5, 4, 2), (10, 5, 1), (5, 10, 1)]:
        total = 4 * nx * ny * nz
        if total < n_spheres:
            continue
        if total > n_spheres * 1.2:
            continue
        
        a = 1.0 / max(nx, ny, nz)  # lattice constant to fit in cube
        
        centers_list = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    base = np.array([ix, iy, iz]) * a
                    # FCC positions within cell
                    for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                        pos = base + np.array([dx, dy, dz]) * a
                        centers_list.append(pos)
        
        centers = np.array(centers_list)
        
        # Center and scale to fit in [-0.5, 0.5]^3
        centers = centers - np.mean(centers, axis=0)
        max_extent = np.max(np.abs(centers))
        if max_extent > 0:
            centers = centers / max_extent * 0.48
        
        # If we have more than n_spheres, randomly select
        if len(centers) > n_spheres:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(centers), size=n_spheres, replace=False)
            centers = centers[idx]
        
        # If we have fewer, add random points
        if len(centers) < n_spheres:
            rng = np.random.RandomState(42)
            extra = rng.uniform(-0.45, 0.45, size=(n_spheres - len(centers), 3))
            centers = np.vstack([centers, extra])
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers
    
    if best_centers is None:
        # Fallback: simple grid
        xs = np.linspace(-0.45, 0.45, 10)
        ys = np.linspace(-0.45, 0.45, 10)
        centers = np.array([[x, y, 0] for x in xs for y in ys[:10]])
        best_centers = centers[:n_spheres]
    
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations (limited to save time)
    inits = []
    
    # FCC lattice init
    fcc_centers = _fcc_init(N)
    inits.append(fcc_centers)
    
    # Grid init
    xs = np.linspace(-0.45, 0.45, 10)
    ys = np.linspace(-0.45, 0.45, 10)
    grid = np.array([[x, y, 0] for x in xs for y in ys])
    inits.append(grid[:N])
    
    # One random init for diversity
    rc = rng.uniform(-0.45, 0.45, size=(N, 3))
    inits.append(rc)
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        
        current_centers = init.copy()
        
        # Phase 1: Adam optimization with annealed soft-min temperature
        adam_deadline = deadline - 15.0
        current_centers = _adam_optimize(
            current_centers,
            max_iters=200,
            initial_temp=0.05,
            final_temp=0.0005,
            lr=0.003,
            deadline=adam_deadline
        )
        
        r_curr = _compute_radius(current_centers)
        if r_curr > best_radius:
            best_radius = r_curr
            best_centers = current_centers.copy()
        
        # Phase 2: L-BFGS-B soft-min refinement (short)
        if time.time() < deadline - 10.0:
            bounds = [(-0.5, 0.5)] * (N * 3)
            # Use a single short L-BFGS pass with a moderate temperature
            temp = 0.01
            def obj_grad(x):
                return _soft_min_objective_and_grad(x, temperature=temp)
            
            try:
                res = minimize(
                    obj_grad,
                    current_centers.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'disp': False, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                current_centers = res.x.reshape(-1, 3)
            except Exception:
                pass
            
            r_curr = _compute_radius(current_centers)
            if r_curr > best_radius:
                best_radius = r_curr
                best_centers = current_centers.copy()
        
        # Phase 3: Active-set polish
        if time.time() < deadline - 5.0:
            current_centers = _polish_with_active_set(current_centers, deadline=deadline - 2.0)
            
            r_curr = _compute_radius(current_centers)
            if r_curr > best_radius:
                best_radius = r_curr
                best_centers = current_centers.copy()
    
    # Final local search perturbations if time remains
    if best_centers is not None and time.time() < deadline - 5.0:
        for attempt in range(3):
            if time.time() > deadline - 3.0:
                break
            
            perturbed = best_centers + rng.normal(0, 0.002, size=(N, 3))
            perturbed = np.clip(perturbed, -0.4999, 0.4999)
            
            # Quick Adam refinement
            perturbed = _adam_optimize(
                perturbed,
                max_iters=100,
                initial_temp=0.01,
                final_temp=0.001,
                lr=0.002,
                deadline=deadline - 2.0
            )
            
            # Quick polish
            perturbed = _polish_with_active_set(perturbed, deadline=deadline - 1.0)
            
            r_pert = _compute_radius(perturbed)
            if r_pert > best_radius:
                best_radius = r_pert
                best_centers = perturbed.copy()
    
    # Final safety check
    if best_centers is None:
        best_centers = inits[0]
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        
    return best_centers
# EVOLVE-BLOCK-END
