import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Precompute indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def _compute_radius(c):
    """
    Compute the exact largest common radius for centers c (N, 3).
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, beta):
    """
    Compute softmin radius and its gradient using log-sum-exp.
    beta controls the sharpness of the approximation.
    """
    diff = c[_iu] - c[_ju]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(dists_sq)
    
    # Avoid division by zero
    dists = np.where(dists < 1e-12, 1e-12, dists)
    
    pair_vals = dists / 2.0
    wall_vals = 0.5 - np.abs(c)
    
    # Log-sum-exp for stability
    # We want: -1/beta * log( sum(exp(-beta * val)) )
    # val can be pair_vals or wall_vals
    
    # Combine all constraints
    # Pair constraints: N_pairs values
    # Wall constraints: N*3 values
    
    # To avoid large arrays, we can compute log-sum-exp separately and combine?
    # log(A + B) = max(A, B) + log(exp(A-max) + exp(B-max))
    # But here we have sums of many terms.
    
    # Let's just concatenate for simplicity if memory allows (1225 + 150 = 1375 floats). It's fine.
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    
    # For gradient, we need weights w_k = exp(-beta * val_k) / sum(exp(-beta * val_j))
    
    # Shift for stability
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_shifted = np.exp(-beta * shifted)
    
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
         return min_val, np.zeros_like(c)
    
    # Softmin value
    softmin_val = min_val - (1.0/beta) * np.log(sum_exp)
    
    # Weights
    weights = exp_shifted / sum_exp
    
    # Split weights back
    w_pairs = weights[:_NUM_PAIRS]
    w_walls = weights[_NUM_PAIRS:]
    
    grad_c = np.zeros_like(c)
    
    # Pair gradient contribution
    # grad c_i gets -0.5 * w * (c_i - c_j) / d_ij
    # grad c_j gets +0.5 * w * (c_i - c_j) / d_ij
    
    unit_vecs = diff / dists[:, np.newaxis]
    
    # Vectorized gradient accumulation
    # pair_grad[i] -= 0.5 * w_pairs[i] * unit_vecs[i]
    # pair_grad[j] += 0.5 * w_pairs[i] * unit_vecs[i]
    
    contrib = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    # Use np.add.at for efficient scattering
    # Note: np.add.at is slower than a simple loop for very small N, but here N=50, pairs=1225.
    # Actually, a simple loop in Python might be slower than np.add.at?
    # Let's try to avoid np.add.at if possible, or use it correctly.
    # For 1225 pairs, a Python loop is ~1225 iterations. np.add.at is C-level but has overhead.
    # Let's stick to the loop for clarity unless it's a bottleneck, but the prompt asks for structural change/speed.
    # Let's use np.add.at.
    
    np.add.at(grad_c, _iu, -contrib)
    np.add.at(grad_c, _ju, contrib)
    
    # Wall gradient contribution
    # grad c_ik += w_walls[i*3+k] * sign(c_ik)
    wall_weights_reshaped = w_walls.reshape(N, 3)
    signs = np.sign(c)
    grad_c += wall_weights_reshaped * signs
    
    return softmin_val, grad_c

def _polish(c_init, deadline, beta_start=100, beta_end=1000, n_steps=10):
    """
    Polish the configuration using L-BFGS-B on the soft-min objective.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        def obj(x):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), beta)
            return -r
        
        def grad(x):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), beta)
            return g.ravel()
            
        x0 = c.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

def _gradient_sa(c_init, deadline, t_start=0.05, t_end=0.001, n_steps=1000):
    """
    Gradient-based Simulated Annealing.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.geomspace(t_start, t_end, 20)
    steps_per_temp = max(1, n_steps // len(temps))
    
    for t in temps:
        if time.time() > deadline:
            break
        
        for step in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Compute gradient of soft-min (using a moderate beta for gradient, e.g., 50)
            # Using a fixed beta for SA gradient helps guide the search
            r_soft, grad = _softmin_radius_and_grad(c, 50.0)
            
            # Propose move: gradient step + noise
            # Step size scales with temperature
            step_size = t * 0.1
            noise = np.random.normal(0, step_size, size=c.shape)
            
            c_new = c + step_size * grad + noise
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            # Metropolis acceptance
            # We accept if r_new > r_old or with prob exp(-(r_old - r_new)/t)
            # Note: we are maximizing r, so delta = r_new - r_old
            # If delta < 0, accept with exp(delta/t)
            delta_r = r_new - _compute_radius(c)
            
            if delta_r >= 0 or np.random.random() < np.exp(delta_r / t):
                c = c_new
                
    return best_c, best_r

def _seed_grid():
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # 4x4x4 = 64 points, pick 50
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_fcc():
    """
    Generate an FCC-inspired initial configuration.
    """
    m = 3
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                # Add FCC offsets
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points

def _seed_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_c = None
    best_r = 0.0
    
    # Define seeds
    seeds = [
        _seed_grid(),
        _seed_fcc(),
        _seed_random(),
        _seed_random(),
        _seed_grid()
    ]
    
    # Allocate time per seed
    time_per_seed = (deadline - start_time) / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - 5.0)
        
        if time.time() > seed_deadline:
            continue
            
        c_current = s.copy()
        
        # Stage 1: Gradient SA
        c_sa, r_sa = _gradient_sa(c_current, seed_deadline, t_start=0.02, t_end=0.001, n_steps=500)
        if r_sa > best_r:
            best_r = r_sa
            best_c = c_sa.copy()
        c_current = c_sa
        
        # Stage 2: Polish
        c_pol, r_pol = _polish(c_current, seed_deadline, beta_start=50, beta_end=500, n_steps=8)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol.copy()
            
    # Final polish on global best if time remains
    if best_c is not None and time.time() < deadline - 5.0:
        final_deadline = deadline - 2.0
        c_final, r_final = _polish(best_c, final_deadline, beta_start=100, beta_end=2000, n_steps=10)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
    if best_c is None:
        return _seed_grid()
        
    return best_c
