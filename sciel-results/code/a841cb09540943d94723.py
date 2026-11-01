"""Improved packing: Hybrid Force-Directed Relaxation and Smooth Surrogate Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """Compute the maximum common radius for given centers (exact, for evaluation)."""
    centers = np.asarray(centers)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    n = len(centers)
    if n < 2:
        return r_wall
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via log-sum-exp (negative log-sum-exp)."""
    values = np.asarray(values)
    m = np.min(values)
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(all constraints).
    """
    c = x.reshape(-1, 3)
    wall_constraints = 0.5 - np.abs(c).ravel()
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    s = _soft_min(all_constraints, beta)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    return -s + penalty

def _initial_grid_jittered():
    """Jittered grid initialization to break symmetry."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    centers = grid[perm[:N]].copy()
    # Add small jitter
    jitter = rng.uniform(-0.01, 0.01, size=centers.shape)
    centers += jitter
    # Clip to valid range
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _force_directed_relaxation(centers, max_iters=2000, deadline=None):
    """
    Structural change: Use repulsive forces to spread spheres.
    This is a gradient descent on a potential energy surface where
    energy increases as spheres get closer.
    """
    c = centers.copy()
    dt = 0.005
    min_dist_target = _score(c) * 2.0 # Current diameter
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        # We want to push apart pairs that are closer than the current min distance
        # Using a soft repulsion: F = 1/d^2 for d < threshold
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Threshold: slightly larger than current min distance to encourage uniformity
        threshold = min_dist_target * 1.2
        # Only apply force if dist < threshold
        mask = dists < threshold
        # Avoid division by zero
        dists_safe = np.where(mask, dists, 1.0)
        
        # Force magnitude: 1/d^2, direction: normalized diff
        # F_ij on i is (c_i - c_j) / d^2
        inv_d2 = np.where(mask, 1.0 / (dists_safe ** 2), 0.0)
        # Sum forces on each point
        # diff[i,j] is c_i - c_j
        # We want sum over j of (c_i - c_j) / d_ij^2
        # This is sum_j (diff[i,j] * inv_d2[i,j])
        forces += np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        # Push away from walls if close
        # Distance to wall: 0.5 - |c_ik|
        # Force: +sign(c_ik) * (1 / wall_dist^2) if wall_dist < threshold
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < 0.1 # Fixed threshold for walls
        wall_dists_safe = np.where(wall_mask, wall_dists, 1.0)
        wall_force_mag = np.where(wall_mask, 1.0 / (wall_dists_safe ** 2), 0.0)
        # Direction: away from wall, so sign of coordinate
        wall_forces = np.sign(c) * wall_force_mag
        forces += wall_forces
        
        # Update positions
        c += dt * forces
        
        # Keep within bounds
        c = np.clip(c, -0.499, 0.499)
        
        # Decay step size
        dt *= 0.995
        
        # Update min_dist_target occasionally to adapt
        if it % 50 == 0:
            current_score = _score(c)
            min_dist_target = current_score * 2.0
            if min_dist_target < 0.1: # Fallback
                min_dist_target = 0.1

    return c

def construct():
    t_start = time.time()
    deadline = t_start + 140  # Safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # 1. Generate initial candidates
    inits = [
        _initial_grid_jittered()
    ]
    
    # Add a few random inits for diversity
    rng = np.random.RandomState(123)
    for _ in range(2):
        rand_centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rand_centers)
        
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        cur = init.copy()
        cur_score = _score(cur)
        
        if best_centers is None or cur_score > best_score:
            best_centers = cur.copy()
            best_score = cur_score
            
        # 2. Structural Change: Force-Directed Relaxation
        # This spreads points out effectively
        relaxed = _force_directed_relaxation(cur, max_iters=500, deadline=deadline - 20)
        relaxed_score = _score(relaxed)
        
        if relaxed_score > best_score:
            best_centers = relaxed.copy()
            best_score = relaxed_score
            cur = relaxed.copy()
            
        # 3. Refinement: Smooth Surrogate with L-BFGS-B
        # Use the smooth objective to polish the result
        if time.time() > deadline - 15:
            continue
            
        for beta in [50.0, 100.0]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    _smooth_objective,
                    cur.ravel(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                if new_score > best_score:
                    best_centers = new_centers.copy()
                    best_score = new_score
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # 4. Final Polish: Exact Coordinate Descent
        if time.time() > deadline - 5:
            break
            
        cur = best_centers.copy()
        improved = True
        step = 0.001
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(N):
                for coord in range(3):
                    for delta in [step, -step, step*0.5, -step*0.5]:
                        new_c = cur.copy()
                        new_c[i, coord] += delta
                        if np.all(np.abs(new_c) <= 0.5):
                            s = _score(new_c)
                            if s > best_score + 1e-12:
                                cur = new_c
                                best_centers = new_c.copy()
                                best_score = s
                                improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break

    return best_centers
