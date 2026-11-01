import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_energy(pos, r, beta=100.0):
    """
    Compute the smooth energy for a given configuration and radius.
    pos: (N, 2) array of centers.
    r: radius.
    beta: sharpness of the log-sum-exp approximation.
    """
    pos = np.asarray(pos).reshape(N, 2)
    energy = 0.0
    
    # Pairwise overlaps
    # d_ij is distance between i and j
    # We want to penalize d_ij < 2r
    # Using log-sum-exp: E_pair = (1/beta) * sum(log(1 + exp(beta * (2r - d_ij))))
    # To avoid overflow, use the trick: log(1+exp(x)) = max(0,x) + log1p(exp(-|x|))
    
    # Vectorized calculation of distances
    # This might be memory heavy for N=50? 50x50 matrix is small.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf) # Ignore self
    
    # Overlap amount: 2r - dist
    overlap = 2 * r - dist
    # Soft penalty
    # For large positive overlap, log(1+exp(beta*overlap)) ~ beta*overlap
    # For large negative, ~ 0
    # We can just sum the soft-max
    # To be safe against overflow, we can clip or use the stable form.
    # stable_log1pexp(x) = np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    
    x = beta * overlap
    # Only compute for upper triangle to save time? No, full matrix is fine for N=50.
    # But we need to be careful with inf.
    x = np.where(np.isinf(x), 0, x) # dist inf -> overlap -inf -> x -inf -> log(1+exp(-inf))=0. Correct.
    
    # Stable log1pexp
    log1pexp_x = np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    energy += np.sum(log1pexp_x) / beta
    
    # Wall constraints
    # x_i must be in [-0.5 + r, 0.5 - r]
    # Penalty if x_i < L or x_i > R
    L = -0.5 + r
    R = 0.5 - r
    
    # x coords
    x_pen = np.maximum(0, L - pos[:, 0]) + np.maximum(0, pos[:, 0] - R)
    # Smooth it? The max(0, z) is non-smooth.
    # Use soft-plus: sp(z) = log(1+exp(beta*z))/beta
    # z1 = L - x_i, z2 = x_i - R
    z1 = beta * (L - pos[:, 0])
    z2 = beta * (pos[:, 0] - R)
    log1pexp_z1 = np.where(z1 > 0, z1 + np.log1p(np.exp(-z1)), np.log1p(np.exp(z1)))
    log1pexp_z2 = np.where(z2 > 0, z2 + np.log1p(np.exp(-z2)), np.log1p(np.exp(z2)))
    energy += np.sum(log1pexp_z1 + log1pexp_z2) / beta
    
    # y coords
    z3 = beta * (L - pos[:, 1])
    z4 = beta * (pos[:, 1] - R)
    log1pexp_z3 = np.where(z3 > 0, z3 + np.log1p(np.exp(-z3)), np.log1p(np.exp(z3)))
    log1pexp_z4 = np.where(z4 > 0, z4 + np.log1p(np.exp(-z4)), np.log1p(np.exp(z4)))
    energy += np.sum(log1pexp_z3 + log1pexp_z4) / beta
    
    return energy

def is_valid(pos, r, tol=1e-5):
    """Check if configuration is strictly valid (no overlaps, inside walls)."""
    pos = np.asarray(pos).reshape(N, 2)
    # Check bounds
    L = -0.5 + r
    R = 0.5 - r
    if np.any(pos[:, 0] < L - tol) or np.any(pos[:, 0] > R + tol):
        return False
    if np.any(pos[:, 1] < L - tol) or np.any(pos[:, 1] > R + tol):
        return False
    
    # Check overlaps
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 2 * r - tol:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    # Calculate initial radius
    def calc_radius(p):
        p = np.asarray(p).reshape(N, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        min_wall = np.min(np.minimum(np.minimum(p[:,0], p[:,1]), 
                                     np.minimum(0.5 - p[:,0], 0.5 - p[:,1])))
        return min(min_dist / 2.0, min_wall)
        
    r_current = calc_radius(pos)
    r_max_target = 0.071377103865 # Known best
    r_min = r_current
    
    best_pos = pos.copy()
    best_r = r_current
    
    # Continuation method: increase r slowly
    # Start with a small step
    delta_r = 0.001
    beta = 50.0 # Sharpness of potential
    
    # We will try to increase r. 
    # If optimization fails, we reduce beta (smooth out) or try a different start.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        r_try = r_current + delta_r
        if r_try > r_max_target:
            r_try = r_max_target
            
        # Objective function for optimizer
        def objective(x):
            return compute_energy(x, r_try, beta)
        
        # Try optimization
        # Use L-BFGS-B
        # Bounds: we can use soft bounds, but L-BFGS-B supports hard bounds.
        # Hard bounds: [-0.5+r_try, 0.5-r_try]
        # Note: If r_try is too large, bounds become invalid (L > R).
        if -0.5 + r_try > 0.5 - r_try:
             # r_try > 0.25, impossible for 50 circles
             break
             
        bounds = [(-0.5 + r_try, 0.5 - r_try)] * (2 * N)
        
        try:
            res = minimize(objective, pos, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Check if it converged to a valid state
            if is_valid(res.x, r_try, tol=1e-4):
                # Success! Update current state
                pos = res.x
                r_current = r_try
                best_pos = pos.copy()
                best_r = r_current
                # Increase step size slightly? Or keep it.
                # delta_r = min(delta_r * 1.1, 0.002) # Maybe keep constant or increase slowly
                delta_r = min(delta_r * 1.05, 0.005)
            else:
                # Failure. Try to reduce beta to make landscape smoother
                if beta > 10.0:
                    beta = max(10.0, beta * 0.8)
                else:
                    # If beta is low and still failing, try a smaller step
                    delta_r = delta_r * 0.5
                    # Also, try restarting from a slightly perturbed position
                    # pos = pos + np.random.normal(0, 0.001, size=pos.shape)
                    # But we must ensure validity for the *current* r_current first?
                    # The optimizer might have moved to a bad spot.
                    # Revert to last known good pos for next attempt?
                    # pos is already overwritten by res.x in minimize? No, res.x is new.
                    # We should keep pos as the last valid one if this attempt failed.
                    # So we don't update pos on failure.
                    pass
                    
        except Exception as e:
            # If optimizer crashes, reduce step
            delta_r = delta_r * 0.5
            if delta_r < 1e-5:
                break

    return best_pos
