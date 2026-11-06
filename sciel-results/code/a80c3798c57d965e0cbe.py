"""
Strategy: Structured Parameterization + Simulated Annealing.
Instead of optimizing 16 free coordinates, we parameterize the 8 points 
as perturbations of the regular octagon:
p_k = r_k * [cos(theta_k), sin(theta_k)]
where r_k = 1 + dr_k and theta_k = 2*pi*k/8 + dtheta_k.

This reduces the search space to 16 parameters (8 radii, 8 angles) but 
constrains them to be "near" the octagon initially, and the structured 
nature helps the optimizer. We use Simulated Annealing to explore this 
space, which is robust to local minima and can find non-trivial deformations.

The regular octagon corresponds to dr=0, dtheta=0.
We allow dr and dtheta to vary. To avoid points collapsing or becoming 
too far, we bound the parameters.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _score(points):
    """
    Compute Delta / n^n for 8 points.
    Delta is the product of ordered pair distances after scaling to diameter 2.
    """
    n = len(points)
    pts = np.array(points, dtype=float)
    
    if pts.shape != (8, 2):
        return 0.0
        
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    
    # Scale to diameter 2
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = dists[mask]
    
    # Check for distinct points
    if np.any(ordered_dists < 1e-12):
        return 0.0
        
    # Scale distances
    scaled_dists = ordered_dists / diam * 2.0
    
    # Product of ordered pair distances
    # log(Delta) = sum(log(scaled_dists))
    log_delta = np.sum(np.log(scaled_dists))
    
    # Score = Delta / n^n
    # log(Score) = log_delta - n*log(n)
    log_score = log_delta - n * np.log(n)
    
    return np.exp(log_score)

def _params_to_points(params):
    """
    Convert 16 parameters (8 dr, 8 dtheta) to 8 points.
    params[0:8] are dr (radial perturbations)
    params[8:16] are dtheta (angular perturbations)
    """
    n = 8
    dr = params[0:n]
    dtheta = params[n:2*n]
    
    # Base angles for regular octagon
    base_angles = 2 * np.pi * np.arange(n) / n
    
    # Final angles
    angles = base_angles + dtheta
    
    # Final radii (base radius is 1.0)
    radii = 1.0 + dr
    
    # Ensure radii are positive to avoid collapse to center
    # If a radius becomes too small, points might get too close to each other
    # or to the origin, which is fine as long as they are distinct.
    # However, if radius is negative, the point flips, which is equivalent to 
    # angle + pi and positive radius. 
    # To keep the mapping smooth and avoid discontinuities, we clip radii to a small positive value.
    # Actually, allowing negative radii is mathematically fine for the geometry, 
    # but let's stick to positive radii to maintain the "star-shaped" structure 
    # relative to the origin, which is a reasonable assumption for this problem.
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    pts = np.column_stack((x, y))
    return pts

def _score_from_params(params):
    pts = _params_to_points(params)
    return _score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    n_params = 16
    
    # Initial best: regular octagon
    best_params = np.zeros(n_params)
    best_score = _score_from_params(best_params)
    
    # Simulated Annealing parameters
    # We start with a "temperature" that allows large moves and cool down.
    # Move size is controlled by temperature.
    
    # Initial temperature
    T = 0.1
    # Cooling rate
    alpha = 0.9995
    
    # Number of iterations
    # We will run until time limit or until T is very small
    current_params = np.zeros(n_params)
    current_score = _score_from_params(current_params)
    
    rng = np.random.RandomState(42)
    
    # We will use a simple SA loop
    # Step size is proportional to T
    # We perturb each parameter independently
    
    while time.time() < deadline - 5.0:
        # Generate a new candidate by perturbing current params
        # Perturbation size is T
        step_size = T * 0.5
        delta = rng.normal(0, step_size, n_params)
        
        # We might want to restrict the range of parameters to avoid 
        # exploring too far from the origin where the structure breaks down.
        # dr in [-0.5, 0.5] -> radii in [0.5, 1.5]
        # dtheta in [-0.5, 0.5] -> angle shift up to ~30 degrees
        # Let's clip the parameters to a reasonable range
        # This effectively creates a bounded search space for the perturbations
        new_params = current_params + delta
        
        # Clip parameters to avoid extreme values
        # dr: [-0.5, 0.5]
        new_params[0:8] = np.clip(new_params[0:8], -0.5, 0.5)
        # dtheta: [-0.5, 0.5] radians
        new_params[8:16] = np.clip(new_params[8:16], -0.5, 0.5)
        
        new_score = _score_from_params(new_params)
        
        # Acceptance criterion
        if new_score > current_score:
            current_params = new_params
            current_score = new_score
        else:
            # Metropolis criterion
            # delta_score = new_score - current_score (negative)
            # Accept with probability exp(delta_score / T)
            # Since scores are positive and we want to maximize,
            # we can use the ratio or difference. 
            # Using difference in log-space is often more stable.
            # Let's use log scores for acceptance to avoid issues with small scores.
            # log(new) - log(current)
            if current_score > 0 and new_score > 0:
                log_ratio = np.log(new_score) - np.log(current_score)
                # log_ratio is negative if new < current
                accept_prob = np.exp(log_ratio / T)
                if rng.random() < accept_prob:
                    current_params = new_params
                    current_score = new_score
            else:
                # If scores are 0 or very small, reject
                pass
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
        
        # Cool down
        T *= alpha
        
        # If T is very small, we can stop or reset T for a restart
        if T < 1e-6:
            # Restart with a new random point or keep cooling?
            # Let's do a few restarts if we have time
            # Restart with a random configuration near the best found
            if time.time() < deadline - 20.0:
                # Perturb the best params
                restart_step = 0.1
                current_params = best_params + rng.normal(0, restart_step, n_params)
                current_params[0:8] = np.clip(current_params[0:8], -0.5, 0.5)
                current_params[8:16] = np.clip(current_params[8:16], -0.5, 0.5)
                current_score = _score_from_params(current_params)
                T = 0.05 # Reset temperature to a lower value
            else:
                break

    # Final local refinement using Nelder-Mead on the structured parameters
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            def _neg_score_params(params):
                return -_score_from_params(params)
            
            # Use the best params as start
            # Bounds for minimize? Nelder-Mead doesn't use bounds directly, 
            # but we can penalize out-of-bounds or just rely on the clipping in _params_to_points?
            # Actually, _params_to_points clips radii but not angles.
            # Let's add a penalty to the objective function if params go out of the desired range
            # to keep the optimizer in the structured region.
            
            def _penalized_neg_score(params):
                s = _score_from_params(params)
                # Penalty for large parameters
                penalty = 0.0
                if np.any(np.abs(params[0:8]) > 0.5):
                    penalty += 100.0 * np.sum((np.abs(params[0:8]) - 0.5)**2)
                if np.any(np.abs(params[8:16]) > 0.5):
                    penalty += 100.0 * np.sum((np.abs(params[8:16]) - 0.5)**2)
                return -s + penalty
            
            res = minimize(
                _penalized_neg_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if -res.fun > best_score:
                # Verify the score without penalty
                refined_score = _score_from_params(res.x)
                if refined_score > best_score:
                    best_score = refined_score
                    best_params = res.x
        except Exception:
            pass

    # Convert best params to points
    points = _params_to_points(best_params)
    
    # Sanity check: distinct points
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        # Fallback to regular octagon
        octagon = np.array([[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)])
        points = octagon
        
    return [[float(p[0]), float(p[1])] for p in points]
# EVOLVE-BLOCK-END
