"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _soft_min_radius(centers, target_r, beta=100.0):
    """
    Smooth approximation of the violation of the radius constraint.
    We want to maximize the minimum distance d_min.
    Equivalent to minimizing max(0, target_r - d_min).
    Using soft-min: -1/beta * log(sum(exp(-beta * d_i))) approximates min(d_i).
    So we want to maximize -1/beta * log(sum(exp(-beta * d_i))).
    Or minimize 1/beta * log(sum(exp(-beta * d_i))) + penalty for walls.
    
    Actually, let's minimize the "violation" energy:
    E = sum_{i<j} softplus(beta * (2*target_r - d_ij)) / beta
      + sum_i softplus(beta * (target_r - w_i)) / beta
      
    where softplus(x) = log(1+exp(x)).
    This is 0 when constraints are satisfied and grows linearly when violated.
    """
    n = len(centers)
    if n < 2:
        wall_dists = 0.5 - np.abs(centers)
        wall_viol = np.maximum(0, target_r - wall_dists)
        return np.sum(wall_viol)
        
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider upper triangle to avoid double counting and self-interaction
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu]
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Soft violation for pairs: want d_ij >= 2*target_r
    # Violation v = 2*target_r - d_ij
    # Penalty = softplus(beta * v) / beta
    pair_viol = 2.0 * target_r - pair_dists
    pair_arg = np.clip(beta * pair_viol, -500, 500)
    pair_energy = np.sum(np.logaddexp(0, pair_arg)) / beta
    
    # Soft violation for walls: want w_i >= target_r
    # Violation v = target_r - w_i
    # Penalty = softplus(beta * v) / beta
    wall_viol = target_r - wall_dists
    wall_arg = np.clip(beta * wall_viol, -500, 500)
    wall_energy = np.sum(np.logaddexp(0, wall_arg)) / beta
    
    return pair_energy + wall_energy


def _true_radius(centers):
    """Compute the exact maximum common radius."""
    n = len(centers)
    if n < 2:
        return np.min(0.5 - np.abs(centers))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _grid_init(n):
    """Generate a grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def _random_init(n, seed=None):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def _optimize_radius(centers_init, target_r, beta, max_iter, time_limit, start_time):
    """
    Optimize centers to satisfy radius target_r using L-BFGS-B.
    """
    if time.time() > start_time + time_limit:
        return centers_init, _true_radius(centers_init)
    
    def objective(x):
        c = x.reshape(-1, 3)
        return _soft_min_radius(c, target_r, beta)
        
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    try:
        res = minimize(
            objective,
            centers_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        opt_centers = res.x.reshape(-1, 3)
    except Exception:
        opt_centers = centers_init
    
    # Check time again
    if time.time() > start_time + time_limit:
        return centers_init, _true_radius(centers_init)
        
    r = _true_radius(opt_centers)
    return opt_centers, r


def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial setup
    best_centers = _grid_init(N)
    best_radius = _true_radius(best_centers)
    
    # Also try random starts to pick a better initial basin
    for seed in range(5):
        if time.time() > deadline - 20.0:
            break
        c = _random_init(N, seed)
        r = _true_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
            
    # Ramping strategy
    # Start from a low radius and increase
    # Best known is ~0.11138
    # Grid radius is roughly 1/(2*5) = 0.1
    
    current_r = best_radius
    # We will try to push the radius up
    # Let's define a sequence of targets
    # Or simply iteratively increase the target and optimize
    
    # Set initial target slightly above current best to force improvement
    target_r = best_radius * 1.01
    
    beta = 50.0
    max_iter = 500
    
    while time.time() < deadline - 10.0:
        # Increase beta for sharper approximation as we get closer to optimum?
        # Or keep it moderate to stay smooth.
        
        # Try to optimize for current target_r
        centers, r = _optimize_radius(
            best_centers, target_r, beta, max_iter, 
            deadline - time.time() - 5.0, start_time
        )
        
        # Update best if improved
        if r > best_radius:
            best_radius = r
            best_centers = centers
            # If we improved, we can try a slightly higher target
            # Increase target by a small amount
            target_r = best_radius * 1.005
            # Maybe increase beta slightly to tighten the constraint
            beta = min(beta * 1.1, 200.0)
        else:
            # If no improvement, we might be stuck.
            # Try a different perturbation or increase target more aggressively?
            # Or try to reduce beta to smooth out the landscape?
            # Let's try perturbing the current best solution
            rng = np.random.RandomState(int(time.time()*1000) % 10000)
            perturbed = best_centers + rng.randn(N, 3) * 0.005
            perturbed = np.clip(perturbed, -0.4999, 0.4999)
            
            centers_p, r_p = _optimize_radius(
                perturbed, target_r, beta, max_iter,
                deadline - time.time() - 5.0, start_time
            )
            
            if r_p > best_radius:
                best_radius = r_p
                best_centers = centers_p
                target_r = best_radius * 1.005
            else:
                # Stuck. Increase target more to force escape?
                # Or break if time is tight
                if time.time() > deadline - 20.0:
                    break
                target_r = best_radius * 1.01
                # Lower beta to allow smoother escape
                beta = max(beta * 0.8, 10.0)
                
    # Final polish with high beta and tight tolerance
    if time.time() < deadline - 2.0:
        final_target = best_radius * 1.001
        centers_f, r_f = _optimize_radius(
            best_centers, final_target, 200.0, 1000,
            deadline - time.time() - 1.0, start_time
        )
        if r_f > best_radius:
            best_radius = r_f
            best_centers = centers_f
            
    # Ensure validity
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
