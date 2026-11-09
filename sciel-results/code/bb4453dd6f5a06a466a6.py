"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def _compute_radius(centres):
    """Compute the largest common radius for given centres in [-0.5, 0.5]^2."""
    if centres.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds
    if np.any(centres < -0.5 - 1e-12) or np.any(centres > 0.5 + 1e-12):
        return 0.0
    # Check for coincident centres
    d = pdist(centres)
    if np.min(d) < 1e-12:
        return 0.0
    min_pair_dist = np.min(d)
    min_boundary_dist = np.min(0.5 - np.abs(centres))
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _smooth_objective(centres_flat, sigma=0.01):
    """Smooth objective: maximize the minimum of pair distances and boundary distances.
    We minimize the negative of a smooth approximation."""
    centres = centres_flat.reshape(N, 2)
    # Pairwise distances
    d = pdist(centres)
    # Boundary distances
    bdist = 0.5 - np.abs(centres)
    bdist_flat = bdist.flatten()
    
    # Use log-sum-exp for smooth minimum
    all_dists = np.concatenate([d, bdist_flat])
    
    # Smooth minimum: -log(sum(exp(-x/sigma))) * sigma
    # This approximates min(all_dists) when sigma is small
    neg_exp = -all_dists / sigma
    max_val = np.max(neg_exp)
    log_sum = max_val + np.log(np.sum(np.exp(neg_exp - max_val)))
    smooth_min = -sigma * log_sum
    
    return -smooth_min  # Negate because we minimize


def _gradient(centres_flat, sigma=0.01):
    """Compute gradient of the smooth objective."""
    centres = centres_flat.reshape(N, 2)
    
    # We'll use numerical gradient for simplicity, but let's try analytical
    # Actually, let's just use the finite difference approach via scipy
    pass


def _grid_init():
    """Create a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init():
    """Random initialization inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.95


def _annealing_step(centres, r_current, temp, cooling):
    """One step of simulated annealing."""
    new_centres = centres.copy()
    # Pick a random circle and move it
    idx = np.random.randint(N)
    new_centres[idx] += np.random.normal(0, temp, size=2)
    
    # Clamp to square
    new_centres[idx] = np.clip(new_centres[idx], -0.49, 0.49)
    
    r_new = _compute_radius(new_centres)
    
    if r_new >= r_current or np.random.random() < np.exp((r_new - r_current) / max(temp, 1e-10)):
        return new_centres, r_new, True
    return centres, r_current, False


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # Start with grid
    best_centres = _grid_init()
    best_r = _compute_radius(best_centres)
    
    # Phase 1: Local optimization from multiple starting points using L-BFGS-B
    rng = np.random.default_rng(42)
    
    # Try several random initializations
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        
        init_centres = (rng.random((N, 2)) - 0.5) * 0.95
        init_flat = init_centres.flatten()
        
        # Use L-BFGS-B to maximize the smooth objective
        try:
            result = minimize(
                _smooth_objective,
                init_flat,
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        opt_centres = result.x.reshape(N, 2)
        # Clamp to ensure validity
        opt_centres = np.clip(opt_centres, -0.49, 0.49)
        
        r = _compute_radius(opt_centres)
        if r > best_r:
            best_r = r
            best_centres = opt_centres.copy()
    
    # Phase 2: Simulated annealing for global exploration
    temp = 0.02
    cooling = 0.999
    current_centres = best_centres.copy()
    current_r = best_r
    
    while time.time() < deadline - 5:
        current_centres, current_r, accepted = _annealing_step(
            current_centres, current_r, temp, cooling
        )
        
        if current_r > best_r:
            best_r = current_r
            best_centres = current_centres.copy()
        
        temp *= cooling
        
        # If temperature gets too low, restart with higher temp occasionally
        if temp < 0.001:
            temp = 0.01
            # Perturb current solution
            current_centres = best_centres + np.random.normal(0, 0.005, size=(N, 2))
            current_centres = np.clip(current_centres, -0.49, 0.49)
            current_r = _compute_radius(current_centres)
    
    # Phase 3: Final local refinement from best solution
    if time.time() < deadline - 2:
        best_flat = best_centres.flatten()
        try:
            result = minimize(
                _smooth_objective,
                best_flat,
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            refined = result.x.reshape(N, 2)
            refined = np.clip(refined, -0.49, 0.49)
            r = _compute_radius(refined)
            if r > best_r:
                best_r = r
                best_centres = refined
        except Exception:
            pass
    
    return best_centres
# EVOLVE-BLOCK-END
