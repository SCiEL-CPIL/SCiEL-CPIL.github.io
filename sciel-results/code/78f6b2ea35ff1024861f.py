"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

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
    # Distance to wall is 0.5 - max(|x|, |y|, |z|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # pdist computes condensed distances, shape (N*(N-1)/2,)
    dists = pdist(centers)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _penalty(x, R):
    """
    Calculates the sum of squared violations of the distance constraints for radius R.
    x: (N, 3) array of centers.
    R: target radius.
    """
    centers = x
    n = len(centers)
    
    # 1. Wall violations
    # distance to wall = 0.5 - max(|x|, |y|, |z|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    wall_violations = np.maximum(0, R - wall_dists)
    penalty_wall = np.sum(wall_violations**2)
    
    # 2. Pairwise violations
    # dists is condensed, shape (N*(N-1)/2,)
    dists = pdist(centers)
    # violation if dist < 2R
    pair_violations = np.maximum(0, 2*R - dists)
    penalty_pair = np.sum(pair_violations**2)
    
    return penalty_wall + penalty_pair

def _sa_optimize(init_centers, target_R, deadline, max_iter=50000):
    """
    Optimizes the configuration to satisfy radius target_R using Simulated Annealing.
    Returns the best centers found.
    """
    current_centers = init_centers.copy()
    current_penalty = _penalty(current_centers, target_R)
    
    best_centers = current_centers.copy()
    best_penalty = current_penalty
    
    # SA parameters
    T = 0.01  # Initial temperature
    cooling_rate = 0.9995
    step_size = 0.05
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Propose a move: perturb a random subset of points
        # Usually perturbing all points with small noise or a few points with larger noise
        # Here we perturb all points with small Gaussian noise
        # We scale step_size by T to allow larger moves at higher T
        move = rng.normal(0, step_size * T, size=(N, 3))
        proposed_centers = current_centers + move
        
        # Keep points inside the cube (hard constraint for validity, but penalty handles it softly)
        # However, for SA efficiency, it's good to stay in bounds
        proposed_centers = np.clip(proposed_centers, -0.5, 0.5)
        
        proposed_penalty = _penalty(proposed_centers, target_R)
        
        delta = proposed_penalty - current_penalty
        
        # Metropolis criterion
        if delta < 0 or (T > 1e-8 and rng.random() < np.exp(-delta / T)):
            current_centers = proposed_centers
            current_penalty = proposed_penalty
            
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_centers = current_centers.copy()
                
                # If we found a perfect solution, we can stop or tighten
                if best_penalty < 1e-12:
                    # We could break here, but maybe we want to polish?
                    # For SA, breaking is fine, the caller will increase R
                    break
        
        T *= cooling_rate
        
    return best_centers, best_penalty

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial setup
    # Start with a grid to ensure a valid starting point
    # 5x4x5 grid = 100 points
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Add small noise to break symmetry
    rng = np.random.default_rng(42)
    best_pos = grid + rng.normal(0, 0.001, size=(N, 3))
    best_pos = np.clip(best_pos, -0.499, 0.499)
    
    # Calculate initial radius
    best_r = _compute_radius(best_pos)
    
    # If grid is bad, try random
    if best_r < 0.1:
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)
        
    # Iterative Radius Increase with SA
    current_r = best_r
    step_size = 0.0005
    max_failures = 10
    failures = 0
    
    # We will run SA multiple times, increasing R gradually
    # To speed up, we can reduce max_iter as R gets closer to optimum
    # or just let the deadline handle it.
    
    while time.time() < deadline - 5.0:
        target_r = current_r + step_size
        
        # Run SA to try to achieve target_r
        # We use the current best position as start
        opt_pos, penalty = _sa_optimize(best_pos, target_r, deadline, max_iter=10000)
        
        # Check if the optimized position is valid for target_r
        # We need to check if the actual radius is at least target_r
        actual_r = _compute_radius(opt_pos)
        
        if actual_r >= target_r - 1e-7:
            # Success
            best_pos = opt_pos
            current_r = actual_r
            failures = 0
            # Optionally increase step size slightly if we are very fast
            # step_size = min(step_size * 1.1, 0.002)
        else:
            # Failure to reach target_r
            failures += 1
            # Try to recover by running SA on current_r again? 
            # Or just reduce step size.
            if failures > max_failures:
                step_size *= 0.8
                failures = 0
                if step_size < 1e-7:
                    break
            
    # Final Polish: 
    # Use the best position found.
    # We can do a final local optimization using L-BFGS-B on the soft-min or penalty
    # to squeeze out the last few bits of radius.
    # However, SA with small step sizes at the end acts as a local optimizer.
    # Let's do one more SA pass with very small step size and low T to polish.
    
    if time.time() < deadline - 2.0:
        # Polish
        # Run SA with current_r, but with very small step size to fine-tune
        # Actually, _sa_optimize already cools down.
        # Let's just ensure we return the best valid position.
        
        # One more check: try to increase radius by a tiny amount with a very short SA run
        tiny_step = 0.0001
        target_r = current_r + tiny_step
        opt_pos, penalty = _sa_optimize(best_pos, target_r, deadline, max_iter=5000)
        actual_r = _compute_radius(opt_pos)
        if actual_r > current_r:
            best_pos = opt_pos
            current_r = actual_r

    # Ensure validity
    best_pos = np.clip(best_pos, -0.499999, 0.499999)
    
    return best_pos
# EVOLVE-BLOCK-END
