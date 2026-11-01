"""
Strategy: Simulated Annealing with Push-Apart forces.
Instead of gradient descent on a smooth surrogate, we use a stochastic search
that directly targets the minimum distance constraint. This allows the system
to 'break' symmetric traps by accepting temporary overlaps (if the objective
allows, or via the force model naturally pushing them apart) and finding
denser, asymmetric packings.
"""

import numpy as np
import time

N = 50
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum radius for the given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -CUBE_HALF) or np.any(centers > CUBE_HALF):
        return 0.0
        
    # Pairwise distances
    # Using broadcasting for pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist, min_wall_dist) / 2.0
    return r

def _seed_fcc():
    """Generate an FCC-like lattice structure truncated to 50 points."""
    # FCC lattice points are at (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # scaled appropriately.
    # Let's generate a dense grid and pick the best 50, or just use a standard grid 
    # that is closer to optimal aspect ratios.
    # Actually, for N=50, a 5x5x2 grid is the "standard" seed, but FCC is denser.
    # Let's create a 4x4x4 FCC-like set (32 points) and add more?
    # Simpler: Use a high-density grid and let SA fix it.
    
    # Let's use a "Body Centered Cubic" style or just a dense grid.
    # A 5x5x2 grid is the current seed. Let's try a 4x4x4 grid (64 points) 
    # and randomly select 50? No, that might be messy.
    
    # Better: Generate points on a fine grid and select 50 that maximize min-dist?
    # That's an optimization problem itself.
    
    # Let's just use a slightly perturbed 5x5x2 grid as one seed,
    # and a random seed as another.
    
    xs = -0.5 + (np.arange(5) + 0.5) * 0.2
    ys = -0.5 + (np.arange(5) + 0.5) * 0.2
    zs = -0.5 + (np.arange(2) + 0.5) * 0.5
    
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _seed_random():
    """Generate random points in the cube."""
    return np.random.uniform(-0.45, 0.45, (N, 3))

def _seed_bcc():
    """Generate a BCC-like structure."""
    # BCC: corners and center of cubes.
    # Let's just generate a 3x3x3 grid (27) + centers (27) = 54, pick 50.
    # Or simpler: just return a random one for variety, SA is robust.
    return np.random.uniform(-0.45, 0.45, (N, 3))

def _simulated_anneal(centers, time_limit):
    """
    Run Simulated Annealing to maximize the minimum distance.
    """
    start_time = time.time()
    current_centers = centers.copy()
    current_r = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    # Temperature schedule
    # Start high, decay exponentially
    T_start = 0.1
    T_end = 1e-6
    alpha = 0.995 # Cooling rate
    
    T = T_start
    
    # Precompute indices for efficiency? 
    # The force calculation is O(N^2).
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random perturbation
        # Move a random subset of points? Or all points slightly?
        # Moving all points slightly is safer for maintaining global structure.
        # Let's move all points with small Gaussian noise.
        step_size = T * 0.5
        noise = np.random.normal(0, step_size, (N, 3))
        new_centers = current_centers + noise
        
        # Project back to cube? 
        # If we clip, we lose gradient info but ensure validity.
        # Better: apply a penalty in the objective or just clip and accept/reject based on r.
        # Clipping is safe.
        new_centers = np.clip(new_centers, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        new_r = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_r >= current_r:
            current_centers = new_centers
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            delta = new_r - current_r
            # Since r is small (0.13), delta is small.
            # exp(delta/T)
            # If T is large, delta/T is small, exp is close to 1.
            # If T is small, delta/T is large negative, exp is close to 0.
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                current_centers = new_centers
                current_r = new_r
        
        T *= alpha
        
        # Optional: Every few steps, do a "push-apart" relaxation to help escape overlaps
        # This is a heuristic local search step.
        if np.random.random() < 0.1:
            # Push apart any overlapping pairs
            for _ in range(5):
                changed = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = current_centers[i] - current_centers[j]
                        dist = np.linalg.norm(d)
                        min_dist = 2 * best_r # Target distance
                        if dist < min_dist and dist > 1e-9:
                            # Push apart
                            push = (min_dist - dist) * 0.5
                            d_norm = d / dist
                            current_centers[i] += d_norm * push
                            current_centers[j] -= d_norm * push
                            changed = True
                if not changed:
                    break
            # Re-clip
            current_centers = np.clip(current_centers, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            current_r = _compute_radius(current_centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()

    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple seeds
    seeds = [
        _seed_fcc(),
        _seed_random(),
        _seed_bcc()
    ]
    
    # Add a slightly perturbed grid
    grid_seed = _seed_fcc() + np.random.normal(0, 0.01, (N, 3))
    seeds.append(grid_seed)
    
    best_overall_centers = None
    best_overall_r = 0.0
    
    # Allocate time to each seed
    # We have ~145s. 4 seeds. ~35s each.
    # But SA might converge faster. We can stop early if no improvement for a while?
    # For simplicity, split time evenly.
    
    time_per_seed = (deadline - time.time()) / len(seeds)
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
            
        # Run SA
        centers, r = _simulated_anneal(seed, time_limit=time_per_seed)
        
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = centers.copy()
            
        # Recalculate time for remaining seeds
        remaining_seeds = len(seeds) - i - 1
        if remaining_seeds > 0:
            time_per_seed = (deadline - time.time()) / remaining_seeds
    
    # Final polish: A quick local search on the best result
    if best_overall_centers is not None and time.time() < deadline - 2:
        # Do a few more SA iterations with very low T
        # Or just return
        pass
    
    if best_overall_centers is None:
        # Fallback
        return _seed_fcc()
        
    return best_overall_centers

# EVOLVE-BLOCK-END
