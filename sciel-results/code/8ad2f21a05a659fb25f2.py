"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Finds a packing of 50 spheres in a unit cube [-0.5, 0.5]^3.
    Uses Simulated Annealing with a penalty function for a target radius R.
    Binary searches for the maximum feasible R.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    def calc_energy_contrib(pos, others, R):
        """
        Calculate energy contribution of a single position 'pos' 
        with respect to 'others' (array of positions) and walls, for radius R.
        """
        # Pairwise distances to others
        diffs = others - pos
        dists = np.linalg.norm(diffs, axis=1)
        # Overlap penalty: max(0, 2R - dist)^2
        # Note: dists includes distance to self? No, 'others' excludes self.
        overlaps = np.maximum(0, 2 * R - dists)
        pair_energy = np.sum(overlaps ** 2)
        
        # Wall penalty: max(0, (0.5 - R) - |coord|)^2 for each coord
        # dist_to_wall = 0.5 - abs(coord)
        # violation = max(0, R - dist_to_wall) = max(0, R - (0.5 - abs(coord)))
        # Let's use: max(0, (0.5 - R) - abs(coord)) is the clearance.
        # If clearance < 0, it's a violation.
        # Penalty: max(0, R - (0.5 - abs(coord)))^2
        abs_coords = np.abs(pos)
        wall_clearances = 0.5 - abs_coords
        wall_violations = np.maximum(0, R - wall_clearances)
        wall_energy = np.sum(wall_violations ** 2)
        
        return pair_energy + wall_energy

    def run_sa(R, num_steps, T_start=0.1, T_end=1e-6, seed=42):
        """
        Run Simulated Annealing for a fixed radius R.
        Returns (is_feasible, best_config, min_distance).
        """
        rng = np.random.default_rng(seed)
        
        # Initialize positions randomly
        X = rng.uniform(-0.5 + R, 0.5 - R, size=(N, 3))
        
        # Pre-calculate initial energy
        # This is O(N^2), acceptable for init
        E_total = 0.0
        for i in range(N):
            others = np.delete(X, i, axis=0)
            E_total += calc_energy_contrib(X[i], others, R)
        
        best_E = E_total
        best_X = X.copy()
        best_min_dist = -1.0
        
        # Cooling schedule
        cooling_rate = (T_end / T_start) ** (1.0 / num_steps)
        T = T_start
        
        # Pre-allocate array for others to avoid memory allocation in loop?
        # We can just use slicing, it's fine for N=50.
        
        for step in range(num_steps):
            if time.time() - start_time > time_limit:
                break
                
            # Pick random index
            i = rng.integers(0, N)
            
            # Old position
            pos_old = X[i].copy()
            
            # New position: Gaussian perturbation
            # Scale noise with T? Or fixed?
            # Fixed noise is often better for packing.
            noise_scale = 0.05
            pos_new = pos_old + rng.normal(0, noise_scale, size=3)
            
            # Clip to cube
            pos_new = np.clip(pos_new, -0.5, 0.5)
            
            # Calculate delta E
            others = np.delete(X, i, axis=0)
            
            E_old_contrib = calc_energy_contrib(pos_old, others, R)
            E_new_contrib = calc_energy_contrib(pos_new, others, R)
            
            delta_E = E_new_contrib - E_old_contrib
            
            # Acceptance criterion
            if delta_E < 0:
                accept = True
            else:
                # Avoid log(0)
                if T > 1e-10:
                    accept = rng.random() < np.exp(-delta_E / T)
                else:
                    accept = False
            
            if accept:
                X[i] = pos_new
                E_total += delta_E
                
                # Update best if energy is lower
                if E_total < best_E:
                    best_E = E_total
                    best_X = X.copy()
                    # Calculate min distance for score tracking (expensive, do sparingly or at end)
                    # For now, just track energy. We'll calc min_dist at the end.
                    
            T *= cooling_rate
            
        # Calculate actual min distance for the best config found
        # min_dist = min( min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|) )
        min_dist = 0.5 # Max possible is 0.5
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_X[i] - best_X[j])
                if d / 2.0 < min_dist:
                    min_dist = d / 2.0
        # Walls
        for i in range(N):
            for k in range(3):
                d_wall = 0.5 - abs(best_X[i][k])
                if d_wall < min_dist:
                    min_dist = d_wall
        
        # Feasibility check for the specific R
        # If min_dist >= R, it's feasible for R
        is_feasible = (min_dist >= R - 1e-9)
        
        return is_feasible, best_X, min_dist

    # Binary Search for R
    # Known best is ~0.1359
    low_R = 0.12
    high_R = 0.14
    
    best_config = None
    best_score = 0.0
    
    # We will try to find the max R.
    # To save time, we do a coarse search first, then fine.
    # Or just a standard binary search with a time check.
    
    iterations = 0
    while (high_R - low_R > 1e-4) and (time.time() - start_time < time_limit - 5.0):
        mid_R = (low_R + high_R) / 2.0
        
        # Run SA with mid_R
        # Use different seeds for diversity?
        # For a single check, one run is risky.
        # Let's run 2 short SAs or 1 longer one.
        # Given time limit, 1 run of 20k steps is fast.
        feasible, config, score = run_sa(mid_R, num_steps=30000, seed=iterations)
        
        if feasible:
            low_R = mid_R
            if score > best_score:
                best_score = score
                best_config = config
        else:
            high_R = mid_R
        
        iterations += 1
        
    # If we ran out of time or converged, try to improve the best_config found
    # by running SA with a slightly lower R to allow more movement?
    # Or just return best_config.
    
    if best_config is None:
        # Fallback: Grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_config = grid[:N]
        
    return best_config

# EVOLVE-BLOCK-END
