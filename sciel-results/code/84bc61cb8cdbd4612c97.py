"""
Strategy: Simulated Annealing with Adaptive Radius Target
This approach abandons the differentiable soft-min surrogate. Instead, it uses
Simulated Annealing (SA) to explore the configuration space.

Key differences from the current program:
1. No gradients: Uses random perturbations (moves) rather than gradient ascent.
2. Explicit Radius Annealing: The target radius 'r' is increased over time.
   The objective is to minimize a penalty function that penalizes:
   - Pairwise distances < 2*r
   - Wall distances < r
   - A global penalty for r itself (to encourage increasing r)
3. Stochastic Acceptance: Moves that increase the penalty are accepted with
   probability exp(-delta_E / T), allowing escape from local minima.
4. Multi-start: If time permits, we restart with different initial configurations
   or higher starting radii to explore different basins of attraction.

The energy function E(c, r) is defined as:
E = sum_{i<j} max(0, 2*r - |c_i - c_j|)^2 + sum_{i,k} max(0, r - (0.5 - |c_ik|))^2
We minimize E while maximizing r. In practice, we fix r to a current target and
minimize E. If E becomes small enough (below a threshold), we increase r.
"""

import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Best result tracking
    best_centers = None
    best_radius = 0.0
    
    # Helper to compute exact radius
    def compute_exact_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    # Energy function for SA
    # We want to minimize overlap. 
    # E = sum (max(0, min_dist_target - actual_dist))^2
    # Here min_dist_target is 2*r for pairs, r for walls.
    def compute_energy(c, r):
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, 2*r) # Ignore self
        
        # Violations: dists < 2*r
        violations = np.maximum(0, 2*r - dists)
        pair_energy = np.sum(violations ** 2)
        
        # Walls
        wall_dists = 0.5 - np.abs(c)
        wall_violations = np.maximum(0, r - wall_dists)
        wall_energy = np.sum(wall_violations ** 2)
        
        return pair_energy + wall_energy

    # Initialization strategy: 
    # Start with a reasonable grid, then jitter.
    # We will run multiple SA "runs" with different starting radii and temperatures.
    
    # Known good radius is ~0.111. Let's start a bit lower to ensure feasibility, then anneal up.
    # Or start high and let SA find a way? Usually better to start feasible and push up.
    
    # Phase 0: Quick Force-Directed Initialization to get a decent starting point
    # This is kept from the previous strategy as it's fast and effective for initialization.
    # But we will use it only to get a starting point for SA, not the final answer.
    
    rng = np.random.default_rng(42)
    
    # Simple grid init
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c_init = grid[:N].copy()
    c_init += rng.normal(0, 0.005, size=c_init.shape)
    c_init = np.clip(c_init, -0.49, 0.49)
    
    # Quick force relaxation to spread them out a bit more than the grid
    for _ in range(100):
        diff = c_init[:, np.newaxis, :] - c_init[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, 1.0)
        forces = diff / (dists[:, :, np.newaxis] ** 3)
        net_forces = np.sum(forces, axis=1)
        c_init += 0.01 * net_forces
        c_init = np.clip(c_init, -0.49, 0.49)
        
    r_init = compute_exact_radius(c_init)
    best_centers = c_init.copy()
    best_radius = r_init
    
    # SA Parameters
    # We will anneal the temperature T from T_start to T_end.
    # And anneal the target radius r from r_start to r_max_possible.
    
    # Theoretical max radius for 100 spheres in unit cube?
    # Volume argument: 100 * (4/3)pi r^3 <= 1 => r <= (3/400pi)^(1/3) ~ 0.129
    # But packing efficiency is lower. 0.111 is known.
    # Let's aim for 0.115 to see if we can beat it.
    
    r_start = max(0.05, r_init * 0.9)
    r_target_max = 0.115
    
    T_start = 0.05
    T_end = 0.0001
    
    # Number of SA steps
    # We have ~120s. 
    # Energy calc is O(N^2) ~ 10^4 ops. 
    # Let's do as many steps as possible.
    
    current_c = c_init.copy()
    current_r = r_start
    current_energy = compute_energy(current_c, current_r)
    
    # Annealing schedule
    total_steps = 50000 # We will break based on time
    
    for step in range(total_steps):
        if time.time() > deadline - 10:
            break
            
        # Update temperature
        progress = step / total_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Update target radius
        # Linearly increase r, but slower than T decrease
        # We want to reach r_target_max by the end
        current_r = r_start + (r_target_max - r_start) * (progress ** 0.5) # Faster start, slower end
        
        # Propose a move: Pick a random sphere and move it randomly
        idx = rng.integers(0, N)
        # Move magnitude proportional to T and current_r?
        # Typical move size ~ 0.01 * T_factor
        move_mag = 0.02 * (T / T_start + 0.1)
        delta = rng.normal(0, move_mag, size=3)
        
        new_c = current_c.copy()
        new_c[idx] += delta
        # Clip to bounds to maintain validity (or allow slight violation and penalize?)
        # Clipping is safer for the "inside cube" constraint.
        new_c[idx] = np.clip(new_c[idx], -0.5, 0.5)
        
        new_energy = compute_energy(new_c, current_r)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            # Improvement
            current_c = new_c
            current_energy = new_energy
        else:
            # Worse, accept with prob exp(-delta_E / T)
            if rng.random() < np.exp(-delta_E / T):
                current_c = new_c
                current_energy = new_energy
        
        # Periodic check of exact radius
        if step % 100 == 0:
            r_exact = compute_exact_radius(current_c)
            if r_exact > best_radius:
                best_radius = r_exact
                best_centers = current_c.copy()
                
        # If energy is very low, we might be able to jump r faster?
        # The linear schedule is simple. Let's stick to it for now.
        
    # Final check
    r_final = compute_exact_radius(best_centers)
    if r_final > best_radius:
        best_radius = r_final
        
    # Ensure validity
    if np.any(np.isnan(best_centers)) or np.any(np.isinf(best_centers)):
        return c_init # Fallback
        
    if np.any(best_centers > 0.5) or np.any(best_centers < -0.5):
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers

if __name__ == "__main__":
    pass
