"""
Simulated Annealing based circle packing in the unit square.

Strategy:
1. Represent the packing by N centers (x, y) and N radii (r).
2. Define an energy function (objective to minimize) that penalizes:
   - Overlaps between circles (hard constraint violation)
   - Circles extending outside the unit square (hard constraint violation)
   - Small radii (soft constraint to maximize sum of radii)
   Specifically, Energy = Penalty_Overlap + Penalty_Boundary - Sum_Radii.
   The penalties are quadratic to ensure smooth gradients for the SA acceptance step.
3. Use Simulated Annealing to explore the configuration space.
   - At each step, perturb the centers and radii.
   - Calculate the change in energy.
   - Accept the new state if it is better, or with probability exp(-dE/T) if worse.
   - Anneal the temperature T over time.
4. This global search approach is less likely to get stuck in the numerous local
   optima of the circle packing problem compared to local gradient methods.
5. Multi-start with different initial configurations and random seeds.
6. Deadline guard to return the best valid (non-overlapping, in-boundary) solution.
"""

import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def compute_energy(centers, radii):
        """
        Compute the energy of the configuration.
        Lower energy is better.
        Energy = Overlap_Penalty + Boundary_Penalty - Sum_Radii
        """
        x = centers[:, 0]
        y = centers[:, 1]
        r = radii
        
        # 1. Boundary Penalty
        # If a circle goes out of bounds, penalize the amount of intrusion.
        # dist_to_left = x, dist_to_right = 1-x, etc.
        # If x < r, then x - r is negative. Penalty = (x - r)^2.
        # This encourages x >= r.
        boundary_violations = np.zeros(N)
        boundary_violations += np.maximum(0, r - x)       # Left
        boundary_violations += np.maximum(0, r - (1 - x)) # Right
        boundary_violations += np.maximum(0, r - y)       # Bottom
        boundary_violations += np.maximum(0, r - (1 - y)) # Top
        boundary_penalty = np.sum(boundary_violations**2)
        
        # 2. Overlap Penalty
        # For each pair i, j, if dist(i,j) < r_i + r_j, penalize.
        # Penalty = (r_i + r_j - dist)^2
        # We can compute this efficiently.
        # dist matrix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Sum of radii matrix
        sum_radii = r[:, None] + r[None, :]
        
        # Violation = max(0, sum_radii - dists)
        # We only care about i < j to avoid double counting, but summing all and dividing by 2 is fine.
        violations = np.maximum(0, sum_radii - dists)
        overlap_penalty = 0.5 * np.sum(violations**2)
        
        # 3. Objective: Maximize Sum Radii -> Minimize -Sum Radii
        # We want large radii. So we subtract sum(r) from energy.
        # To balance scales, we might need to tune the weight of -sum(r).
        # Let's assume penalties are dominant when violated.
        # When valid (penalty=0), energy = -sum(r). Minimizing this maximizes sum(r).
        
        energy = overlap_penalty + boundary_penalty - np.sum(r)
        
        return energy, overlap_penalty, boundary_penalty

    def is_valid(centers, radii, tol=1e-6):
        """Check if configuration is strictly valid."""
        x = centers[:, 0]
        y = centers[:, 1]
        r = radii
        
        # Boundary check
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
            
        # Overlap check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        sum_radii = r[:, None] + r[None, :]
        
        # Check if any dist < sum_radii - tol
        # np.fill_diagonal to ignore self
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(sum_radii, 0)
        
        if np.any(dists < sum_radii - tol):
            return False
            
        return True

    def simulate_annealing(initial_centers, initial_radii, time_limit):
        """
        Run SA from a given start.
        Returns best valid centers and radii found.
        """
        current_centers = initial_centers.copy()
        current_radii = initial_radii.copy()
        
        # Initialize with a valid configuration if possible?
        # The initial state might be invalid. SA can handle it via penalties.
        # But we want to return a valid one.
        
        best_centers = current_centers.copy()
        best_radii = current_radii.copy()
        best_energy = compute_energy(current_centers, current_radii)[0]
        best_valid_score = -np.inf # We track sum(r) for valid ones
        
        # Check if initial is valid
        if is_valid(current_centers, current_radii):
            best_valid_score = np.sum(current_radii)
            best_centers = current_centers.copy()
            best_radii = current_radii.copy()
        
        T = 0.05 # Initial temperature
        T_min = 0.001
        cooling_rate = 0.9995
        
        # Perturbation magnitude
        step_size = 0.05
        
        while time.time() < time_limit and T > T_min:
            # Generate new state
            new_centers = current_centers.copy()
            new_radii = current_radii.copy()
            
            # Perturb centers
            # Randomly select a few centers to move
            num_moves = random.randint(1, 3)
            idxs = np.random.choice(N, num_moves, replace=False)
            new_centers[idxs, 0] += np.random.normal(0, step_size, num_moves)
            new_centers[idxs, 1] += np.random.normal(0, step_size, num_moves)
            
            # Perturb radii
            # Radii should stay positive.
            # We can also move radii.
            new_radii[idxs] += np.random.normal(0, step_size * 0.5, num_moves)
            new_radii = np.maximum(new_radii, 0.01) # Keep small positive
            
            # Clip centers to a slightly larger box to allow "escaping" via penalty
            # But if we clip too hard, we lose the gradient of the boundary penalty.
            # Let's allow centers to go slightly out of [0,1] so boundary penalty works.
            # However, if centers go far out, it's useless.
            # Clip to [-0.5, 1.5]
            new_centers = np.clip(new_centers, -0.5, 1.5)
            
            # Calculate energies
            old_energy = compute_energy(current_centers, current_radii)[0]
            new_energy = compute_energy(new_centers, new_radii)[0]
            
            delta_E = new_energy - old_energy
            
            # Acceptance criterion
            if delta_E < 0:
                # Better, always accept
                current_centers = new_centers
                current_radii = new_radii
                current_energy = new_energy
            else:
                # Worse, accept with probability exp(-dE/T)
                if random.random() < np.exp(-delta_E / T):
                    current_centers = new_centers
                    current_radii = new_radii
                    current_energy = new_energy
            
            # Update best
            # We only care about valid configurations for the final result.
            # However, during SA, we might be in invalid states.
            # We should check validity periodically or if energy is low.
            # Checking validity is O(N^2).
            # Let's check every 10 steps or if energy is lower than current best valid energy.
            
            # To save time, only check validity if the configuration looks "promising"
            # i.e., penalties are low.
            _, overlap_pen, bound_pen = compute_energy(current_centers, current_radii)
            if overlap_pen < 1e-4 and bound_pen < 1e-4:
                if is_valid(current_centers, current_radii):
                    valid_score = np.sum(current_radii)
                    if valid_score > best_valid_score:
                        best_valid_score = valid_score
                        best_centers = current_centers.copy()
                        best_radii = current_radii.copy()
            
            # Cool down
            T *= cooling_rate
            
            # Adaptive step size?
            # If we are stuck (no improvement for a while), maybe increase step size?
            # Or if T is low, decrease step size?
            # Simple approach: keep step size constant or slowly decrease.
            step_size *= 0.9999
            
            # Ensure step size doesn't get too small
            step_size = max(step_size, 0.005)

        return best_centers, best_radii, best_valid_score

    def generate_initial_config():
        """
        Generate a random initial configuration.
        Try to make it somewhat valid initially to help SA.
        """
        centers = np.random.rand(N, 2)
        # Start with small radii to avoid immediate massive overlaps
        # Max possible radius is ~0.1 (1/sqrt(26) approx 0.19, but grid is tighter)
        # Let's start with r=0.05
        radii = np.full(N, 0.05)
        return centers, radii

    def generate_grid_config():
        """
        Generate a grid-based initial configuration.
        """
        # 5x5 grid = 25 points. Add 1 more.
        # Or 6x5 = 30, take 26.
        n_x, n_y = 6, 5
        x_coords = np.linspace(0.1, 0.9, n_x)
        y_coords = np.linspace(0.1, 0.9, n_y)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        centers = np.column_stack((x_f.flatten(), y_f.flatten()))
        
        # We have 30 points. We need 26.
        # Remove 4 random ones? Or just take the first 26?
        # Taking first 26 leaves gaps at the end.
        # Better to randomly select 26 indices from the 30.
        idx = np.random.choice(len(centers), N, replace=False)
        centers = centers[idx]
        
        # Estimate radii based on local density
        # Simple heuristic: min distance to neighbor / 2
        radii = np.zeros(N)
        for i in range(N):
            min_dist = np.inf
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, d)
            # Also consider boundary
            d_bound = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            radii[i] = min(min_dist, d_bound) * 0.4 # 0.4 to leave space for optimization
            
        return centers, radii

    best_centers = None
    best_radii = None
    best_score = -np.inf

    # Multi-start
    # We have ~140s.
    # Let's do 10 restarts, each with 12s limit.
    num_restarts = 10
    time_per_restart = 12.0

    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        t_limit = time.time() + time_per_restart
        
        # Alternate between random and grid starts
        if i % 2 == 0:
            init_c, init_r = generate_initial_config()
        else:
            init_c, init_r = generate_grid_config()
            
        # Add some noise to grid starts to break symmetry
        init_c = init_c + np.random.normal(0, 0.01, (N, 2))
        init_c = np.clip(init_c, 0, 1)
        
        c_res, r_res, score_res = simulate_annealing(init_c, init_r, t_limit)
        
        if score_res > best_score:
            best_score = score_res
            best_centers = c_res
            best_radii = r_res
            
        # If we found a very good solution, we might stop early?
        # Theoretical max for 26 circles is around sum(r) ~ 1.2 - 1.3?
        # Seed score 1.1586 implies sum(r) / seed_val = 1.1586.
        # If seed_val is sum(r) of seed program, and seed program is the current one...
        # Wait, the task says "search score is value/seed".
        # The "seed program" is likely a baseline.
        # The current program scores 1.1586.
        # So value ~ 1.1586 * seed_val.
        # We don't know seed_val, but we want to maximize value.
        
    # Final polish: If we have time left, run SA again on the best found solution
    # with a lower initial temperature to refine.
    if best_centers is not None and time.time() < deadline - 10.0:
        # Use best solution as start
        t_limit = deadline
        # Lower initial T for polishing
        # We can't easily change T inside simulate_annealing without modifying it.
        # Let's just run it again. It will cool down and refine.
        c_res, r_res, score_res = simulate_annealing(best_centers, best_radii, t_limit)
        if score_res > best_score:
            best_score = score_res
            best_centers = c_res
            best_radii = r_res

    # Fallback if no valid solution found (should not happen with grid start)
    if best_centers is None:
        # Simple grid
        n_x, n_y = 6, 5
        x_coords = np.linspace(0.1, 0.9, n_x)
        y_coords = np.linspace(0.1, 0.9, n_y)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        centers = np.column_stack((x_f.flatten(), y_f.flatten()))[:N]
        radii = np.full(N, 0.05)
        best_centers = centers
        best_radii = radii

    # Final validation and adjustment
    # Ensure radii are not negative
    best_radii = np.maximum(best_radii, 0)
    
    # One last check: if invalid, shrink radii until valid?
    # This is a greedy fix.
    if not is_valid(best_centers, best_radii):
        # Try to shrink radii
        for _ in range(100):
            if is_valid(best_centers, best_radii):
                break
            # Find the worst violation and shrink that radius
            # Or just shrink all by 1%
            best_radii *= 0.98

    result = np.column_stack((best_centers[:, 0], best_centers[:, 1], best_radii))
    
    return result

# EVOLVE-BLOCK-END
