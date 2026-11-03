"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius.
    Uses a physics-based repulsive force model with simulated annealing.
    """
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        # Pairwise distances
        # Use broadcasting to get all pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        min_r_pairs = min_dist / 2.0
        
        # Wall distances: distance from each center to nearest wall
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)

    def physics_step(positions, temp, force_scale=1.0):
        """
        Perform one step of the physics simulation.
        Forces are repulsive between spheres and from walls.
        """
        positions = positions.copy()
        forces = np.zeros_like(positions)
        
        # 1. Sphere-Sphere Repulsion
        # We want to maximize the minimum distance. 
        # A common potential is 1/r^2 or exp(-r^2). 
        # Using a soft repulsion that is strong when close.
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Avoid division by zero
        dists[dists < 1e-8] = 1e-8
        
        # Force magnitude: k / dist^2 (Coulomb-like) or similar.
        # We want to push them apart.
        # To make it stable, we scale by temperature to allow large moves early on.
        # Force direction is normalized diff.
        # F_ij = (p_i - p_j) / |p_i - p_j|^3 * strength
        # Let's use a smoother potential gradient: -d/dr (1/r) = 1/r^2.
        
        # Normalize diff
        norms = np.linalg.norm(diff, axis=2)
        norms[norms < 1e-8] = 1e-8
        # diff_unit = diff / norms[..., np.newaxis]
        
        # Potential: V = 1/r. Gradient w.r.t p_i is (p_i - p_j)/r^2.
        # We sum forces from all j != i.
        # force_i = sum_j (p_i - p_j) / |p_i - p_j|^2
        
        # To avoid O(N^2) memory issues for large N (though N=50 is fine), 
        # we can compute in chunks or just do it directly since 50x50 is small.
        inv_dists_sq = 1.0 / (dists**2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        inv_dists_sq[~mask] = 0.0
        
        # Force contribution from each pair
        # F_i = sum_j (p_i - p_j) * inv_dists_sq[i,j]
        # diff[i,j] = p_i - p_j
        # So F_i = sum_j diff[i,j] * inv_dists_sq[i,j]
        forces = np.sum(diff * inv_dists_sq[..., np.newaxis], axis=1)
        
        # 2. Wall Repulsion
        # We want to keep spheres inside [-0.5, 0.5].
        # If a sphere is too close to a wall, push it inward.
        # Distance to wall: 0.5 - |x|
        # If dist < threshold, apply force.
        # Let's use a similar 1/d^2 style force from walls if close.
        # Actually, simply projecting or using a barrier potential is better.
        # Barrier: V_wall = 1/(0.5 - |x|)^2 if |x| > 0.4, else 0.
        # Gradient is 2*(0.5-|x|)^-3 * sign(x) * d(0.5-|x|)/dx ... 
        # Simpler: If x > 0.4, push left. If x < -0.4, push right.
        # Force magnitude proportional to how close it is.
        
        for dim in range(3):
            x = positions[:, dim]
            # Distance to nearest wall in this dimension
            dist_to_wall = 0.5 - np.abs(x)
            # Only apply force if close to wall (e.g., dist < 0.1)
            close_mask = dist_to_wall < 0.1
            if np.any(close_mask):
                # Force magnitude: k / dist^2
                # Direction: towards center (sign of -x)
                # F = -sign(x) * k / dist^2
                # Let's use a stronger force near the wall
                d = dist_to_wall[close_mask]
                d[d < 1e-4] = 1e-4
                f_mag = 1.0 / (d**2)
                # Direction is -sign(x)
                f_dir = -np.sign(x[close_mask])
                # Apply to force vector
                forces[close_mask, dim] += f_dir * f_mag * 5.0 # Scale factor for walls
                
        # Update positions
        # Step size is proportional to temperature
        # Add noise to escape local minima
        noise = np.random.normal(0, temp, size=positions.shape)
        positions += forces * temp + noise
        
        # Hard clip to prevent leaving the box (though forces should prevent it)
        positions = np.clip(positions, -0.5, 0.5)
        
        return positions

    def run_physics_optimization():
        """
        Run the simulated annealing physics simulation.
        """
        # Initialize on a grid to start with non-overlapping spheres
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        positions = grid[:N].copy()
        
        # Add small random perturbation
        np.random.seed(42)
        positions += np.random.uniform(-0.01, 0.01, size=positions.shape)
        
        # Simulated Annealing parameters
        initial_temp = 0.05
        final_temp = 0.0001
        n_steps = 2000
        cooling_rate = (final_temp / initial_temp) ** (1.0 / n_steps)
        
        best_positions = positions.copy()
        best_radius = compute_min_radius(positions)
        
        for i in range(n_steps):
            temp = initial_temp * (cooling_rate ** i)
            
            positions = physics_step(positions, temp)
            
            # Evaluate current state
            current_radius = compute_min_radius(positions)
            
            # Keep track of best
            if current_radius > best_radius:
                best_radius = current_radius
                best_positions = positions.copy()
        
        return best_positions, best_radius

    def refine_with_nelder_mead(positions):
        """
        Refine the solution using Nelder-Mead on a smooth objective.
        """
        # Use a smooth approximation of the min radius to help NM
        # f = -logsumexp(-beta * constraints)
        beta = 20.0
        
        def smooth_obj(x):
            pos = x.reshape(-1, 3)
            # Pairwise
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            pair_vals = dists / 2.0
            # Walls
            walls = 0.5 - np.abs(pos)
            wall_vals = walls.flatten()
            
            all_vals = np.concatenate([pair_vals, wall_vals])
            # Soft min
            neg_beta_x = -beta * all_vals
            max_val = np.max(neg_beta_x)
            log_sum = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
            soft_min = -log_sum / beta
            return -soft_min

        try:
            res = minimize(smooth_obj, positions.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    # Main execution
    np.random.seed(42)
    
    # Run physics optimization
    best_positions, best_radius = run_physics_optimization()
    print(f"Physics optimization result: radius = {best_radius:.10f}")
    
    # Refine with Nelder-Mead
    refined_positions = refine_with_nelder_mead(best_positions)
    refined_radius = compute_min_radius(refined_positions)
    print(f"Refined result: radius = {refined_radius:.10f}")
    
    if refined_radius > best_radius:
        best_positions = refined_positions
        best_radius = refined_radius
        
    # Try a few more restarts with different seeds if time permits? 
    # The physics simulation is deterministic with seed 42.
    # Let's try one more run with a different seed for diversity.
    np.random.seed(123)
    alt_positions, alt_radius = run_physics_optimization()
    alt_refined = refine_with_nelder_mead(alt_positions)
    alt_refined_radius = compute_min_radius(alt_refined)
    print(f"Alt refined result: radius = {alt_refined_radius:.10f}")
    
    if alt_refined_radius > best_radius:
        best_positions = alt_refined
        best_radius = alt_refined_radius

    best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    print(f"Final best radius: {best_radius:.10f}")
    
    return best_positions
# EVOLVE-BLOCK-END
