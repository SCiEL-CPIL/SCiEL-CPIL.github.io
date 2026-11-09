"""Improved packing for 100 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Maximizes the radius of N equal spheres in a unit cube [-0.5, 0.5]^3.
    Uses a force-directed layout with simulated annealing to spread points.
    """
    np.random.seed(42) # For reproducibility, can be removed if random restarts are used
    
    # 1. Initialization: Use a simple cubic grid or random points.
    # A grid is often a better starting point for repulsion methods than pure random.
    # For N=100, a 5x5x4 grid is perfect.
    # We'll create a grid of points.
    nx, ny, nz = 5, 5, 4
    x_coords = np.linspace(-0.5, 0.5, nx)
    y_coords = np.linspace(-0.5, 0.5, ny)
    z_coords = np.linspace(-0.5, 0.5, nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # We have exactly 5*5*4 = 100 points.
    # If N != nx*ny*nz, we would need to adjust. Here N=100 matches.
    
    # 2. Force-Directed Optimization
    # Objective: Maximize the minimum distance between any two points and from points to walls.
    # Strategy: Apply repulsive forces between all pairs and between points and walls.
    # Use a cooling schedule (Simulated Annealing) to avoid local minima.
    
    pos = points.copy()
    
    # Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.99  # Cooling rate
    max_iters = 5000
    step_scale = 0.01 # Base step size
    
    def get_radius(pos):
        """Calculate the current radius (min distance to wall or other point / 2)."""
        # Distance to walls
        # Walls are at +-0.5
        # Distance from point p to wall is min(0.5 - p_x, 0.5 + p_x, ...) 
        # Actually, distance to boundary is min(0.5 - p_i, 0.5 + p_i) for each coord i
        # min over all coords and all points
        d_walls = np.min(np.minimum(0.5 - np.abs(pos[:, 0]), 0.5)) # Wait, this is wrong.
        
        # Correct distance to wall:
        # For a point p, dist to wall in x is min(p[0] - (-0.5), 0.5 - p[0]) = min(p[0]+0.5, 0.5-p[0])
        # Since p is in [-0.5, 0.5], this is just 0.5 - abs(p[0])? No.
        # If p[0] = 0.4, dist to right wall is 0.1, left is 0.9. Min is 0.1.
        # 0.5 - abs(0.4) = 0.1. Correct.
        # If p[0] = -0.4, dist to left is 0.1, right is 0.9. Min is 0.1.
        # 0.5 - abs(-0.4) = 0.1. Correct.
        
        wall_dists = np.min(0.5 - np.abs(pos), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        # Pairwise distances
        # Compute pairwise squared distances
        # To save memory, we can do this in chunks or just use the full matrix for N=100 (100x100 is small)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist_sq = np.min(dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        # Radius is half the min pair distance, but must not exceed distance to wall
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    best_radius = get_radius(pos)
    
    for i in range(max_iters):
        if T < T_min:
            break
            
        # Calculate forces
        # Repulsion between points: F_ij = k / d^2 * direction
        # Repulsion from walls: F_wall = k_wall * (1 - d_wall) * direction_inwards
        
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dists_sq, 1.0) 
        dists = np.sqrt(dists_sq)
        
        # Force magnitude: k / d^2
        # We want to push points apart.
        # F_ij on i is sum over j of (pos_i - pos_j) / d^3 * k
        # Let's use a soft potential or just inverse square law.
        # To prevent explosion, we can cap the force or use a softer potential.
        # Let's use 1/d^2 magnitude, so vector is diff / d^3 * k.
        
        # Normalize diff to get direction
        # dists is 100x100
        # We need to sum forces for each i.
        # force[i] = sum_j (pos[i] - pos[j]) / dist(i,j)^3 * k
        
        # To avoid numerical issues when d is small, add a small epsilon
        inv_d3 = 1.0 / (dists_sq * dists + 1e-12)
        np.fill_diagonal(inv_d3, 0.0)
        
        # forces[i] = sum_j (pos[i] - pos[j]) * inv_d3[i,j] * k
        # This is equivalent to (pos[i] * sum_j inv_d3[i,j] - sum_j pos[j] * inv_d3[i,j])
        # But simpler to just do matrix multiplication if we structure it right?
        # Actually, just looping over i is fine for N=100.
        for i_idx in range(N):
            # diff[i_idx] is pos[i_idx] - pos[j] for all j
            # We want sum over j of (pos[i_idx] - pos[j]) * inv_d3[i_idx, j]
            # This is sum_j (pos[i_idx] * inv_d3[i_idx, j] - pos[j] * inv_d3[i_idx, j])
            # = pos[i_idx] * sum_j inv_d3[i_idx, j] - sum_j (pos[j] * inv_d3[i_idx, j])
            
            # Vectorized:
            # weights = inv_d3[i_idx, :]
            # force_i = pos[i_idx] * np.sum(weights) - np.sum(pos * weights[:, None], axis=0)
            
            weights = inv_d3[i_idx, :]
            force_i = pos[i_idx] * np.sum(weights) - np.sum(pos * weights[:, np.newaxis], axis=0)
            forces[i_idx] += force_i
            
        # Wall repulsion
        # For each point, if it's close to a wall, push it back.
        # Distance to wall in x: d_x = 0.5 - abs(x)
        # If d_x < threshold, force is proportional to (threshold - d_x)
        # Let's use a simple linear spring force towards center if within 0.1 of wall.
        # Or just repulsive force from the boundary.
        # Force_x = k_wall * (0.5 - x) if x > 0 else k_wall * (0.5 + x) ? No, that's towards center.
        # We want to keep them inside. If they hit the wall, push them in.
        # Actually, the objective function handles the penalty. For the gradient/force,
        # we can just project them back or add a force.
        # Let's add a force that pushes points away from the walls.
        # F_x = k * (0.5 - x) for x in [0, 0.5] -> pushes towards 0? No.
        # If x is close to 0.5, 0.5 - x is small.
        # We want force to be large when close to wall.
        # Let's use: if x > 0.4, F_x = -k * (x - 0.4) ?
        # Simpler: Just clip positions and let the pairwise repulsion do the work.
        # The wall constraint is handled by the score function.
        # However, if points get stuck on the wall, they can't move out easily with just pairwise repulsion
        # because the wall is a hard constraint in the score, but not in the force if we don't add it.
        # Let's add a soft wall force.
        
        # Force towards center if outside a "safe zone"
        safe_margin = 0.45
        for i_idx in range(N):
            for dim in range(3):
                val = pos[i_idx, dim]
                if val > safe_margin:
                    forces[i_idx, dim] -= (val - safe_margin) * 10.0
                elif val < -safe_margin:
                    forces[i_idx, dim] -= (val + safe_margin) * 10.0
        
        # Update positions
        # Move in direction of force, scaled by temperature and step size
        # Add noise to escape local minima
        noise = np.random.randn(*pos.shape) * T
        pos += forces * step_scale + noise
        
        # Constrain to cube
        pos = np.clip(pos, -0.5, 0.5)
        
        # Accept/Reject step (Metropolis criterion)
        # Calculate new radius
        new_radius = get_radius(pos)
        
        if new_radius > best_radius:
            best_radius = new_radius
            # We keep the position as is (it's already updated)
        else:
            # Accept with probability exp((new - old) / T)
            # Here "energy" is -radius, so lower energy is better.
            # Delta E = -new_radius - (-best_radius) = best_radius - new_radius
            # If new_radius < best_radius, Delta E > 0.
            delta = best_radius - new_radius
            if np.random.rand() < np.exp(-delta / T):
                # Accept, keep pos
                pass
            else:
                # Reject, revert pos
                pos = pos - forces * step_scale - noise
                pos = np.clip(pos, -0.5, 0.5)
                
        # Cool down
        T *= alpha
        
    return pos
# EVOLVE-BLOCK-END
