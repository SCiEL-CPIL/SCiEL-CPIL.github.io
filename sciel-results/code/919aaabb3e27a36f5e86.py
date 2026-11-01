"""Packing 50 equal circles in a unit circle using Molecular Dynamics."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(positions):
    """Compute the maximum radius for given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Distance to boundary
    norms = np.linalg.norm(positions, axis=1)
    boundary_dist = 1.0 - norms
    
    # Minimum pairwise distance
    if n > 1:
        # Calculate all pairwise distances
        # For N=50, O(N^2) is fine, but we can use scipy for efficiency if needed.
        # Let's stick to NumPy for simplicity in the new approach.
        dists = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :])**2, axis=-1))
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    return min(np.min(boundary_dist), min_pair_dist / 2.0)


def _initial_positions_md(n, seed=42):
    """Generate initial positions using random uniform sampling."""
    rng = np.random.default_rng(seed)
    positions = np.zeros((n, 2))
    
    # Rejection sampling for uniform distribution in unit circle
    for i in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y < 1.0:
                # Shrink slightly to avoid starting on the boundary
                positions[i] = [x * 0.9, y * 0.9]
                break
    return positions


def _improve_md(positions, max_iter=50000, seed=42, dt=0.01, damping=0.99):
    """Molecular Dynamics optimization."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    pos = positions.copy()
    vel = np.zeros((n, 2))
    
    best_radius = _compute_radius(pos)
    best_pos = pos.copy()
    
    # Force parameters
    # We want to maximize the minimum distance.
    # Use a repulsive force.
    # F_repulsive = k * (r_target - r) if r < r_target else 0
    # Or simpler: F = 1/r^2
    # Let's use a force that is strong at short range.
    
    for it in range(max_iter):
        forces = np.zeros((n, 2))
        
        # Pairwise forces
        # Vectorized calculation for speed
        # dists[i,j] = |pos[i] - pos[j]|
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        # Avoid division by zero
        dists_sq[dist_sq < 1e-12] = 1e-12
        dists = np.sqrt(dists_sq)
        
        # Force magnitude: 1/d^2 is a good repulsive potential derivative (up to sign)
        # F = k / d^2
        # We only apply force if they are close? 
        # For packing, we want them to spread out. 
        # A common potential is Lennard-Jones, but simple inverse square is robust.
        # Let's use F = 1 / d^2
        # However, if d is large, force is small. 
        # To ensure they fill the space, we might need a "pressure" or just rely on the container.
        # Since N is fixed, the repulsion will push them to the boundaries.
        
        force_mag = 1.0 / dists_sq
        # Set diagonal to 0
        np.fill_diagonal(force_mag, 0)
        
        # F_vec = force_mag * diff / dists
        # diff is (i-j), so force on i is in direction (i-j) which is away from j. Correct.
        forces += force_mag[:, :, np.newaxis] * diff / dists_sq[:, :, np.newaxis]
        
        # Wall forces
        # Push inward if close to boundary
        # Let's use a soft wall: F_wall = k_wall * (r - (1 - r_est)) * radial_dir
        # But r_est is unknown. 
        # Alternative: Hard wall with velocity reflection? 
        # Better: Strong repulsive force from boundary.
        # F_wall = -k_wall * pos if |pos| > 1 - margin
        # Let's use a quadratic potential near the wall: U = k * (|pos| - (1-eps))^2
        # Gradient: 2*k*(|pos| - (1-eps)) * pos/|pos|
        
        norms = np.linalg.norm(pos, axis=1)
        # Target radius for center: 1 - r_min. 
        # We don't know r_min, but we can use a fixed "soft" boundary at 0.95
        # If norm > 0.95, push in.
        # Force = -strength * (norm - 0.95) * pos/norm
        # Let's use a strong force.
        wall_strength = 10.0
        margin = 0.95
        for i in range(n):
            if norms[i] > margin:
                # Linear increase in force
                f_mag = wall_strength * (norms[i] - margin)
                forces[i] -= (pos[i] / norms[i]) * f_mag
        
        # Update velocity
        vel = (vel + forces * dt) * damping
        
        # Update position
        pos += vel * dt
        
        # Hard constraint: Keep inside circle
        # If a particle goes outside, reflect or clamp.
        # Clamping is safer for stability.
        new_norms = np.linalg.norm(pos, axis=1)
        outside = new_norms > 1.0
        if np.any(outside):
            # Scale back to boundary
            pos[outside] = pos[outside] / new_norms[outside, np.newaxis] * 0.999
            # Zero out velocity component normal to wall? Or just zero velocity?
            # Zeroing velocity is simple and stable.
            vel[outside] = 0.0
            
        # Check improvement
        if it % 10 == 0: # Check every 10 steps to save time
            current_radius = _compute_radius(pos)
            if current_radius > best_radius:
                best_radius = current_radius
                best_pos = pos.copy()
                
    return best_pos, best_radius


def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    best_positions = None
    best_radius = 0.0
    
    # Multi-start strategy
    # MD is deterministic given initial conditions, so we need different starts.
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        positions = _initial_positions_md(N, seed=seed)
        
        # Run MD
        # Parameters:
        # dt: time step
        # damping: velocity damping to remove energy
        # max_iter: number of steps
        positions, radius = _improve_md(positions, max_iter=10000, seed=seed, dt=0.05, damping=0.95)
        
        if radius > best_radius:
            best_radius = radius
            best_positions = positions.copy()
            
    # Final polish: Run a few more iterations on the best result
    if best_positions is not None:
        best_positions, _ = _improve_md(best_positions, max_iter=5000, seed=999, dt=0.01, damping=0.99)
        
    # Ensure valid output
    if best_positions is None:
        best_positions = _initial_positions_md(N, seed=0)
        
    # Final boundary check
    for i in range(N):
        norm = np.linalg.norm(best_positions[i])
        if norm > 0.999:
            best_positions[i] = 0.999 * best_positions[i] / norm
            
    return best_positions
# EVOLVE-BLOCK-END
