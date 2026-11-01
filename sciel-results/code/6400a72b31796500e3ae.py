# EVOLVE-BLOCK-START
import numpy as np

def _relax_packing(positions, iterations=10000, damping=0.9, force_constant=1.0, boundary_strength=5.0):
    """
    Improve packing using physical relaxation with repulsive forces.
    """
    n = len(positions)
    pos = positions.copy().astype(float)
    best_radius = -np.inf
    best_pos = pos.copy()
    
    # Precompute indices for pairwise distances
    # We will compute all pairwise distances in a vectorized manner
    
    for it in range(iterations):
        # Calculate pairwise distances
        # pos[:, np.newaxis, :] - pos[np.newaxis, :, :] gives (n, n, 2)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Calculate forces
        # Repulsive force between circles: F = k / d^2 * direction
        # We want to maximize the minimum distance, so we push apart if close.
        # A simple potential is V = -1/d, F = -grad V = -1/d^2 * r_hat
        # Or just a strong repulsion for all pairs to spread them out.
        
        # Force magnitude
        # Use a threshold to avoid huge forces when very close, but keep them strong.
        # Let's use F = 1 / dist_sq for all pairs, scaled.
        force_mag = force_constant / dist_sq
        # Zero out self-interaction
        np.fill_diagonal(force_mag, 0.0)
        
        # Force vector
        # F_ij = force_mag[i,j] * (pos[i] - pos[j]) / dist[i,j]
        # We can compute this as: force_mag[:, :, np.newaxis] * diff / dist[:, :, np.newaxis]
        forces = force_mag[:, :, np.newaxis] * diff / dist[:, :, np.newaxis]
        
        # Sum forces on each particle
        net_forces = np.sum(forces, axis=1)
        
        # Boundary force: Push inward if near boundary
        # dist_to_center = norm(pos)
        # If dist_to_center > 0.9, push towards center
        norms = np.linalg.norm(pos, axis=1)
        boundary_forces = np.zeros_like(pos)
        mask = norms > 0.9
        if np.any(mask):
            # Stronger force as we get closer to boundary
            excess = norms[mask] - 0.9
            # Direction is -pos / norm
            boundary_forces[mask] = -pos[mask] / norms[mask, np.newaxis] * (excess * boundary_strength)
        
        net_forces += boundary_forces
        
        # Update positions
        # Simple Euler integration with damping
        pos += net_forces * 0.01
        pos *= damping # Velocity damping isn't explicit here, but we can add a small global damping to positions? 
        # Actually, standard relaxation is pos += step * force.
        # Let's stick to pos += step * force for simplicity, but maybe scale step down over time.
        
        # Keep inside container
        norms = np.linalg.norm(pos, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pos[mask] = pos[mask] / norms[mask, np.newaxis]
            
        # Calculate current radius
        # Min pairwise distance / 2
        min_pair_dist = np.min(dist)
        # Min distance to boundary
        min_boundary_dist = 1.0 - np.max(norms)
        
        current_radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_pos = pos.copy()
            
        # Optional: Stop if no improvement for a long time? 
        # For now, just run fixed iterations.
        
    return best_pos, best_radius

def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    N = 50
    best_positions = None
    best_radius = 0.0
    
    # Multiple restarts to find a better local minimum
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        rng = np.random.default_rng(seed)
        
        # Random initial positions inside a smaller circle to avoid initial overlaps
        # Radius 0.8 is safe for 50 circles with r ~ 0.12
        theta = rng.uniform(0, 2 * np.pi, N)
        r = rng.uniform(0, 0.8, N)
        
        positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        # Relax
        pos, radius = _relax_packing(positions, iterations=5000)
        
        if radius > best_radius:
            best_radius = radius
            best_positions = pos
            
    # Final refinement with more iterations on the best candidate
    if best_positions is not None:
        best_positions, _ = _relax_packing(best_positions, iterations=10000)
        
    # Ensure strict boundary constraint
    norms = np.linalg.norm(best_positions, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_positions[mask] = best_positions[mask] / norms[mask, np.newaxis]
        
    return best_positions
# EVOLVE-BLOCK-END
