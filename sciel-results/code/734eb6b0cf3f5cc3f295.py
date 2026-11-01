import numpy as np

N = 32

def construct():
    """
    Pack 32 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii.
    Uses a repulsion-based relaxation approach.
    """
    np.random.seed(42) # For reproducibility in evaluation if needed, but usually not required. Let's use random for robustness.
    
    # 1. Initialize positions in a grid
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // side
        col = i % side
        positions[i, 0] = (col + 0.5) / side
        positions[i, 1] = (row + 0.5) / side
    
    radii = np.zeros(N)
    
    # Parameters
    max_iters = 5000
    growth_factor = 0.5 # How aggressively to grow radii
    repulsion_strength = 1.0
    
    for it in range(max_iters):
        # Step 1: Calculate max possible radius for each circle based on current positions
        # Distance to walls
        dist_walls = np.minimum(
            np.minimum(positions[:, 0], 1 - positions[:, 0]),
            np.minimum(positions[:, 1], 1 - positions[:, 1])
        )
        
        # Distance to other circles
        # Compute pairwise distances
        # dist_matrix[i, j] = distance between i and j
        dist_matrix = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :])**2, axis=2))
        
        # For each circle i, the constraint from circle j is: r_i + r_j <= dist(i, j)
        # So r_i <= dist(i, j) - r_j
        # We want the max r_i such that this holds for all j.
        # r_i <= min_j (dist(i, j) - r_j)
        
        # Create a matrix where M[i, j] = dist(i, j) - r_j
        # We need min over j != i
        # Set diagonal to infinity so it doesn't affect min
        M = dist_matrix - radii[np.newaxis, :]
        np.fill_diagonal(M, np.inf)
        
        min_dist_to_others = np.min(M, axis=1)
        
        # The max radius is limited by walls and other circles
        max_radii = np.minimum(dist_walls, min_dist_to_others)
        
        # Ensure radii are non-negative
        max_radii = np.maximum(max_radii, 0)
        
        # Update radii
        # We can set radii = max_radii directly, but this might cause oscillations.
        # A smoother update: radii = growth_factor * max_radii + (1 - growth_factor) * radii
        # However, if we are far from optimum, we want to grow fast.
        # Let's try setting radii to max_radii directly first. If it oscillates, we damp.
        new_radii = max_radii
        
        # Check if radii have converged
        if np.allclose(radii, new_radii, atol=1e-6):
            radii = new_radii
            break
            
        radii = new_radii
        
        # Step 2: Resolve overlaps by moving positions
        # We need to check if dist(i, j) < r_i + r_j
        # dist_matrix is already computed.
        # Overlap condition: dist_matrix[i, j] < radii[i] + radii[j]
        
        overlap_matrix = radii[:, np.newaxis] + radii[np.newaxis, :] - dist_matrix
        # overlap_matrix[i, j] > 0 means overlap
        
        # Apply repulsive forces
        # We iterate over pairs or use vectorization
        # For each i, j with overlap, push apart
        
        # To be efficient, we can compute the vector of displacements
        # Let's do a simple iterative push
        # We might need multiple passes to fully resolve if many overlaps
        for _ in range(10): # Inner loop for stability
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dist = dist_matrix[i, j]
                    if dist < 1e-9:
                        # If they are at the same spot, push randomly
                        dx = np.random.randn(2)
                        dist = np.linalg.norm(dx)
                        dx = dx / dist
                    else:
                        dx = positions[i] - positions[j]
                        dist = np.linalg.norm(dx)
                        dx = dx / dist
                    
                    required_dist = radii[i] + radii[j]
                    if dist < required_dist:
                        # Push apart
                        # Move each by half the overlap
                        push = (required_dist - dist) / 2.0
                        positions[i] += push * dx
                        positions[j] -= push * dx
                        moved = True
                        
            # Clamp positions to [0, 1]
            positions = np.clip(positions, 0, 1)
            
            # Recalculate dist_matrix if we moved
            if moved:
                dist_matrix = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :])**2, axis=2))
            else:
                break

    # Final check and cleanup
    # Ensure no negative radii
    radii = np.maximum(radii, 0)
    
    # Construct result array
    result = np.hstack([positions, radii[:, np.newaxis]])
    
    return result
