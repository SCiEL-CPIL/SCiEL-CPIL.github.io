import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Unit hexagon vertices (flat top)
    # Side length 1
    # Vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Normals for bounding hexagon (flat top)
    # Angles 30, 90, 150, 210, 270, 330
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def calc_side_length(centers):
        # centers: (N, 2) array
        # Generate all vertices
        # Expand centers to (N, 6, 2)
        verts = centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
        # Flatten to (N*6, 2)
        verts_flat = verts.reshape(-1, 2)
        
        # Project onto normals
        # (N*6, 6)
        projections = verts_flat @ normals.T
        # Max absolute projection
        max_proj = np.max(np.abs(projections), axis=0)
        # S = 2/sqrt(3) * max(max_proj)
        return (2.0 / np.sqrt(3)) * np.max(max_proj)
        
    # Generate lattice points
    # Axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    radius = 4
    points = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            # Check if within hexagonal radius
            if max(abs(q), abs(r), abs(-q-r)) <= radius:
                x = np.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                points.append([x, y])
                
    points = np.array(points)
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices]
    
    # Initial guess: first 12 points
    current_idx = list(range(N))
    current_centers = points[current_idx]
    current_score = calc_side_length(current_centers)
    
    best_score = current_score
    best_centers = current_centers.copy()
    
    # Local search: try swapping one point with another
    # Or just try different subsets of the first 50 points
    search_space = 50
    available_indices = list(range(search_space))
    
    # Greedy construction
    # Start with center, add next best point
    # This is slow for N=12? 12 steps, each checking 50 points. 600 evals. Fast.
    
    # Let's do a simple hill climbing on the set of indices
    # Represent set as boolean mask or list
    
    # Restart a few times with different initial sets
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Random initial set from first 30 points
        init_idx = np.random.choice(30, N, replace=False)
        cur_centers = points[init_idx]
        cur_score = calc_side_length(cur_centers)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping each point with any other available point
            for i in range(N):
                if time.time() > deadline:
                    break
                # Get current index
                curr_idx = init_idx[i]
                # Try swapping with any j in available_indices not in init_idx
                # To speed up, just try swapping with points in the first 50
                for j in available_indices:
                    if j in init_idx:
                        continue
                    # Swap
                    new_idx = list(init_idx)
                    new_idx[i] = j
                    new_centers = points[new_idx]
                    new_score = calc_side_length(new_centers)
                    if new_score < cur_score:
                        cur_score = new_score
                        init_idx = np.array(new_idx)
                        cur_centers = new_centers
                        improved = True
                        # Restart inner loop? No, continue scanning
                        if cur_score < best_score:
                            best_score = cur_score
                            best_centers = cur_centers.copy()
        
        # Update best
        if cur_score < best_score:
            best_score = cur_score
            best_centers = cur_centers.copy()
            
    # Convert best_centers to output format
    # inner: (x, y, angle_degrees)
    inner = []
    for c in best_centers:
        inner.append([c[0], c[1], 0.0])
        
    # Outer hexagon
    center = [0.0, 0.0]
    side = best_score
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# The code above is the new implementation
# EVOLVE-BLOCK-END
