import numpy as np


def construct():
        N = 32
        # Initialize in a grid
        side = int(np.ceil(np.sqrt(N)))
        # Use a hexagonal-like start for better packing
        # Or just grid
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    centers.append([x, y])
        centers = np.array(centers)
        
        # Initial radii
        radii = np.zeros(N)
        
        # Relaxation loop
        for _ in range(1000):
            # Update radii based on current centers
            # r_i is limited by walls and neighbors
            # We solve for r_i such that r_i <= dist_wall_i and r_i + r_j <= d_ij
            # This is a system of inequalities.
            # Simple approximation:
            # r_i = min(dist_wall_i, min_j (d_ij - r_j))
            # This requires r_j to be known. We can iterate this.
            
            for k in range(10): # Inner iterations for radius consistency
                for i in range(N):
                    r_max = min(centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1])
                    for j in range(N):
                        if i == j: continue
                        d = np.linalg.norm(centers[i] - centers[j])
                        r_max = min(r_max, d - radii[j])
                    radii[i] = max(0, r_max)
            
            # Move centers to increase radii
            # For each circle, find the direction that allows a larger radius
            # This is the direction towards the "center" of the largest inscribed circle in the Voronoi cell
            # Approximation: Move away from the closest constraints
            
            new_centers = centers.copy()
            for i in range(N):
                # Find closest constraints
                min_dist = np.inf
                constraint_normal = np.array([0.0, 0.0])
                num_constraints = 0
                
                # Check walls
                dists = [
                    (centers[i][0], np.array([-1.0, 0.0])), # Left wall, normal points right
                    (1-centers[i][0], np.array([1.0, 0.0])), # Right wall
                    (centers[i][1], np.array([0.0, -1.0])), # Bottom wall
                    (1-centers[i][1], np.array([0.0, 1.0]))  # Top wall
                ]
                for dist, normal in dists:
                    if dist < min_dist + 1e-6: # Closest
                        if dist < min_dist:
                            min_dist = dist
                            constraint_normal = normal
                            num_constraints = 1
                        else:
                            constraint_normal += normal
                            num_constraints += 1
                
                # Check neighbors
                for j in range(N):
                    if i == j: continue
                    d = np.linalg.norm(centers[i] - centers[j])
                    # Effective distance for radius expansion
                    # If r_i + r_j < d, we can expand.
                    # The limiting factor is the gap.
                    # Move away from j if j is close.
                    # The "radius" contributed by neighbor j is d - r_j.
                    # If this is the minimum, move away from j.
                    gap = d - radii[j]
                    if gap < min_dist + 1e-6:
                        if gap < min_dist:
                            min_dist = gap
                            # Normal points from j to i
                            if d > 0:
                                n = (centers[i] - centers[j]) / d
                            else:
                                n = np.array([1.0, 0.0])
                            constraint_normal = n
                            num_constraints = 1
                        else:
                            if d > 0:
                                n = (centers[i] - centers[j]) / d
                            else:
                                n = np.array([1.0, 0.0])
                            constraint_normal += n
                            num_constraints += 1
                
                # Average the normals
                if num_constraints > 0:
                    move_dir = constraint_normal / num_constraints
                    # Normalize
                    norm = np.linalg.norm(move_dir)
                    if norm > 0:
                        move_dir /= norm
                        # Step size
                        step = 0.01 * (1.0 + _/100.0) # Annealing? Or fixed
                        step = 0.05
                        new_centers[i] += move_dir * step
                        # Clip to bounds
                        new_centers[i][0] = np.clip(new_centers[i][0], 0.01, 0.99)
                        new_centers[i][1] = np.clip(new_centers[i][1], 0.01, 0.99)
            
            centers = new_centers
            
        # Final calculation of radii
        radii = np.zeros(N)
        for k in range(20):
            for i in range(N):
                r_max = min(centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1])
                for j in range(N):
                    if i == j: continue
                    d = np.linalg.norm(centers[i] - centers[j])
                    r_max = min(r_max, d - radii[j])
                radii[i] = max(0, r_max)
        
        result = np.zeros((N, 3))
        for i in range(N):
            result[i] = [centers[i][0], centers[i][1], radii[i]]
        return result
