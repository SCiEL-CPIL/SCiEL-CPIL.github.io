"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Generate FCC lattice points
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # We generate points in a box [-K, K]^3
    K = 10
    points = []
    for i in range(-K, K + 1):
        for j in range(-K, K + 1):
            for k in range(-K, K + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points)
    
    # Calculate distance to origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance
    sorted_indices = np.argsort(dists)
    
    # Take the N closest points
    selected_indices = sorted_indices[:N]
    centers = points[selected_indices]
    
    # Compute minimum pairwise distance
    # N is 100, so N^2 is 10000, which is small
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
                
    # Compute max coordinate magnitude
    max_coord = np.max(np.abs(centers))
    
    # Calculate optimal radius and scale
    # r = 0.5 * d_min / (d_min + 2 * R_max)
    r = 0.5 * min_dist / (min_dist + 2 * max_coord)
    
    # Scale factor
    s = (0.5 - r) / max_coord
    
    centers = centers * s
    
    return centers
# EVOLVE-BLOCK-END
