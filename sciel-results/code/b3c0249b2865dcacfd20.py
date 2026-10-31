"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import science_chemistry

def construct():
    start_time = time.time()
    timeout = 110.0  # 10s margin
    
    # 1. Generate FCC lattice points
    # FCC points are (i,j,k) with i+j+k even, and (i+0.5, j+0.5, k+0.5)
    # We generate a box of points
    points = []
    # Range -4 to 4 is sufficient for 38 atoms
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # Integer FCC points
                if (i + j + k) % 2 == 0:
                    points.append((float(i), float(j), float(k)))
                # Half-integer FCC points
                points.append((float(i) + 0.5, float(j) + 0.5, float(k) + 0.5))
    
    # Convert to array
    points = np.array(points)
    
    # 2. Calculate distances from origin
    dists = np.sum(points**2, axis=1)
    
    # 3. Sort by distance
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices]
    
    # 4. Select first 38
    points = points[:N]
    
    # 5. Scale to LJ minimum distance
    # In this raw lattice, nearest neighbor distance is sqrt(0.5)
    # e.g. (0,0,0) to (0.5, 0.5, 0) -> dx=0.5, dy=0.5, dz=0 -> dist=sqrt(0.5)
    # Target distance is 2^(1/6)
    scale_factor = (2.0 ** (1.0/6.0)) / np.sqrt(0.5)
    points = points * scale_factor
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    # 6. Relax
    # Check time
    if time.time() - start_time > timeout:
        return points
    
    try:
        relaxed_points = science_chemistry.relax(points)
    except Exception:
        relaxed_points = points
        
    # Check time again
    if time.time() - start_time > timeout:
        return points
        
    return relaxed_points
# EVOLVE-BLOCK-END
