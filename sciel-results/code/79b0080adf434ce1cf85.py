"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist


def construct():
    # We want to pack 21 circles in a rectangle with perimeter 4.
    # Perimeter = 2*(w+h) = 4 => w+h = 2.
    # We maximize sum of radii.
    
    # Strategy: Use a grid-based approach with optimization.
    # First, try a few aspect ratios and grid configurations.
    
    # Let's use a more sophisticated approach:
    # 1. Try different aspect ratios
    # 2. For each aspect ratio, place circles on a grid
    # 3. Optimize positions using local search
    
    # For now, let's use a heuristic approach:
    # Try to arrange circles in a roughly rectangular pattern
    
    # The seed program uses a 5x5 grid (since sqrt(21) ≈ 4.58, ceil = 5)
    # with 21 circles in 5x5 = 25 cells.
    
    # Let's try to improve by:
    # 1. Using a better initial arrangement
    # 2. Local optimization to push circles apart and increase radii
    
    # Initial arrangement: use a grid pattern
    # For 21 circles, we can use a 5x5 grid with 4 empty spots
    
    # Let's try a 4x6 arrangement (24 cells, use 21)
    # Or 5x5 with 4 removed
    
    # Actually, let's use a more flexible approach:
    # Start with a grid, then optimize positions and radii
    
    # Let's try multiple configurations and pick the best
    
    best_result = None
    best_sum = -1
    
    # Try different grid configurations
    configs = [
        (5, 5),  # 5x5 grid
        (4, 6),  # 4x6 grid
        (6, 4),  # 6x4 grid
        (3, 7),  # 3x7 grid
        (7, 3),  # 7x3 grid
        (2, 11), # 2x11 grid
        (11, 2), # 11x2 grid
    ]
    
    for nx, ny in configs:
        if nx * ny < N:
            continue
        
        # Try different aspect ratios
        for w_frac in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
            w = 2 * w_frac
            h = 2 - w
            
            # Create grid positions
            xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx)
            ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny)
            
            # Generate all grid positions
            positions = []
            for i in range(nx):
                for j in range(ny):
                    positions.append([xs[i], ys[j]])
            
            positions = np.array(positions[:N])
            
            # Initial radius estimate: half the minimum spacing
            # Compute pairwise distances
            if N > 1:
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                min_dists = np.min(dists, axis=1)
                r = np.min(min_dists) / 2 - 1e-6
            else:
                r = min(w, h) / 2 - 1e-6
            
            # Check boundary constraints
            # Circle at (x, y) with radius r must satisfy:
            # x >= r, x <= w - r, y >= r, y <= h - r
            valid = True
            for i in range(N):
                if positions[i][0] < r - 1e-9 or positions[i][0] > w - r + 1e-9:
                    valid = False
                    break
                if positions[i][1] < r - 1e-9 or positions[i][1] > h - r + 1e-9:
                    valid = False
                    break
            
            if not valid:
                # Adjust radius to fit boundaries
                for i in range(N):
                    max_r = min(positions[i][0], w - positions[i][0], 
                               positions[i][1], h - positions[i][1])
                    r = min(r, max_r - 1e-6)
            
            if r <= 0:
                continue
            
            # Local optimization: try to increase radius
            # Use a simple iterative approach
            for _ in range(50):
                # Compute current minimum distance between any two circles
                min_pair_dist = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d < min_pair_dist:
                            min_pair_dist = d
                
                # Also check boundary distances
                for i in range(N):
                    d_boundary = min(positions[i][0], w - positions[i][0],
                                    positions[i][1], h - positions[i][1])
                    if d_boundary < min_pair_dist:
                        min_pair_dist = d_boundary
                
                new_r = min_pair_dist / 2 - 1e-7
                if new_r <= r:
                    break
                r = new_r
                break  # Simple approach: just update radius once
            
            total_sum = N * r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = positions[:, 0]
                result[:, 1] = positions[:, 1]
                result[:, 2] = r
                best_result = result
    
    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
