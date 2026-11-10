# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.

    Construction:
        Points: 6 x 9 rectangular grid {1,...,6} x {1,...,9} -> 54 points.
        Lines:  Select the 27 lines (from horizontal, vertical, and diagonal directions
                with slopes in {-1, 0, 1}) that pass through the most grid points.
                
    The total number of incidences is 176.
    """
    
    # 1. Generate the 54 points (6x9 grid)
    n_x = 6
    n_y = 9
    xs = np.arange(1, n_x + 1)
    ys = np.arange(1, n_y + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate candidate lines
    # Lines are defined by A*x + B*y + C = 0.
    # We consider directions:
    #   Horizontal: 1*x + 0*y + C = 0  -> x = -C
    #   Vertical:   0*x + 1*y + C = 0  -> y = -C
    #   Slope 1:    -1*x + 1*y + C = 0 -> y = x - C
    #   Slope -1:    1*x + 1*y + C = 0 -> y = -x - C
    
    candidates = []
    
    # Horizontal lines (x = const)
    for x in range(1, n_x + 1):
        candidates.append((1, 0, -x))
        
    # Vertical lines (y = const)
    for y in range(1, n_y + 1):
        candidates.append((0, 1, -y))
        
    # Diagonal lines with slope 1 (y = x + c)
    # x - y + c = 0  => A=-1, B=1, C=c
    # x in [1,6], y in [1,9]
    # x - y ranges from 1-9=-8 to 6-1=5
    for c in range(-8, 6):
        candidates.append((-1, 1, c))
        
    # Diagonal lines with slope -1 (y = -x + c)
    # x + y + c = 0  => A=1, B=1, C=c
    # x + y ranges from 1+1=2 to 6+9=15
    for c in range(-15, -1):
        candidates.append((1, 1, c))
        
    candidates = np.array(candidates, dtype=np.int64)
    
    # 3. Count incidences for each candidate line
    # points: (N, 2)
    # candidates: (M, 3)
    # A*x + B*y + C = 0
    # val = x*A + y*B + C
    # We can compute this efficiently using broadcasting or einsum.
    # x = points[:, 0], y = points[:, 1]
    # val[i, j] = x[i]*A[j] + y[i]*B[j] + C[j]
    
    x_coords = points[:, 0].astype(np.int64)
    y_coords = points[:, 1].astype(np.int64)
    
    # val shape: (54, M)
    val = x_coords[:, None] * candidates[None, 0] + \
          y_coords[:, None] * candidates[None, 1] + \
          candidates[None, 2]
          
    # Count zeros
    counts = np.sum(val == 0, axis=0)
    
    # 4. Select the top 27 lines
    # If there are ties, we just pick the first 27 by index (deterministic enough for now,
    # but let's sort to be safe and consistent).
    # Actually, we want the lines with highest counts.
    # Sort indices by count descending, then by line tuple for determinism if counts are equal.
    # Since we just need any set of 27 with max sum, and all max count lines should be included
    # if possible, we can just take the top 27 by count.
    
    # Sort indices: primary key = count (desc), secondary key = line elements (asc)
    # To handle ties deterministically, we can create a sortable tuple.
    # Simpler: just argsort by count descending. If counts are equal, order is arbitrary but fixed by numpy.
    # To be fully deterministic, let's sort by count desc, then A, B, C asc.
    
    # Create a list of (count, A, B, C, index)
    line_info = []
    for i in range(len(candidates)):
        line_info.append((counts[i], candidates[i, 0], candidates[i, 1], candidates[i, 2], i))
    
    # Sort by count descending, then A, B, C ascending
    line_info.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    
    # Take first 27
    selected_indices = [info[4] for info in line_info[:27]]
    
    lines = candidates[selected_indices, :]
    
    # Ensure dtype is int64
    lines = lines.astype(np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
