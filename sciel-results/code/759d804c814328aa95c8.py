"""Skyline / floor-map based 3D container loading."""
import numpy as np
from itertools import product

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Use a 2D height map on the x-z plane for O(1) overlap checks.
    # The grid cell size is 1 unit. If W*D is too large, we fall back to a sparse approach.
    # For typical container sizes (e.g., up to 200x200), a 2D numpy array is efficient.
    
    # Check if grid approach is feasible
    grid_feasible = (W * D <= 1000000)  # 1M cells is reasonable
    
    if grid_feasible:
        # 2D array: height_map[x][z] = max height of any box covering that cell
        height_map = np.zeros((W, D), dtype=np.int64)
    else:
        # Fallback: use a dictionary for sparse representation
        # height_map[(x, z)] = max height
        height_map = {}
    
    placed = []  # list of (x, y, z, w, h, d)
    final_placements = [None] * n
    
    # Sort boxes by volume descending for better packing
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        
        # Generate 6 rotations
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
        
        best_placement = None
        best_y = float('inf')
        
        for rw, rh, rd in rotations:
            if rw > W or rh > H or rd > D:
                continue
            
            # Generate candidate x and z positions
            # Use edges of placed boxes plus 0
            xs = {0}
            zs = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                xs.add(px + pw)
                zs.add(pz + pd)
            
            # Filter to valid positions
            xs = sorted([x for x in xs if x + rw <= W])
            zs = sorted([z for z in zs if z + rd <= D])
            
            for cx in xs:
                for cz in zs:
                    # Find the minimum y (max height in the footprint)
                    if grid_feasible:
                        # Use numpy slicing for fast max computation
                        x1, x2 = cx, cx + rw
                        z1, z2 = cz, cz + rd
                        footprint = height_map[x1:x2, z1:z2]
                        min_y = int(np.max(footprint))
                    else:
                        # Sparse approach: check each cell in the footprint
                        min_y = 0
                        for x in range(cx, cx + rw):
                            for z in range(cz, cz + rd):
                                key = (x, z)
                                if key in height_map:
                                    if height_map[key] > min_y:
                                        min_y = height_map[key]
                    
                    if min_y + rh <= H:
                        if min_y < best_y:
                            best_y = min_y
                            best_placement = (cx, min_y, cz, rw, rh, rd)
        
        if best_placement is not None:
            cx, cy, cz, rw, rh, rd = best_placement
            placed.append((cx, cy, cz, rw, rh, rd))
            final_placements[idx] = (0, cx, cy, cz, rw, rh, rd)
            
            # Update the height map
            if grid_feasible:
                for x in range(cx, cx + rw):
                    for z in range(cz, cz + rd):
                        if cy + rh > height_map[x, z]:
                            height_map[x, z] = cy + rh
            else:
                for x in range(cx, cx + rw):
                    for z in range(cz, cz + rd):
                        key = (x, z)
                        if key not in height_map or cy + rh > height_map[key]:
                            height_map[key] = cy + rh
        else:
            final_placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    return final_placements
