"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Sort boxes by volume descending, then by largest dimension
    box_volumes = [w * h * d for w, h, d in boxes]
    box_order = sorted(range(n), key=lambda i: (-box_volumes[i], -max(boxes[i])))
    
    # Precompute all 6 orientations for each box
    # Orientation: (w, h, d)
    def get_orientations(w, h, d):
        # 6 unique rotations
        return [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
    
    # Grid-based approach: use a 3D occupancy grid for precise placement
    # But for efficiency, use a "skyline" / "support height" approach in 2D columns
    # Actually, let's use a simpler but effective approach: 
    # Try multiple placement strategies and keep the best
    
    best_volume = -1
    best_placements = None
    
    # Strategy 1: Greedy with best-fit decreasing, trying multiple orientations
    # Use a 3D grid for exact overlap checking
    # Grid resolution: 1 unit
    # Container dimensions W, H, D
    
    # For each box, we need to find a position (x, y, z) and orientation such that
    # the box fits without overlapping existing boxes.
    
    # Efficient approach: maintain a set of "free" positions or use a skyline per (x,y)
    
    # Let's use a different approach: maintain a list of placed boxes and check overlap
    # But that's O(N) per placement. With N=60, that's 60*60 = 3600 checks, which is fine.
    
    # Actually, let's use a 3D boolean array for occupancy
    # W, H, D could be large, but let's assume they're reasonable (e.g., < 1000)
    
    # Check if grid is feasible
    total_cells = W * H * D
    if total_cells > 10**8:
        # Grid too large, use list-based approach
        use_grid = False
    else:
        use_grid = True
    
    def pack_with_order(order, use_grid):
        """Try to pack boxes in the given order."""
        nonlocal deadline
        
        if use_grid:
            grid = np.zeros((W, H, D), dtype=bool)
        else:
            grid = None
            placed = []  # list of (x, y, z, w, h, d)
        
        placements = [None] * n
        total_vol = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            # Try to place this box
            placed_box = False
            
            if use_grid:
                # Try each orientation, find lowest z, then lowest y, then lowest x
                best_pos = None
                best_z = H + 1
                best_y = D + 1
                best_x = W + 1
                
                for ow, oh, od in orients:
                    if ow > W or oh > H or od > D:
                        continue
                    
                    # Find the position with minimum z, then y, then x
                    # Scan z from 0 to H-oh
                    for z in range(H - oh + 1):
                        if z >= best_z:
                            break
                        # Check if the box fits at (x, y, z) for some x, y
                        # We want the first (x, y) in row-major order
                        for y in range(D - od + 1):
                            if y >= best_y and z == best_z:
                                continue
                            for x in range(W - ow + 1):
                                # Check if grid[x:x+ow, y:y+od, z:z+oh] is all False
                                if not grid[x:x+ow, y:y+od, z:z+oh].any():
                                    if z < best_z or (z == best_z and y < best_y) or (z == best_z and y == best_y and x < best_x):
                                        best_z = z
                                        best_y = y
                                        best_x = x
                                        best_pos = (x, y, z, ow, oh, od)
                                    break  # Found first x for this (y, z), move to next y
                        if best_pos and best_pos[2] == z and best_pos[1] == y:
                            # Already found a position at this z, y; we can break if we want first
                            pass
                    
                    if best_pos:
                        break
                
                if best_pos:
                    x, y, z, ow, oh, od = best_pos
                    grid[x:x+ow, y:y+od, z:z+oh] = True
                    placements[idx] = (0, x, y, z, ow, oh, od)
                    total_vol += ow * oh * od
                    placed_box = True
            else:
                # List-based approach
                for ow, oh, od in orients:
                    if ow > W or oh > H or od > D:
                        continue
                    
                    # Try to find a position
                    # Scan all possible (x, y, z)
                    found = False
                    for z in range(H - oh + 1):
                        if found:
                            break
                        for y in range(D - od + 1):
                            if found:
                                break
                            for x in range(W - ow + 1):
                                # Check overlap with all placed boxes
                                overlap = False
                                for px, py, pz, pw, ph, pd in placed:
                                    if x < px + pw and x + ow > px and \
                                       y < py + pd and y + od > py and \
                                       z < pz + ph and z + oh > pz:
                                        overlap = True
                                        break
                                if not overlap:
                                    placed.append((x, y, z, ow, oh, od))
                                    placements[idx] = (0, x, y, z, ow, oh, od)
                                    total_vol += ow * oh * od
                                    placed_box = True
                                    found = True
                                    break
                    
                    if found:
                        break
            
            if not placed_box:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill in any unplaced boxes
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
        
        return placements, total_vol
    
    # Try the primary order: sorted by volume descending
    placements, vol = pack_with_order(box_order, use_grid)
    best_volume = vol
    best_placements = placements
    
    # Try a second order: sorted by aspect ratio (most cube-like first)
    if time.time() < deadline - 5:
        def aspect_ratio(box):
            w, h, d = box
            dims = sorted([w, h, d])
            return dims[2] / dims[0] if dims[0] > 0 else 0
        
        box_order2 = sorted(range(n), key=lambda i: (aspect_ratio(boxes[i]), -box_volumes[i]))
        placements2, vol2 = pack_with_order(box_order2, use_grid)
        if vol2 > best_volume:
            best_volume = vol2
            best_placements = placements2
    
    # Try a third order: sorted by largest dimension descending
    if time.time() < deadline - 5:
        box_order3 = sorted(range(n), key=lambda i: (-max(boxes[i]), -box_volumes[i]))
        placements3, vol3 = pack_with_order(box_order3, use_grid)
        if vol3 > best_volume:
            best_volume = vol3
            best_placements = placements3
    
    return best_placements
# EVOLVE-BLOCK-END
