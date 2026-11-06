import time
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Preprocess boxes: Sort by volume descending (FFD)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Store placements in original index order
    placements = [None] * n
    
    # 2. Initialize Occupancy Grid
    # Grid resolution is 1 unit. If dimensions are large, this might be memory heavy,
    # but for typical container problems (e.g., 100x100x100), 1M cells is fine.
    # If dimensions are very large, we might need to scale or use a different structure,
    # but let's assume standard sizes first.
    # To be safe against memory limits for huge containers, we could check size.
    # 1000*1000*1000 = 1e9 bytes = 1GB. That's too much.
    # Let's check if W*H*D is reasonable.
    if W * H * D > 50_000_000:
         # Fallback to extreme points if grid is too big? Or just use sparse?
         # For now, let's stick to extreme points if grid is huge, but the prompt suggests grid.
         # Let's try to use a sparse approach or just limit grid usage.
         # Actually, the prompt says "3D occupancy grid". Let's assume reasonable sizes.
         # If it's too big, we'll crash. Let's add a check.
         # If W*H*D is too big, we might just use the extreme point method from before?
         # Or we can use a dictionary-based occupancy (sparse grid).
         pass

    # Let's use a dictionary for sparse occupancy to handle large coordinates efficiently
    # Key: (x, y, z), Value: 1 (occupied)
    # This is essentially a sparse grid.
    # However, checking overlap with a sparse grid is O(volume of box) which can be slow if boxes are large.
    # Alternative: Use a dense numpy array if size permits, else sparse.
    
    # Let's try a dense array if W*H*D < 100 million
    use_dense_grid = (W * H * D) < 100_000_000
    
    if use_dense_grid:
        grid = np.zeros((W, H, D), dtype=np.int8)
    else:
        grid = {} # Sparse: (x,y,z) -> 1

    placed_boxes = []
    
    # Helper to check if a box fits at (x,y,z) with dims (w,h,d)
    def fits(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        
        if use_dense_grid:
            # Check if any cell in the box volume is occupied
            # We can use a slice and check if any are non-zero
            # grid[x:x+w, y:y+h, z:z+d]
            # This creates a view, summing it is fast in numpy
            if np.any(grid[x:x+w, y:y+h, z:z+d]):
                return False
        else:
            # Sparse check: iterate over all cells in the box?
            # This is O(w*h*d). If boxes are large, this is slow.
            # But if grid is sparse, most cells are empty.
            # We can optimize by checking only if we have stored keys in that range.
            # For now, let's assume sparse case is rare or boxes are small.
            # Actually, iterating w*h*d is bad.
            # Let's stick to dense grid for performance, and if it's too big, we might need a different strategy.
            # Given the 120s limit, dense numpy is very fast.
            # Let's assume the problem instances are not astronomically large in dimension product.
            # If W*H*D is too big for memory, we'll fail.
            # Let's add a fallback to extreme points if dense grid is too big.
            return True # Placeholder, will replace with extreme point logic if needed

    # 3. Iterate through sorted boxes
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break
            
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        
        # Strategy: Find the lowest z, then lowest y, then lowest x where the box fits.
        # This is a "First Fit Decreasing" with a specific scanning order.
        # To do this efficiently with a grid, we can scan z from 0 to D-d.
        # For each z, scan y from 0 to H-h.
        # For each y, scan x from 0 to W-w.
        # This is O(D*H*W) per box in worst case, which is too slow.
        
        # Better Strategy:
        # The prompt suggests: "binary search on the z-axis to identify the lowest empty slot".
        # This implies a more structured approach.
        # Perhaps we maintain a "skyline" or "height map" for each (x,y) column?
        # If we project onto the XY plane, we can know the minimum Z for each (x,y).
        # But boxes have volume, so they block a range of Z for a range of X,Y.
        
        # Let's use a hybrid:
        # 1. Try to place at (0,0,z) for increasing z? No.
        # 2. Use the Extreme Points method but with a grid for overlap checking?
        #    The Extreme Points method generates O(N^2) points.
        #    Checking overlap with a dense grid is O(1) (using numpy sum on slice).
        #    So EP + Dense Grid is very fast.
        
        # Let's stick to Extreme Points but use the dense grid for O(1) overlap checks.
        # This avoids the O(N) overlap check per EP.
        
        extreme_points = [(0, 0, 0)]
        
        # We need to re-initialize extreme_points if we restart? No, it's persistent.
        # But I declared it inside the loop in the previous thought. Let's move it out.
        
        # Wait, I need to define extreme_points outside the loop.
        # Let's restructure.
        
        # Actually, let's just implement the EP method with grid overlap.
        
    # Let's rewrite the loop properly.
    
    # Reset for the real implementation
    if use_dense_grid:
        grid = np.zeros((W, H, D), dtype=np.int8)
    else:
        # If not dense, we fall back to a list of placed boxes for overlap check
        # because sparse grid overlap check is complex/slow in Python without C extension.
        pass
        
    placed_boxes_list = []
    extreme_points = [(0, 0, 0)]
    
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break
            
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        
        # Sort extreme points by (z, y, x) to prefer lower positions
        # This helps fill the bottom layers first.
        sorted_eps = sorted(extreme_points, key=lambda p: (p[2], p[1], p[0]))
        
        for px, py, pz in sorted_eps:
            if time.time() > deadline:
                break
                
            for ow, oh, od in orients:
                if px + ow > W or py + oh > H or pz + od > D:
                    continue
                
                valid = True
                if use_dense_grid:
                    # Check overlap with grid
                    if np.any(grid[px:px+ow, py:py+oh, pz:pz+od]):
                        valid = False
                else:
                    # Fallback: check against placed_boxes_list
                    for bx, by, bz, bw, bh, bd in placed_boxes_list:
                        if (px < bx + bw and bx < px + ow and
                            py < by + bh and by < py + oh and
                            pz < bz + bd and bz < pz + od):
                            valid = False
                            break
                
                if valid:
                    # Score: prefer lower z, then y, then x
                    score = pz * 1000000 + py * 1000 + px
                    if score < best_score:
                        best_score = score
                        best_placement = (px, py, pz, ow, oh, od)
                    
                    # Optional: Early exit if we find a placement at (0,0,0)?
                    # No, keep looking for better score (lower z/y/x)
                    # Since we sorted EPs by (z,y,x), the first valid one for a specific EP is likely good.
                    # But different EPs might offer better positions.
                    # We iterate all EPs to find the global best (lowest z/y/x).
        
        if best_placement is not None:
            px, py, pz, ow, oh, od = best_placement
            placements[idx] = (0, px, py, pz, ow, oh, od)
            placed_boxes_list.append((px, py, pz, ow, oh, od))
            
            if use_dense_grid:
                grid[px:px+ow, py:py+oh, pz:pz+od] = 1
            
            # Generate new extreme points
            new_points = [
                (px + ow, py, pz),
                (px, py + oh, pz),
                (px, py, pz + od)
            ]
            
            # Add new points to extreme_points, avoiding duplicates
            # To avoid O(N) check for duplicates, we can use a set if list is small,
            # or just check. With 60 boxes, list size ~180, so O(N) is fine.
            for p in new_points:
                if p not in extreme_points:
                    extreme_points.append(p)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (if deadline hit)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
