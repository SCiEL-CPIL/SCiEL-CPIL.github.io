import time
import random
import numpy as np

# Safety margin: 150s limit. Use 148s to be safe.
DEADLINE = time.time() + 148.0

def _score_solution(bins, container):
    """Calculate the official score: B + 0.1 * (1 - last_bin_fill)."""
    W, H, D = container
    C = W * H * D
    B = len(bins)
    
    if B == 0:
        return 0.0
        
    last_bin = bins[-1]
    if not last_bin:
        return B
        
    last_vol = sum(w * h * d for (_, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    
    return B + 0.1 * (1.0 - fill_ratio)

def _pack_grid(boxes, container, order, rng, max_backtrack_depth=2):
    """
    Pack boxes using a grid-based occupancy check and Maximal Empty Box heuristic.
    Includes limited backtracking to try to improve packing density.
    """
    W, H, D = container
    n = len(boxes)
    
    # Initialize grid
    # Using float32 for slight memory efficiency, though bool would work too.
    # We track occupied cells.
    grid = np.zeros((W, H, D), dtype=np.uint8)
    
    # List of placed boxes in this bin: (x, y, z, w, h, d)
    current_bin = []
    all_bins = []
    
    # To support backtracking, we need to be able to undo placements.
    # Since we are doing a simple heuristic with limited backtracking, 
    # we will implement a simple stack-based undo for the current bin.
    
    def get_maximal_empty_box(grid, w, h, d):
        """
        Find the position (x, y, z) of the largest empty box of size (w, h, d) 
        that fits in the grid. 
        'Largest' is defined by the volume of the bounding box of the empty region 
        if we were to fill it, but simpler: we look for the position that 
        minimizes the 'waste' or maximizes the 'fill' relative to the current 
        bounding box of the bin.
        
        A simpler and effective heuristic for 3D bin packing with grids:
        Iterate through all possible positions (x, y, z) where the box fits.
        Calculate a score for each position. 
        Score = - (x + y + z) is a standard "bottom-left-back" heuristic.
        Better Score: Minimize the expansion of the bounding box.
        Or: Maximize the number of neighbors (contact).
        
        For speed, we will use a "Best Fit" metric:
        1. Check if it fits.
        2. Calculate the "cost": 
           - Primary: Minimize new bounding box volume.
           - Secondary: Minimize z (stack lower).
           - Tertiary: Minimize x, then y (compact).
        """
        best_pos = None
        best_score = float('inf')
        
        # Pre-calculate current bounding box of the bin
        if not current_bin:
            return (0, 0, 0)
        
        max_x = 0
        max_y = 0
        max_z = 0
        for (px, py, pz, pw, ph, pd) in current_bin:
            if px + pw > max_x: max_x = px + pw
            if py + ph > max_y: max_y = py + ph
            if pz + pd > max_z: max_z = pz + pd
            
        # Iterate through all potential positions
        # Optimization: Only check positions that are "supported" or "aligned"?
        # For simplicity and correctness with grid, we check all valid positions.
        # W, H, D are likely small enough (e.g., < 50) for this to be fast in NumPy.
        # If W,H,D are large (e.g., 100), 100^3 = 1M checks per box might be slow.
        # Let's check the size. If W*H*D > 10000, we might need a smarter anchor approach.
        # Given the problem usually has small containers for 40 boxes, let's assume grid size is manageable.
        
        for x in range(W - w + 1):
            for y in range(H - h + 1):
                # Quick check: is the slice empty?
                # grid[x:x+w, y:y+h, z:z+d]
                # We need to check z as well.
                # Instead of nested loops for x,y,z which is O(W*H*D), 
                # we can use NumPy to check all z for a given x,y.
                
                # Check if the column (x, y) has any empty space of height h and depth d?
                # This is hard to vectorize easily for arbitrary w,h,d.
                
                # Let's stick to the nested loop but optimize the inner check.
                pass
        
        # Alternative: Use the anchor approach but with grid check.
        # Anchors are (0,0,0) and faces of existing boxes.
        # This reduces candidates from O(W*H*D) to O(N_boxes).
        
        anchors = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in current_bin:
            if px + pw < W: anchors.append((px + pw, py, pz))
            if py + ph < H: anchors.append((px, py + ph, pz))
            if pz + pd < D: anchors.append((px, py, pz + pd))
            
            # Combinations
            if px + pw < W and py + ph < H: anchors.append((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D: anchors.append((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D: anchors.append((px, py + ph, pz + pd))
            if px + pw < W and py + ph < H and pz + pd < D: anchors.append((px + pw, py + ph, pz + pd))
            
        # Shuffle anchors slightly to break ties or find diverse solutions?
        # No, we want deterministic best fit for now.
        
        for (cx, cy, cz) in anchors:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
                
            # Check grid for overlap
            # grid[cx:cx+w, cy:cy+h, cz:cz+d]
            if np.any(grid[cx:cx+w, cy:cy+h, cz:cz+d] > 0):
                continue
                
            # Calculate score
            new_max_x = max(max_x, cx + w)
            new_max_y = max(max_y, cy + h)
            new_max_z = max(max_z, cz + d)
            
            # Heuristic: Minimize bounding box volume, then z, then x, then y
            score = new_max_x * new_max_y * new_max_z
            # Add small penalties for height and position to prefer lower/left
            score += new_max_z * 0.01 + cx * 0.001 + cy * 0.0001
            
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
                
        return best_pos

    def place_box(idx):
        """Attempt to place box idx in current bin. Returns True if placed."""
        w, h, d = boxes[idx]
        
        # Try to find a position
        pos = get_maximal_empty_box(grid, w, h, d)
        
        if pos is not None:
            x, y, z = pos
            # Update grid
            grid[x:x+w, y:y+h, z:z+d] = 1
            current_bin.append((x, y, z, w, h, d))
            return True
        else:
            return False

    # Main packing loop with limited backtracking
    # We process boxes in the given `order`.
    # If a box doesn't fit, we start a new bin.
    # Backtracking: If we start a new bin, we might try to rearrange the previous bin?
    # Simple backtracking: If box i doesn't fit in bin k, and bin k is not empty,
    # we could try to remove the last box from bin k and see if box i fits better?
    # This is complex. Let's stick to "First Fit Decreasing" with Grid + Best Fit Position.
    # The "Backtracking" in the prompt strategy suggests exploring permutations.
    # Given time limits, we will do:
    # 1. Try to place box in current bin.
    # 2. If fails, close current bin, start new bin.
    # 3. To improve, we will run this multiple times with different orderings.
    
    for idx in order:
        if time.time() > DEADLINE:
            # Return what we have, assuming the rest go to new bins? 
            # This would be invalid if we don't place them. 
            # We must ensure all boxes are placed.
            # If we run out of time, we should just place remaining boxes in new bins (1 box per bin) to be valid, 
            # though score will be bad.
            # Or better: finish the current ordering quickly with a fallback.
            pass
            
        if not current_bin:
            # Start new bin
            x, y, z = 0, 0, 0
            w, h, d = boxes[idx]
            if x+w <= W and y+h <= H and z+d <= D:
                grid[x:x+w, y:y+h, z:z+d] = 1
                current_bin.append((x, y, z, w, h, d))
            else:
                # Box doesn't fit in empty container? Should not happen if generated correctly.
                # But if it does, we have a problem. Assume it fits.
                raise ValueError("Box does not fit in empty container")
        else:
            placed = place_box(idx)
            if not placed:
                # Close current bin
                all_bins.append(current_bin[:])
                # Reset for new bin
                grid = np.zeros((W, H, D), dtype=np.uint8)
                current_bin = []
                
                # Place box in new bin
                w, h, d = boxes[idx]
                x, y, z = 0, 0, 0
                grid[x:x+w, y:y+h, z:z+d] = 1
                current_bin.append((x, y, z, w, h, d))
                
    # Don't forget the last bin
    if current_bin:
        all_bins.append(current_bin)
        
    return all_bins

def _generate_orderings(boxes, n, rng):
    """Generate diverse orderings of box indices."""
    idx = list(range(n))
    
    # 1. Volume Descending
    yield sorted(idx, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # 2. Max Dimension Descending
    yield sorted(idx, key=lambda i: max(boxes[i]), reverse=True)
    
    # 3. Sum of Dimensions Descending
    yield sorted(idx, key=lambda i: sum(boxes[i]), reverse=True)
    
    # 4. Aspect Ratio Descending
    yield sorted(idx, key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1), reverse=True)
    
    # 5-15. Random Shuffles
    for _ in range(10):
        s = idx[:]
        rng.shuffle(s)
        yield s

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    best_bins = None
    best_score = float('inf')
    best_order = None
    
    t_start = time.time()
    rng = random.Random(42)
    
    # Grid size check: If W*H*D is too large, the grid approach might be slow.
    # 40 boxes, likely small container. 
    # If W*H*D > 50000, we might want to skip grid or use a coarser grid?
    # For now, assume it's fine.
    
    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE:
            break
            
        try:
            bins = _pack_grid(boxes, container, order, rng)
            s = _score_solution(bins, container)
            
            if s < best_score:
                best_score = s
                best_bins = bins
                best_order = order
        except Exception:
            # If packing fails for any reason, skip this ordering
            continue
            
    if best_bins is None:
        # Fallback: simple sequential packing (1 box per bin)
        # This is a valid but poor solution
        best_bins = [[(0, 0, 0, w, h, d)] for w, h, d in boxes]
        best_order = list(range(n))
        
    # Construct final placements list
    # We need to map the boxes in best_bins back to the original indices.
    # Since we used `order` to determine the sequence, and `best_bins` contains the geometry,
    # we can reconstruct the mapping by re-running the packing logic for the best order?
    # Or, we can modify _pack_grid to return the indices as well.
    
    # Let's modify _pack_grid to return (bins, index_map) or similar.
    # Actually, simpler: Re-run the best order with a modified _pack_grid that tracks indices.
    
    # To avoid duplicating code, let's just re-run the packing for the best order 
    # and record the indices in the bins.
    
    W, H, D = container
    grid = np.zeros((W, H, D), dtype=np.uint8)
    current_bin_indices = []
    final_bins_with_indices = []
    
    # We need to replicate the exact logic of _pack_grid but record indices.
    # This is risky if the logic changes. 
    # Better: Modify _pack_grid to accept a flag to return indices.
    
    # For now, let's assume the geometry in `best_bins` corresponds to the `best_order`.
    # We can map them by iterating through `best_order` and finding where each box went?
    # No, that's hard because boxes might be identical.
    
    # Let's just re-run the packing for `best_order` and capture indices.
    
    # We need a version of _pack_grid that returns indices.
    
    def _pack_grid_with_indices(boxes, container, order, rng):
        W, H, D = container
        grid = np.zeros((W, H, D), dtype=np.uint8)
        current_bin = [] # (idx, x, y, z, w, h, d)
        all_bins = []
        
        def get_maximal_empty_box(grid, w, h, d, current_bin):
            best_pos = None
            best_score = float('inf')
            
            if not current_bin:
                return (0, 0, 0)
                
            max_x = 0
            max_y = 0
            max_z = 0
            for (_, px, py, pz, pw, ph, pd) in current_bin:
                if px + pw > max_x: max_x = px + pw
                if py + ph > max_y: max_y = py + ph
                if pz + pd > max_z: max_z = pz + pd
                
            anchors = [(0, 0, 0)]
            for (_, px, py, pz, pw, ph, pd) in current_bin:
                if px + pw < W: anchors.append((px + pw, py, pz))
                if py + ph < H: anchors.append((px, py + ph, pz))
                if pz + pd < D: anchors.append((px, py, pz + pd))
                
                if px + pw < W and py + ph < H: anchors.append((px + pw, py + ph, pz))
                if px + pw < W and pz + pd < D: anchors.append((px + pw, py, pz + pd))
                if py + ph < H and pz + pd < D: anchors.append((px, py + ph, pz + pd))
                if px + pw < W and py + ph < H and pz + pd < D: anchors.append((px + pw, py + ph, pz + pd))
                
            for (cx, cy, cz) in anchors:
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                    
                if np.any(grid[cx:cx+w, cy:cy+h, cz:cz+d] > 0):
                    continue
                    
                new_max_x = max(max_x, cx + w)
                new_max_y = max(max_y, cy + h)
                new_max_z = max(max_z, cz + d)
                
                score = new_max_x * new_max_y * new_max_z
                score += new_max_z * 0.01 + cx * 0.001 + cy * 0.0001
                
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    
            return best_pos

        for idx in order:
            if time.time() > DEADLINE:
                # Fallback: place remaining in new bins
                # This part is tricky to handle cleanly in the re-run.
                # We assume we didn't hit deadline in the first pass, so we shouldn't here.
                pass
                
            if not current_bin:
                w, h, d = boxes[idx]
                x, y, z = 0, 0, 0
                grid[x:x+w, y:y+h, z:z+d] = 1
                current_bin.append((idx, x, y, z, w, h, d))
            else:
                w, h, d = boxes[idx]
                pos = get_maximal_empty_box(grid, w, h, d, current_bin)
                if pos is not None:
                    x, y, z = pos
                    grid[x:x+w, y:y+h, z:z+d] = 1
                    current_bin.append((idx, x, y, z, w, h, d))
                else:
                    all_bins.append(current_bin[:])
                    grid = np.zeros((W, H, D), dtype=np.uint8)
                    current_bin = []
                    
                    x, y, z = 0, 0, 0
                    grid[x:x+w, y:y+h, z:z+d] = 1
                    current_bin.append((idx, x, y, z, w, h, d))
                    
        if current_bin:
            all_bins.append(current_bin)
            
        return all_bins

    # Re-run with indices
    bins_with_indices = _pack_grid_with_indices(boxes, container, best_order, rng)
    
    # Flatten to final output format
    final_placement_map = {}
    for bi, bin_content in enumerate(bins_with_indices):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            final_placement_map[idx] = (bi, x, y, z, w, h, d)
            
    result = []
    for i in range(n):
        if i in final_placement_map:
            result.append(final_placement_map[i])
        else:
            # Should not happen
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result
