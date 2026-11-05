import time
import numpy as np
from scipy import ndimage

def _can_place_grid(grid, x, y, z, w, h, d, W, H, D):
    """Check if a box fits at (x,y,z) using the grid."""
    if x + w > W or y + h > H or z + d > D:
        return False
    if bool(grid[x:x+w, y:y+h, z:z+d].any()):
        return False
    return True

def _find_position_bfd(box, grid, W, H, D):
    """
    Find the Best-Fit-Decreasing position (lowest z, then y, then x).
    Uses candidate positions derived from existing box boundaries.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Get candidate coordinates from existing placements
    # We need to find all positions where a box of size (w,h,d) could fit
    # Candidate x: 0 and x = px + pw for each existing box
    # Same for y and z
    
    # For efficiency, let's use a grid-based approach
    # We want the lexicographically smallest (z, y, x) position
    
    # Build candidate sets
    # First, find all occupied cells to derive boundaries
    # Actually, let's just iterate z from 0 up, y from 0 up, x from 0 up
    # but only at candidate positions to save time
    
    # For correctness and speed, let's use the approach of checking
    # candidate positions derived from box faces
    
    # Get all x-coordinates that are faces of existing boxes
    occupied = np.argwhere(grid)
    if len(occupied) == 0:
        return (0, 0, 0)
    
    # Extract unique x, y, z coordinates from occupied cells
    # But we need the "right" faces: x_max + 1, etc.
    # Actually, let's use a simpler but effective approach:
    # Scan z from 0 to D-d, and for each z, scan y from 0 to H-h, 
    # and for each (z,y), scan x from 0 to W-w
    # But use the grid to quickly check
    
    # More efficient: use the fact that valid positions must have
    # their "shadow" free. Let's use a 2D projection approach.
    
    # For each z level, compute a 2D map of where (w,h) rectangles can fit
    # in the slice, then combine across d slices
    
    # Let's use a direct but optimized scan
    # Start from z=0, y=0, x=0 and find the first valid position
    
    # To speed up, we can precompute for each z a 2D "free" map
    # free_2d[x, y] = True if grid[x:x+w, y:y+h, z:z+d] is all False
    
    # But building this for each z is expensive. Let's just do the scan
    # with early termination.
    
    for z in range(D - d + 1):
        # Check if the z-range is free
        # We need grid[:, :, z:z+d] to have a (w,h) free rectangle
        # Quick check: if any cell in grid[:, :, z:z+d] is True, we still
        # might have a free spot, so we can't skip entirely
        # But we can check if the entire z-slice range is free
        if not grid[:, :, z:z+d].any():
            # Entire z-range is free, so (0, 0, z) is valid
            return (0, 0, z)
        
        # Check if there's any free (w,h) rectangle in the z-range
        # We need to find if there exists (x,y) such that 
        # grid[x:x+w, y:y+h, z:z+d] is all False
        # This is equivalent to: for all dz in [0,d), 
        # grid[x:x+w, y:y+h, z+dz] has a free (w,h) rectangle at (x,y)
        
        # Let's compute the AND across d slices of the "can place (w,h) at (x,y)" map
        # For a single slice, can_place_2d[x, y] = not grid[x:x+w, y:y+h, zz].any()
        # We can use minimum_filter on the inverted grid slice
        
        # For each z in the range, compute a 2D map
        # Then AND them all together
        
        # This is still potentially expensive. Let's just do the direct scan
        # but with early exit.
        
        for y in range(H - h + 1):
            # Quick check: if grid[:, y:y+h, z:z+d] has no free x-range of length w
            # We can check if there's any x where grid[x:x+w, y:y+h, z:z+d] is all False
            # Let's scan x
            for x in range(W - w + 1):
                if not grid[x:x+w, y:y+h, z:z+d].any():
                    return (x, y, z)
    
    return None

def _find_position_best_fit(box, grid, W, H, D):
    """
    Find the position that maximizes contact with existing boxes and walls.
    This tends to pack more tightly.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Use the BFD approach as a baseline, but also consider other positions
    # For now, let's just use BFD which is simpler and often good enough
    return _find_position_bfd(box, grid, W, H, D)

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, with tie-breaking by max dimension
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda t: (t[1][0]*t[1][1]*t[1][2], max(t[1])), reverse=True)
    
    bins = []  # List of lists of placements
    bin_grids = []  # List of numpy boolean arrays
    result = [None] * n
    
    def _update_grid(grid, box, pos, add=True):
        x, y, z = pos
        w, h, d = box
        grid[x:x+w, y:y+h, z:z+d] = add
    
    # Phase 1: Initial Packing
    for orig_idx, box in indexed_boxes:
        if time.time() > deadline - 10.0:
            # Emergency: place remaining in new bins
            for oi, ob in indexed_boxes:
                if result[oi] is None:
                    bi = len(bins)
                    w, h, d = ob
                    bins.append([(bi, 0, 0, 0, w, h, d)])
                    grid = np.zeros((W, H, D), dtype=bool)
                    _update_grid(grid, ob, (0,0,0))
                    bin_grids.append(grid)
                    result[oi] = (bi, 0, 0, 0, w, h, d)
            break
            
        w, h, d = box
        placed = False
        
        if bins:
            # Try existing bins, sorted by remaining volume (tightest first)
            bin_vols = []
            for bi in range(len(bins)):
                vol = int(np.sum(bin_grids[bi]))
                bin_vols.append((W*H*D - vol, bi))
            bin_vols.sort(key=lambda x: x[0])
            
            for _, bi in bin_vols:
                if time.time() > deadline - 5.0:
                    break
                pos = _find_position_bfd(box, bin_grids[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    _update_grid(bin_grids[bi], box, pos)
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, box, (0,0,0))
            bin_grids.append(grid)
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
    
    # Ensure all placed
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, boxes[i], (0,0,0))
            bin_grids.append(grid)
            result[i] = (bi, 0, 0, 0, w, h, d)
    
    # Phase 2: Post-processing to improve last bin fill
    if time.time() < deadline - 5.0:
        last_bin_id = max(r[0] for r in result)
        
        moved_any = True
        while moved_any and time.time() < deadline - 3.0:
            moved_any = False
            for idx in range(n):
                if time.time() > deadline - 3.0:
                    break
                if result[idx][0] == last_bin_id:
                    continue
                
                cur_bin = result[idx][0]
                box = boxes[idx]
                w, h, d = box
                box_pos = result[idx][1:4]
                
                # Remove from source
                _update_grid(bin_grids[cur_bin], box, box_pos, add=False)
                
                # Try to place in last bin
                pos = _find_position_bfd(box, bin_grids[last_bin_id], W, H, D)
                
                if pos is not None:
                    x, y, z = pos
                    bins[last_bin_id].append((last_bin_id, x, y, z, w, h, d))
                    _update_grid(bin_grids[last_bin_id], box, pos)
                    # Remove from source bin list
                    for j, p in enumerate(bins[cur_bin]):
                        if p[1:4] == box_pos and p[4:7] == (w, h, d):
                            bins[cur_bin].pop(j)
                            break
                    result[idx] = (last_bin_id, x, y, z, w, h, d)
                    moved_any = True
                    break
                else:
                    # Restore
                    _update_grid(bin_grids[cur_bin], box, box_pos, add=True)
    
    # Phase 3: Try to reduce B by consolidating bins
    if time.time() < deadline - 5.0:
        # Try to empty the highest non-last bin
        occupied = [bi for bi in range(len(bins)) if bins[bi]]
        if len(occupied) > 1:
            # Sort by bin index descending, skip last_bin_id
            candidates = sorted([bi for bi in occupied if bi != last_bin_id], reverse=True)
            for src_bin in candidates:
                if time.time() > deadline - 5.0:
                    break
                boxes_to_move = list(bins[src_bin])
                can_move_all = True
                temp = []
                
                for p in boxes_to_move:
                    bi, x, y, z, w, h, d = p
                    box = (w, h, d)
                    
                    # Find box index
                    box_idx = None
                    for i in range(n):
                        if result[i][0] == bi and result[i][1:4] == (x, y, z) and result[i][4:7] == (w, h, d):
                            box_idx = i
                            break
                    if box_idx is None:
                        can_move_all = False
                        break
                    
                    _update_grid(bin_grids[src_bin], box, (x, y, z), add=False)
                    
                    placed = False
                    for dst_bin in range(len(bins)):
                        if dst_bin == src_bin:
                            continue
                        pos = _find_position_bfd(box, bin_grids[dst_bin], W, H, D)
                        if pos is not None:
                            nx, ny, nz = pos
                            bins[dst_bin].append((dst_bin, nx, ny, nz, w, h, d))
                            _update_grid(bin_grids[dst_bin], box, (nx, ny, nz))
                            result[box_idx] = (dst_bin, nx, ny, nz, w, h, d)
                            placed = True
                            break
                    
                    if not placed:
                        _update_grid(bin_grids[src_bin], box, (x, y, z), add=True)
                        can_move_all = False
                        break
                
                if can_move_all:
                    bins[src_bin] = []
                    last_bin_id = max(r[0] for r in result)
                    break
    
    # Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted([bi for bi in range(len(bins)) if bins[bi]])
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        for i in range(n):
            old_bin = result[i][0]
            if old_bin in label_map:
                new_bin = label_map[old_bin]
                result[i] = (new_bin, result[i][1], result[i][2], result[i][3],
                             result[i][4], result[i][5], result[i][6])
    
    return result
