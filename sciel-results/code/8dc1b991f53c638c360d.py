"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # 1. Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # bins is a list of lists. Each inner list contains (x, y, z, w, h, d, orig_idx)
    bins = []
    bin_vols = [] 
    bin_grids = []
    
    def fits_in_bin(bin_grid, x, y, z, w, h, d):
        """Check if box fits at position (x,y,z) without overlap and within bounds using grid."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if any part of the box overlaps with occupied grid cells
        if np.any(bin_grid[x:x+w, y:y+h, z:z+d]):
            return False
        return True

    def find_position(bin_items, bin_grid, w, h, d):
        """
        Find the best position for a box in a bin.
        Heuristic: Bottom-Left-Back (BLB). Minimize (x + y + z).
        Tie-breaker: Minimize (x * H * D + y * D + z) to prioritize lower x, then y, then z.
        """
        if not bin_items:
            return (0, 0, 0)
        
        candidates = [(0, 0, 0)]
        
        # Generate candidates based on faces and corners of existing boxes
        # We want to place boxes as close to the origin as possible.
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            # Faces
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            
            # Corners (combinations of face offsets)
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))
            
        best_pos = None
        best_score = float('inf')
        
        seen = set()
        for cx, cy, cz in candidates:
            if (cx, cy, cz) in seen:
                continue
            seen.add((cx, cy, cz))
            
            if fits_in_bin(bin_grid, cx, cy, cz, w, h, d):
                # Score: Bottom-Left-Back
                # Primary: sum of coordinates
                # Secondary: lexicographic order (x, then y, then z) to ensure consistency
                score = cx + cy + cz
                # To break ties strictly by BLB priority:
                # We can use a tuple (cx + cy + cz, cx, cy, cz)
                current_tuple = (score, cx, cy, cz)
                
                if best_pos is None:
                    best_pos = (cx, cy, cz)
                    best_score = current_tuple
                else:
                    # Compare tuples
                    if current_tuple < best_score:
                        best_score = current_tuple
                        best_pos = (cx, cy, cz)
                        
        return best_pos

    def place_box_in_bin(bin_idx, box_idx, w, h, d):
        """Attempts to place a box into a specific bin. Returns True if successful."""
        bin_items = bins[bin_idx]
        bin_grid = bin_grids[bin_idx]
        pos = find_position(bin_items, bin_grid, w, h, d)
        if pos is not None:
            x, y, z = pos
            bins[bin_idx].append((x, y, z, w, h, d, box_idx))
            bin_vols[bin_idx] += w*h*d
            bin_grid[x:x+w, y:y+h, z:z+d] = True
            return True
        return False

    def new_bin(box_idx, w, h, d):
        """Creates a new bin and places the box at (0,0,0)."""
        bins.append([(0, 0, 0, w, h, d, box_idx)])
        bin_vols.append(w*h*d)
        grid = np.zeros((W, H, D), dtype=bool)
        grid[0:w, 0:h, 0:d] = True
        bin_grids.append(grid)

    def score_solution():
        """Calculates the score for the current state."""
        if not bins:
            return float('inf')
        B = len(bins)
        # Last bin is the one with the highest index (which is len(bins)-1 since we append)
        # But wait, the problem says "largest label defines the last bin".
        # In our implementation, bins are 0-indexed and appended, so the last bin is indeed the last one.
        last_bin_idx = len(bins) - 1
        last_vol = bin_vols[last_bin_idx]
        container_vol = W * H * D
        fill = last_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - fill)

    deadline = time.time() + 145.0 # Safety margin

    # Initial Packing Phase
    for i, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback: place all remaining boxes in new bins
            for rem_i, rw, rh, rd, rv in indexed_boxes[i:]:
                new_bin(rem_i, rw, rh, rd)
            break
            
        best_bin_idx = -1
        best_pos = None
        best_bin_score = -1
        
        # Try to fit in existing bins
        for b_idx in range(len(bins)):
            bin_grid = bin_grids[b_idx]
            pos = find_position(bins[b_idx], bin_grid, w, h, d)
            if pos is not None:
                # Score for bin selection: Prefer bins that are already fuller
                # to minimize the number of bins used (B)
                current_bin_vol = bin_vols[b_idx]
                if current_bin_vol > best_bin_score:
                    best_bin_score = current_bin_vol
                    best_bin_idx = b_idx
                    best_pos = pos
                    
        if best_bin_idx == -1:
            # No existing bin works, start a new one
            new_bin(i, w, h, d)
        else:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, i))
            bin_vols[best_bin_idx] += w*h*d
            bin_grids[best_bin_idx][best_pos[0]:best_pos[0]+w, best_pos[1]:best_pos[1]+h, best_pos[2]:best_pos[2]+d] = True

    # Optimization Phase
    # Strategy 1: Try to move boxes from the last bin to other bins to reduce B
    # We iterate backwards from the last bin. If we can empty it, B decreases by 1.
    
    # We need to be careful with time.
    while time.time() < deadline - 5.0:
        if len(bins) <= 1:
            break
            
        last_bin_idx = len(bins) - 1
        last_bin_items = bins[last_bin_idx][:] # Copy to iterate
        
        # Try to move each box in the last bin to another bin
        # We iterate in reverse order of volume? Or just reverse index?
        # Let's try moving the largest boxes first from the last bin to others.
        # Sort items in last bin by volume descending
        last_bin_items_sorted = sorted(last_bin_items, key=lambda x: x[3]*x[4]*x[5], reverse=True)
        
        moved_any = False
        for item in last_bin_items_sorted:
            if time.time() > deadline - 5.0:
                break
                
            x, y, z, w, h, d, orig_idx = item
            
            # Remove from last bin
            # We need to update the grid
            bin_grids[last_bin_idx][x:x+w, y:y+h, z:z+d] = False
            bins[last_bin_idx].remove(item)
            bin_vols[last_bin_idx] -= w*h*d
            
            # Try to place in other bins (0 to last_bin_idx-1)
            placed = False
            for b_idx in range(last_bin_idx):
                if time.time() > deadline - 5.0:
                    break
                if place_box_in_bin(b_idx, orig_idx, w, h, d):
                    placed = True
                    break
            
            if not placed:
                # Put it back in the last bin
                # We need to find a valid position again? Or just re-add at same pos?
                # The grid was cleared, so same pos is valid if no other box moved there.
                # Since we check validity in place_box_in_bin, we can just try to place it back.
                # However, if we failed to place it elsewhere, it might still fit in the last bin.
                # But wait, if we failed to place it elsewhere, it definitely fits in the last bin
                # because it was there before.
                # We can just re-add it.
                place_box_in_bin(last_bin_idx, orig_idx, w, h, d)
            else:
                moved_any = True
                
        # If the last bin is now empty, remove it
        if not bins[last_bin_idx]:
            bins.pop()
            bin_vols.pop()
            bin_grids.pop()
        else:
            # If we didn't move any box, or the bin is not empty, we might stop to avoid infinite loops
            # where we move a box out and put it back.
            if not moved_any:
                break
            # If we moved some, the last bin might still have items.
            # The loop will continue to try to empty it further.
            # But if we can't empty it completely, we are stuck with B bins.
            # We should probably break if we can't empty the last bin.
            # Actually, the goal is to reduce B. If we can't empty the last bin, we can't reduce B.
            # So if the last bin is not empty after attempting to move all its items, we should stop.
            if bins[last_bin_idx]:
                 # Check if we made progress. If the volume of the last bin decreased, we might try again?
                 # No, if we can't empty it, we can't reduce B.
                 # So we break.
                 break

    # Construct final placements indexed by original box index
    placements = [None] * n
    
    for b_idx, bin_items in enumerate(bins):
        for item in bin_items:
            x, y, z, w, h, d, orig_idx = item
            placements[orig_idx] = (b_idx, x, y, z, w, h, d)
            
    # Ensure all boxes are placed
    for i in range(n):
        if placements[i] is None:
            # This should not happen, but as a safety net
            placements[i] = (len(bins), 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            # Add to a new bin
            w, h, d = boxes[i]
            bins.append([(0, 0, 0, w, h, d, i)])
            bin_vols.append(w*h*d)
            grid = np.zeros((W, H, D), dtype=bool)
            grid[0:w, 0:h, 0:d] = True
            bin_grids.append(grid)

    return placements
# EVOLVE-BLOCK-END
