"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _can_place(box, x, y, z, placements, W, H, D):
    """Check if box at (x,y,z) overlaps with any existing placement."""
    w, h, d = box
    if x + w > W or y + h > H or z + d > D:
        return False
    for (_, ex, ey, ez, ew, eh, ed) in placements:
        if x < ex + ew and ex < x + w and \
           y < ey + eh and ey < y + h and \
           z < ez + ed and ez < z + d:
            return False
    return True

def _find_position(box, placements, W, H, D):
    """Find a valid position for box in bin. Uses bottom-left-back preference."""
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Generate candidate coordinates from existing box boundaries
    xs = set([0])
    ys = set([0])
    zs = set([0])
    for (_, ex, ey, ez, ew, eh, ed) in placements:
        if ex + ew <= W: xs.add(ex + ew)
        if ey + eh <= H: ys.add(ey + eh)
        if ez + ed <= D: zs.add(ez + ed)
    
    xs = sorted([x for x in xs if x + w <= W])
    ys = sorted([y for y in ys if y + h <= H])
    zs = sorted([z for z in zs if z + d <= D])
    
    if not xs or not ys or not zs:
        return None

    # Sort candidates to prioritize bottom (z), then back (y), then left (x)
    # Note: In standard 3D packing, usually z is up, y is depth, x is width.
    # The prompt doesn't specify axis orientation, but score depends on volume.
    # We assume z is the "height" axis for "filling" intuition, but geometrically
    # any axis works. We'll stick to z->y->x order for "lowest, then back, then left".
    best_pos = None
    best_score = float('inf')
    
    for z in zs:
        for y in ys:
            for x in xs:
                if _can_place(box, x, y, z, placements, W, H, D):
                    # Score: prefer lower z, then lower y, then lower x
                    # This creates a stable "corner" packing
                    score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                    # Early exit if we found the absolute best (0,0,0)
                    if x == 0 and y == 0 and z == 0:
                        return (0, 0, 0)
    
    return best_pos

def _constructive_pack(boxes, container, order):
    """
    Construct a packing given an order of box indices.
    Returns list of placements (bin, x, y, z, w, h, d) and the bin assignments.
    """
    W, H, D = container
    n = len(boxes)
    bins = [] # List of lists of (x, y, z, w, h, d)
    result = [None] * n
    
    for idx in order:
        box = boxes[idx]
        w, h, d = box
        placed = False
        
        # Try existing bins
        # Heuristic: Try bins with most remaining volume? Or just first fit?
        # First Fit Decreasing (FFD) is standard. Here we iterate bins 0..k
        for bi in range(len(bins)):
            pos = _find_position(box, bins[bi], W, H, D)
            if pos is not None:
                x, y, z = pos
                bins[bi].append((x, y, z, w, h, d))
                result[idx] = (bi, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Open new bin
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            result[idx] = (bi, 0, 0, 0, w, h, d)
            
    return result, bins

def _score(boxes, container, result):
    """Calculate the score: mean B + 0.1 * (1 - last-bin occupied volume fraction)."""
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return 0.0
        
    if not result:
        return 0.0

    # Count occupied bins
    bin_ids = set()
    for r in result:
        if r is not None:
            bin_ids.add(r[0])
    
    B = len(bin_ids)
    
    if B == 0:
        return 0.0

    last_bin = max(bin_ids)
    
    # Calculate volume in last bin
    last_bin_vol = 0
    for r in result:
        if r is not None and r[0] == last_bin:
            w, h, d = r[4], r[5], r[6]
            last_bin_vol += w * h * d
            
    fill_ratio = last_bin_vol / container_vol
    score = B + 0.1 * (1 - fill_ratio)
    return score

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Initial deterministic order: Volume Descending
    indices = list(range(n))
    indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    best_result, best_bins = _constructive_pack(boxes, container, indices)
    best_score = _score(boxes, container, best_result)
    
    # Local Search / Multi-Start
    # We will try to improve the solution by:
    # 1. Moving boxes to other bins (if it reduces B or improves last bin fill)
    # 2. Swapping boxes between bins
    # 3. Trying different initial orderings (randomized)
    
    # To manage time, we will alternate between:
    # A) Local search on the current best
    # B) New random constructive runs
    
    # Pre-compute box volumes
    vols = [b[0]*b[1]*b[2] for b in boxes]
    
    # Local Search Procedure
    def local_search(result, bins):
        """
        Attempts to improve the packing by moving or swapping boxes.
        Returns improved (result, bins) or original if no improvement.
        """
        current_score = _score(boxes, container, result)
        improved = True
        while improved:
            improved = False
            # Iterate through boxes
            for i in range(n):
                if result[i] is None:
                    continue
                bi, x, y, z, w, h, d = result[i]
                
                # Try to move box i to another bin
                # Target bins: all bins except bi
                # We prioritize bins that are "fuller" or lower index?
                # Let's try all bins.
                for tj in range(len(bins)):
                    if tj == bi:
                        continue
                    
                    # Remove box i from bin bi
                    # Find index in bins[bi]
                    remove_idx = None
                    for k, p in enumerate(bins[bi]):
                        if p[0]==x and p[1]==y and p[2]==z and p[3]==w and p[4]==h and p[5]==d:
                            remove_idx = k
                            break
                    if remove_idx is None:
                        continue
                    
                    bins[bi].pop(remove_idx)
                    
                    # Try to place in tj
                    pos = _find_position(boxes[i], bins[tj], W, H, D)
                    if pos is not None:
                        x2, y2, z2 = pos
                        bins[tj].append((x2, y2, z2, w, h, d))
                        new_result = result[:]
                        new_result[i] = (tj, x2, y2, z2, w, h, d)
                        
                        new_score = _score(boxes, container, new_result)
                        
                        if new_score < current_score - 1e-9:
                            # Accept
                            current_score = new_score
                            result = new_result
                            improved = True
                            # Note: bins is already modified correctly
                            break # Restart loop for this box? Or continue?
                            # Breaking inner loop, but we need to be careful with state.
                            # Since we modified bins, we can continue to next box.
                        else:
                            # Revert
                            bins[tj].pop()
                            bins[bi].insert(remove_idx, (x, y, z, w, h, d))
                            
                    # If we broke out due to improvement, we should probably restart the whole local search
                    # or at least re-evaluate. For simplicity, we'll just mark improved and break.
                    if improved:
                        break
                
                if improved:
                    break # Restart the while loop with new configuration
        
        return result, bins, current_score

    # Main Loop
    while time.time() < deadline - 5:
        
        # 1. Local Search on best
        if time.time() < deadline - 10:
            # Deep copy for safety? The function modifies bins in place.
            # Let's work on copies to avoid corrupting best_result if it fails
            # Actually, local_search above modifies bins and returns new result.
            # It's safer to copy.
            bins_copy = [list(b) for b in best_bins]
            res_copy = list(best_result)
            
            # Timebox local search
            ls_start = time.time()
            # We can't easily timebox the inner while loop of local_search without passing deadline.
            # Let's create a simplified local search step that is fast.
            
            # Single pass of moves
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                if res_copy[i] is None:
                    continue
                bi, x, y, z, w, h, d = res_copy[i]
                
                for tj in range(len(bins_copy)):
                    if tj == bi:
                        continue
                    
                    # Remove
                    remove_idx = None
                    for k, p in enumerate(bins_copy[bi]):
                        if p[0]==x and p[1]==y and p[2]==z and p[3]==w and p[4]==h and p[5]==d:
                            remove_idx = k
                            break
                    if remove_idx is None:
                        continue
                    
                    bins_copy[bi].pop(remove_idx)
                    pos = _find_position(boxes[i], bins_copy[tj], W, H, D)
                    if pos is not None:
                        x2, y2, z2 = pos
                        bins_copy[tj].append((x2, y2, z2, w, h, d))
                        new_res = res_copy[:]
                        new_res[i] = (tj, x2, y2, z2, w, h, d)
                        
                        # Quick score check
                        # Instead of full score, check if B decreased or last bin fill increased
                        # Full score is O(N), which is fine for N=40.
                        new_score = _score(boxes, container, new_res)
                        
                        if new_score < best_score - 1e-9:
                            best_score = new_score
                            best_result = new_res
                            best_bins = bins_copy
                            # Restart local search?
                            # For now, accept and continue
                            break
                        else:
                            bins_copy[tj].pop()
                            bins_copy[bi].insert(remove_idx, (x, y, z, w, h, d))
                    else:
                        bins_copy[bi].insert(remove_idx, (x, y, z, w, h, d))
            
            # 2. Random Restart / Constructive
            if time.time() < deadline - 5:
                # Generate a new random order
                # Mix of volume descending and random
                if random.random() < 0.5:
                    new_order = list(range(n))
                    random.shuffle(new_order)
                else:
                    # Perturbation of best order?
                    # Just use volume descending with some noise
                    new_order = list(range(n))
                    new_order.sort(key=lambda i: vols[i] + random.uniform(-0.1, 0.1)*max(vols), reverse=True)
                
                res_new, bins_new = _constructive_pack(boxes, container, new_order)
                score_new = _score(boxes, container, res_new)
                
                if score_new < best_score - 1e-9:
                    best_score = score_new
                    best_result = res_new
                    best_bins = bins_new
                else:
                    # If new solution is close, maybe local search it?
                    # To save time, only if it's very close
                    if score_new < best_score + 0.5 and time.time() < deadline - 15:
                         # Simple local search on new
                         # (Reusing the logic above is complex, so we'll just skip for now
                         #  unless we have lots of time)
                         pass

    # Final cleanup: Re-label bins to be consecutive
    if best_result:
        bin_to_label = {}
        new_label = 0
        for bi in range(len(best_bins)):
            if best_bins[bi]:
                bin_to_label[bi] = new_label
                new_label += 1
        
        for i in range(n):
            if best_result[i] is not None:
                old_bin = best_result[i][0]
                new_bin = bin_to_label.get(old_bin, 0)
                if new_bin != old_bin:
                    best_result[i] = (new_bin, best_result[i][1], best_result[i][2], best_result[i][3], 
                                    best_result[i][4], best_result[i][5], best_result[i][6])
                    
        return best_result
    
    return []
# EVOLVE-BLOCK-END
