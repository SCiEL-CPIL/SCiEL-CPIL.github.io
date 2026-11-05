import time
import math

START_TIME = time.time()
TIME_LIMIT = 148.0  # Safety margin for 150s limit

def _now():
    return time.time()

def _time_exceeded():
    return _now() - START_TIME > TIME_LIMIT

def _fits_at(boxes_in_bin, cx, cy, cz, w, h, d, W, H, D):
    """
    Check if a box of size (w, h, d) fits at (cx, cy, cz) in a bin of size (W, H, D)
    without overlapping existing boxes.
    """
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    
    for (bx, by, bz, bw, bh, bd) in boxes_in_bin:
        # Overlap check
        if (cx < bx + bw and bx < cx + w and
            cy < by + bh and by < cy + h and
            cz < bz + bd and bz < cz + d):
            return False
    return True

def _get_mer_candidates(boxes_in_bin, W, H, D):
    """
    Generate candidate positions using Maximal Empty Rectangles (MER) logic.
    We approximate this by generating all valid "corner" positions derived from
    existing box boundaries. This is a standard heuristic for 3D packing.
    """
    candidates = [(0, 0, 0)]
    
    # Collect all unique x, y, z boundaries
    xs = {0, W}
    ys = {0, H}
    zs = {0, D}
    
    for (bx, by, bz, bw, bh, bd) in boxes_in_bin:
        xs.add(bx)
        xs.add(bx + bw)
        ys.add(by)
        ys.add(by + bh)
        zs.add(bz)
        zs.add(bz + bd)
        
    # To keep it fast, we don't iterate all combinations (O(N^3)), but rather
    # the "forward" faces of existing boxes, which is a subset and usually sufficient.
    # However, MER is more robust. Let's use the forward faces + origin.
    # This is effectively what the "anchors" in the previous solution did, but we
    # will order them by "bottom-left-back" priority to be deterministic and greedy.
    
    # For a more robust MER-like behavior without full O(N^3) cost:
    # We consider positions defined by the max coordinate of any box in the negative direction.
    # Simpler approach: Use the existing anchor logic but sort candidates by (x+y+z) to prefer
    # compact packing.
    
    for (bx, by, bz, bw, bh, bd) in boxes_in_bin:
        if bx + bw < W: candidates.append((bx + bw, by, bz))
        if by + bh < H: candidates.append((bx, by + bh, bz))
        if bz + bd < D: candidates.append((bx, by, bz + bd))
        
        # Corners
        if bx + bw < W and by + bh < H: candidates.append((bx + bw, by + bh, bz))
        if bx + bw < W and bz + bd < D: candidates.append((bx + bw, by, bz + bd))
        if by + bh < H and bz + bd < D: candidates.append((bx, by + bh, bz + bd))
        if bx + bw < W and by + bh < H and bz + bd < D: candidates.append((bx + bw, by + bh, bz + bd))
        
    # Remove duplicates
    unique_candidates = list(set(candidates))
    
    # Sort by "bottom-left-back" heuristic: lower x, y, z is better
    # This encourages filling from the corner, which is generally good for packing.
    unique_candidates.sort(key=lambda p: (p[0], p[1], p[2]))
    
    return unique_candidates

def _find_best_position(boxes_in_bin, w, h, d, W, H, D):
    """
    Find the best valid position for a box in a bin.
    Returns (x, y, z) or None.
    """
    if not boxes_in_bin:
        return (0, 0, 0)
        
    candidates = _get_mer_candidates(boxes_in_bin, W, H, D)
    
    best_pos = None
    best_score = float('inf')
    
    for (cx, cy, cz) in candidates:
        if _fits_at(boxes_in_bin, cx, cy, cz, w, h, d, W, H, D):
            # Heuristic: Prefer positions that are "lower" and "closer" to origin
            # This is a simple proxy for "bottom-left-back"
            score = cx + cy + cz
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
                
    return best_pos

def _calculate_score(bins, container):
    """
    Calculate the official score: B + 0.1 * (1 - last_bin_fill).
    bins: list of lists of (idx, x, y, z, w, h, d)
    """
    W, H, D = container
    C = W * H * D
    B = len(bins)
    
    if B == 0:
        return 0.0
        
    last_bin = bins[-1]
    if not last_bin:
        return B
        
    last_vol = sum(w * h * d for (_, _, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    
    return B + 0.1 * (1.0 - fill_ratio)

def _pack_greedy(boxes, container):
    """
    Phase 1: First-Fit Decreasing (FFD) using the geometric placement heuristic.
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort indices by volume descending
    idxs = list(range(n))
    idxs.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    bins = []
    for idx in idxs:
        w, h, d = boxes[idx]
        placed = False
        
        # Try existing bins
        for bi in range(len(bins)):
            # Extract geometry for fast fitting check
            geom = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[bi]]
            pos = _find_best_position(geom, w, h, d, W, H, D)
            if pos:
                bins[bi].append((idx, pos[0], pos[1], pos[2], w, h, d))
                placed = True
                break
        
        if not placed:
            # New bin
            bins.append([(idx, 0, 0, 0, w, h, d)])
            
    return bins

def _compact_last_bin(bins, container):
    """
    Phase 2: Aggressive Last-Bin Compaction.
    Try to move boxes from the last bin to previous bins if they fit,
    and then re-pack the last bin's remaining boxes to maximize fill.
    Actually, the score rewards a FULL last bin. So we want the last bin to be as full as possible.
    Wait, the score is B + 0.1 * (1 - fill_last).
    If we reduce B by 1, score drops by 1.
    If we increase fill_last, score drops by 0.1 * delta_fill.
    So reducing B is 10x more valuable than filling the last bin by 100%.
    However, if we cannot reduce B, we should maximize the fill of the last bin.
    
    Strategy:
    1. Try to move boxes FROM the last bin TO earlier bins. This might reduce B if the last bin becomes empty.
    2. If B doesn't change, try to rearrange boxes within the last bin to see if we can fit MORE boxes in it
       by moving boxes from EARLIER bins INTO the last bin? No, that increases B or keeps it same but changes which bin is "last".
       
    Actually, the "last bin" is the one with the highest index.
    If we have bins 0, 1, 2. Last is 2.
    If we move a box from bin 2 to bin 1, bin 2 might become empty -> B decreases by 1. HUGE WIN.
    If we move a box from bin 1 to bin 2, B stays same, but last bin (2) gets fuller. SMALL WIN.
    
    So the priority is:
    1. Empty the last bin by moving its contents to earlier bins.
    2. If last bin is not empty, try to fill it more by moving boxes from earlier bins into it?
       This only helps if the last bin was not full. But usually, the last bin is the one that "overflowed".
       Moving boxes into it might allow us to pack it tighter? No, if it's the last bin, it's the one with the least space left in previous bins.
       
    Let's focus on:
    1. Try to remove boxes from the last bin and place them in any earlier bin.
    2. If a box cannot be placed in earlier bins, it stays.
    3. If the last bin becomes empty, we drop it.
    
    Additionally, we can try to "swap" boxes between the last bin and earlier bins to improve geometry?
    That's complex. Let's stick to:
    - For each box in the last bin (in reverse order of volume, i.e., smallest first), try to move it to any other bin.
    - If it fits, move it.
    - If the last bin becomes empty, pop it.
    """
    W, H, D = container
    if not bins:
        return bins
    
    last_idx = len(bins) - 1
    if last_idx == 0:
        return bins
        
    last_bin = bins[last_idx]
    if not last_bin:
        bins.pop()
        return bins
        
    # Try to move boxes out of the last bin
    # Sort last bin boxes by volume ascending (try to move small ones first? or large? 
    # Small ones are easier to fit in gaps. Large ones are harder. 
    # Let's try all, but maybe small first is better for filling gaps in other bins.)
    items_in_last = list(last_bin)
    items_in_last.sort(key=lambda item: item[4]*item[5]*item[6]) # Volume ascending
    
    moved_any = True
    while moved_any and not _time_exceeded():
        moved_any = False
        # Re-evaluate last bin as items are moved
        if not bins[last_idx]:
            bins.pop()
            last_idx -= 1
            if last_idx < 0:
                break
            continue
            
        last_bin = bins[last_idx]
        # Make a copy to iterate
        for item in last_bin[:]:
            idx, x, y, z, w, h, d = item
            
            # Try to place this item in any bin before last_idx
            found = False
            for bi in range(last_idx):
                geom = [(xx, yy, zz, ww, hh, dd) for (_, xx, yy, zz, ww, hh, dd) in bins[bi]]
                pos = _find_best_position(geom, w, h, d, W, H, D)
                if pos:
                    # Move it
                    bins[bi].append((idx, pos[0], pos[1], pos[2], w, h, d))
                    # Remove from last bin
                    bins[last_idx].remove(item)
                    moved_any = True
                    found = True
                    break
                    
            if found:
                continue
                
    return bins

def _optimize_last_bin_fill(bins, container):
    """
    Phase 3: Try to increase the fill of the last bin by moving boxes FROM earlier bins INTO the last bin?
    This seems counter-intuitive if we want to reduce B, but if B is fixed, we want the last bin full.
    However, moving a box from bin i to bin j (where j is last) doesn't change B.
    It changes the volume in bin j.
    If bin j was not full, and we can fit a box from bin i into bin j, bin j becomes fuller.
    But bin i becomes emptier.
    Does this help?
    Score = B + 0.1 * (1 - V_last / V_container).
    Yes, if V_last increases, score decreases.
    
    So, we should try to pack the last bin as tightly as possible, even if it means taking boxes from other bins.
    But we must ensure we don't create an empty earlier bin (which would reduce B, which is GOOD, so that's fine).
    Actually, if we move a box from bin 0 to bin 2 (last), and bin 0 becomes empty, B drops by 1. HUGE WIN.
    So this phase is essentially: "Try to move any box from any bin to the last bin, provided it fits."
    And if that empties a bin, B drops.
    
    Let's implement:
    1. Identify the last bin.
    2. For each other bin, for each box in that bin, try to move it to the last bin.
    3. If it fits, move it.
    4. If the source bin becomes empty, remove it (and re-identify last bin).
    """
    W, H, D = container
    if not bins:
        return bins
        
    while not _time_exceeded():
        if len(bins) < 2:
            break
            
        last_idx = len(bins) - 1
        last_bin = bins[last_idx]
        last_geom = [(x, y, z, w, h, d) for (_, x, y, z, w, h, d) in last_bin]
        
        moved_any = False
        
        # Iterate over other bins
        for bi in range(last_idx):
            if not bins[bi]:
                continue
                
            # Get items in bin bi
            items = list(bins[bi])
            # Try to move items to last bin
            # Sort by volume descending? Large items are harder to fit but save more space if they do?
            # Actually, we just want to fill the last bin.
            for item in items:
                idx, x, y, z, w, h, d = item
                
                # Check if it fits in last bin
                pos = _find_best_position(last_geom, w, h, d, W, H, D)
                if pos:
                    # Move it
                    bins[last_idx].append((idx, pos[0], pos[1], pos[2], w, h, d))
                    bins[bi].remove(item)
                    # Update last_geom
                    last_geom.append((pos[0], pos[1], pos[2], w, h, d))
                    moved_any = True
                    break # Restart loop for this bin if we moved something? Or just continue?
                    # If we move one, the geometry changes. Let's break and restart the bin iteration to be safe.
                    
        if not moved_any:
            break
            
        # Clean up empty bins
        new_bins = []
        for b in bins:
            if b:
                new_bins.append(b)
            else:
                # If we removed a bin before the last, indices shift.
                pass
        # Actually, if we remove a bin, the "last bin" might change if we removed the actual last one?
        # No, we only move TO the last bin. So the last bin never becomes empty.
        # But earlier bins might become empty.
        # If an earlier bin becomes empty, B decreases.
        # We should compact the list.
        bins = [b for b in bins if b]
        
    return bins

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
        
    # Phase 1: Greedy FFD
    bins = _pack_greedy(boxes, container)
    
    # Phase 2: Compact last bin (try to empty it)
    bins = _compact_last_bin(bins, container)
    
    # Phase 3: Optimize last bin fill (try to fill it by taking from others)
    # This might seem contradictory to Phase 2, but Phase 2 tries to EMPTY it.
    # Phase 3 tries to FILL it.
    # If Phase 2 emptied it, B dropped. Phase 3 does nothing if B=1 or last bin is the only one.
    # If Phase 2 didn't empty it, Phase 3 tries to make it fuller.
    
    # Actually, Phase 3 logic: Move boxes INTO last bin.
    # This increases V_last.
    # It might empty other bins, reducing B.
    # So it's a double win potential.
    
    bins = _optimize_last_bin_fill(bins, container)
    
    # Final check and construction
    final_placement_map = {}
    for bi, bin_content in enumerate(bins):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            final_placement_map[idx] = (bi, x, y, z, w, h, d)
            
    result = []
    for i in range(n):
        if i in final_placement_map:
            result.append(final_placement_map[i])
        else:
            # Fallback
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result
