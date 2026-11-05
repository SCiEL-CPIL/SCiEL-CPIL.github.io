import time
import math
import random

_START_TIME = time.time()
TIME_LIMIT = 145.0 # Safety margin for 150s limit

def _time_exceeded():
    return time.time() - _START_TIME > TIME_LIMIT

def _fits_at(boxes_in_bin, x, y, z, w, h, d):
    """Check if box (w,h,d) at (x,y,z) overlaps with any box in bin."""
    for (px, py, pz, pw, ph, pd) in boxes_in_bin:
        if (x < px + pw and px < x + w and
            y < py + ph and py < y + h and
            z < pz + pd and pz < z + d):
            return False
    return True

def _get_candidate_positions(boxes_in_bin, W, H, D, box_w, box_h, box_d):
    """
    Generate candidate (x, y, z) positions for a box in a bin.
    Uses the 'corner' method: candidates are 0 and (x_i + w_i), (y_i + h_i), (z_i + d_i)
    for all existing boxes i.
    """
    if not boxes_in_bin:
        return [(0, 0, 0)]
    
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (x, y, z, w, h, d) in boxes_in_bin:
        if x + w <= W: xs.add(x + w)
        if y + h <= H: ys.add(y + h)
        if z + d <= D: zs.add(z + d)
    
    candidates = []
    for x in xs:
        if x + box_w > W: continue
        for y in ys:
            if y + box_h > H: continue
            for z in zs:
                if z + box_d > D: continue
                candidates.append((x, y, z))
    
    return candidates

def _find_blb_position(boxes_in_bin, W, H, D, w, h, d):
    """
    Find the Bottom-Left-Back position for a box in a bin.
    Returns (x, y, z) or None if no fit.
    """
    cands = _get_candidate_positions(boxes_in_bin, W, H, D, w, h, d)
    # Sort by z (bottom), then y (left), then x (back)
    cands.sort(key=lambda c: (c[2], c[1], c[0]))
    
    for (cx, cy, cz) in cands:
        if _fits_at(boxes_in_bin, cx, cy, cz, w, h, d):
            return (cx, cy, cz)
    return None

def _pack_greedy(boxes, container, order):
    """
    Fast greedy packing.
    Returns: list of lists of (box_idx, x, y, z)
    """
    W, H, D = container
    n = len(boxes)
    bins = [] # List of lists of (idx, x, y, z)
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        
        # Try existing bins
        for bi in range(len(bins)):
            geom = [(x, y, z, boxes[bi_idx][0], boxes[bi_idx][1], boxes[bi_idx][2]) 
                    for (bi_idx, x, y, z) in bins[bi]]
            
            pos = _find_blb_position(geom, W, H, D, w, h, d)
            if pos:
                bins[bi].append((idx, pos[0], pos[1], pos[2]))
                placed = True
                break
        
        if not placed:
            # New bin
            if w <= W and h <= H and d <= D:
                bins.append([(idx, 0, 0, 0)])
            else:
                # Fallback: place at 0,0,0 even if it overflows (should not happen in valid instances)
                bins.append([(idx, 0, 0, 0)])
                
    return bins

def _calculate_score(bins, boxes, container):
    """
    Calculate the official score.
    B = number of occupied bins.
    Last bin fill = volume of last bin / container volume.
    """
    W, H, D = container
    C = W * H * D
    B = len(bins)
    if B == 0:
        return 0.0
    
    # The last bin is the one with the highest index in the compacted list
    last_vol = 0
    for (idx, x, y, z) in bins[-1]:
        w, h, d = boxes[idx]
        last_vol += w * h * d
        
    fill = last_vol / C if C > 0 else 0.0
    return B + 0.1 * (1.0 - fill)

def _compact_bins(bins):
    """Remove empty bins and re-index."""
    return [b for b in bins if b]

def _optimize_last_bin(bins, boxes, container):
    """
    Aggressive optimization to increase the fill ratio of the last bin.
    We try to move small boxes from earlier bins to the last bin if they fit.
    This directly improves the 0.1 * (1 - fill) term.
    """
    if len(bins) < 2:
        return bins
    
    W, H, D = container
    # Deep copy to avoid modifying original during trial
    cur_bins = [b[:] for b in bins]
    
    last_bin_idx = len(cur_bins) - 1
    last_bin = cur_bins[last_bin_idx]
    
    # Get geometry of last bin
    def get_geom(bin_list):
        return [(x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2]) for (idx, x, y, z) in bin_list]
    
    # We want to fill the last bin.
    # Identify boxes in other bins that are "small" and might fit in the last bin.
    # Sort candidate boxes from other bins by volume ascending (smallest first to pack tight).
    candidates = []
    for bi in range(len(cur_bins) - 1):
        for entry in cur_bins[bi]:
            idx = entry[0]
            w, h, d = boxes[idx]
            vol = w * h * d
            candidates.append((vol, bi, idx, entry[1], entry[2], entry[3]))
    
    candidates.sort(key=lambda x: x[0])
    
    improved = False
    for vol, src_bin, idx, sx, sy, sz in candidates:
        if _time_exceeded():
            break
            
        w, h, d = boxes[idx]
        
        # Check if it fits in the last bin
        last_geom = get_geom(last_bin)
        pos = _find_blb_position(last_geom, W, H, D, w, h, d)
        
        if pos:
            # It fits! Move it.
            # Remove from source bin
            # Find and remove
            for i, e in enumerate(cur_bins[src_bin]):
                if e[0] == idx:
                    cur_bins[src_bin].pop(i)
                    break
            
            # Add to last bin
            last_bin.append((idx, pos[0], pos[1], pos[2]))
            improved = True
            
            # Update last_geom for next iteration? 
            # _find_blb_position recalculates candidates each time, so it's safe but slow.
            # Since we are just checking fit, and n is small (40), it's okay.
            
            # Optional: Check if source bin is empty and compact?
            # We will compact at the end.
            
    # Compact bins
    return _compact_bins(cur_bins)

def _local_search_relocate(bins, boxes, container, max_iters=200):
    """
    Local search to reduce B by moving boxes from the last bin to earlier bins.
    If a box in the last bin fits in an earlier bin, move it.
    This might reduce B if the last bin becomes empty, or just improve fill if not.
    Actually, moving OUT of the last bin REDUCES the fill of the last bin, which INCREASES the score (bad).
    Wait. Score = B + 0.1 * (1 - fill).
    If we move a box OUT of the last bin:
    - B stays same (unless last bin becomes empty).
    - fill decreases -> (1 - fill) increases -> Score increases (worse).
    
    So we should NOT move boxes out of the last bin unless it reduces B.
    
    Conversely, moving boxes INTO the last bin (from earlier bins):
    - B stays same.
    - fill increases -> (1 - fill) decreases -> Score decreases (better).
    
    So _optimize_last_bin is the correct direction.
    
    What about reducing B?
    If we can move ALL boxes from the last bin into earlier bins, B decreases by 1.
    Score change: -1 + 0.1 * (1 - new_fill).
    Usually -1 is a big win.
    
    So, let's try to empty the last bin.
    """
    if len(bins) < 2:
        return bins
        
    W, H, D = container
    cur_bins = [b[:] for b in bins]
    
    for _ in range(max_iters):
        if _time_exceeded():
            break
            
        last_bin_idx = len(cur_bins) - 1
        last_bin = cur_bins[last_bin_idx]
        
        if not last_bin:
            cur_bins.pop()
            continue
            
        # Try to move each box in the last bin to an earlier bin
        moved_any = False
        boxes_to_move = last_bin[:] # Copy to iterate
        
        for entry in boxes_to_move:
            idx, x, y, z = entry
            w, h, d = boxes[idx]
            
            # Try to place in earlier bins
            placed = False
            for bi in range(last_bin_idx):
                geom = [(bx, by, bz, boxes[bidx][0], boxes[bidx][1], boxes[bidx][2]) 
                        for (bidx, bx, by, bz) in cur_bins[bi]]
                
                pos = _find_blb_position(geom, W, H, D, w, h, d)
                if pos:
                    # Move it
                    # Remove from last bin
                    for i, e in enumerate(cur_bins[last_bin_idx]):
                        if e[0] == idx:
                            cur_bins[last_bin_idx].pop(i)
                            break
                    # Add to earlier bin
                    cur_bins[bi].append((idx, pos[0], pos[1], pos[2]))
                    placed = True
                    moved_any = True
                    break
            
            if placed:
                # Check if last bin is empty
                if not cur_bins[last_bin_idx]:
                    cur_bins.pop()
                    # Recalculate last_bin_idx for next iteration? 
                    # The loop continues, so we just break out to restart logic
                    break
                    
        if not moved_any:
            break
            
    return _compact_bins(cur_bins)

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    rng = random.Random(42)
    
    # Generate diverse orderings
    orders = []
    # 1. Volume desc
    orders.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    # 2. Max dim desc
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    # 3. Min dim desc (helps with tight fits)
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    # 4. Random shuffles
    for _ in range(5):
        o = list(range(n))
        rng.shuffle(o)
        orders.append(o)
        
    best_bins = None
    best_score = float('inf')
    
    # Phase 1: Greedy Packing
    for order in orders:
        if _time_exceeded():
            break
        bins = _pack_greedy(boxes, container, order)
        
        # Phase 2: Optimize Last Bin Fill (Move small boxes INTO last bin)
        bins_optimized = _optimize_last_bin(bins, boxes, container)
        
        # Phase 3: Try to Reduce B (Move boxes OUT of last bin to earlier bins)
        # Note: This might undo some of Phase 2 if it moves large boxes out, 
        # but reducing B by 1 is worth 1.0 score, while fill is worth 0.1.
        # So we prioritize reducing B.
        bins_final = _local_search_relocate(bins_optimized, boxes, container)
        
        score = _calculate_score(bins_final, boxes, container)
        
        if score < best_score:
            best_score = score
            best_bins = [b[:] for b in bins_final]
            
    # If no valid packing found (should not happen), fallback
    if best_bins is None:
        best_bins = [[(i, 0, 0, 0)] for i in range(n)]
        best_score = _calculate_score(best_bins, boxes, container)
        
    # Final Polish: 
    # Sometimes, after reducing B, the new last bin might have low fill.
    # Run _optimize_last_bin again on the best result.
    if not _time_exceeded():
        polished_bins = _optimize_last_bin(best_bins, boxes, container)
        polished_score = _calculate_score(polished_bins, boxes, container)
        if polished_score < best_score:
            best_bins = polished_bins
            best_score = polished_score
            
    # Construct output
    result = []
    box_to_placement = {}
    for bi, b in enumerate(best_bins):
        for (idx, x, y, z) in b:
            w, h, d = boxes[idx]
            box_to_placement[idx] = (bi, x, y, z, w, h, d)
            
    for i in range(n):
        if i in box_to_placement:
            result.append(box_to_placement[i])
        else:
            # Should not happen
            result.append((0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result

# Test harness (not part of the submission)
if __name__ == "__main__":
    # Simple test
    boxes = [(1, 1, 1)] * 10
    container = (2, 2, 2)
    res = pack(boxes, container)
    print(res)
    # Check validity
    from collections import defaultdict
    bins = defaultdict(list)
    for b, x, y, z, w, h, d in res:
        bins[b].append((x, y, z, w, h, d))
        
    for b, items in bins.items():
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                x1, y1, z1, w1, h1, d1 = items[i]
                x2, y2, z2, w2, h2, d2 = items[j]
                if (x1 < x2 + w2 and x2 < x1 + w1 and
                    y1 < y2 + h2 and y2 < y1 + h1 and
                    z1 < z2 + d2 and z2 < z1 + d1):
                    print(f"Overlap in bin {b}: {items[i]} and {items[j]}")
    print("Test complete.")
