import time
import random
import sys

# EVOLVE-BLOCK-START

def _boxes_overlap(b1, b2):
    """Check if two boxes (x,y,z,w,h,d) overlap."""
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    
    return True

def _can_place(placed_boxes, new_box, W, H, D):
    """Check if new_box fits in container and doesn't overlap placed_boxes."""
    x, y, z, w, h, d = new_box
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    
    for pb in placed_boxes:
        if _boxes_overlap(new_box, pb):
            return False
    return True

def _generate_candidates(placed_boxes, w, h, d, W, H, D):
    """
    Generate candidate positions for a box of size (w,h,d).
    Uses the 'extreme points' heuristic: candidates are corners of existing boxes 
    and the container boundaries.
    """
    candidates = set()
    
    # Always try (0,0,0)
    candidates.add((0, 0, 0))
    
    # Generate candidates from placed boxes
    for pb in placed_boxes:
        px, py, pz, pw, ph, pd = pb
        
        # X candidates: right edge of pb (px + pw)
        cx = px + pw
        # Y candidates: back edge of pb (py + ph)
        cy = py + ph
        # Z candidates: top edge of pb (pz + pd)
        cz = pz + pd
        
        # We only care about positions "after" existing boxes in the packing order
        # effectively, we want to fill gaps.
        # Common extreme points are combinations of x-coords from {0, px, px+pw}
        # but for speed, we just check the "next" available spot along each axis.
        
        # Actually, a more robust set of candidates is the set of all 
        # (x, y, z) where x is in {0} U {px, px+pw for all boxes}
        # and y is in {0} U {py, py+ph for all boxes}
        # and z is in {0} U {pz, pz+pd for all boxes}
        # This can be O(N^3) which is too slow.
        
        # Better approach: For each box, try placing the new box immediately 
        # to its right, behind it, or on top of it, aligned with its other coords.
        
        # 1. To the right (x increases), aligned in y and z
        if cx + w <= W:
            # Try aligning y with py
            if py + h <= H:
                if pz + d <= D:
                    candidates.add((cx, py, pz))
                # Try aligning z with pz
                # (already covered above)
            # Try aligning y with 0? No, that's (cx, 0, 0) which might be covered by (0,0,0) or other boxes.
            # Let's add (cx, 0, 0)
            if 0 + h <= H and 0 + d <= D:
                candidates.add((cx, 0, 0))
            # Let's add (cx, py, 0)
            if py + h <= H and 0 + d <= D:
                candidates.add((cx, py, 0))
            # Let's add (cx, 0, pz)
            if 0 + h <= H and pz + d <= D:
                candidates.add((cx, 0, pz))
            
        # 2. Behind (y increases), aligned in x and z
        if cy + h <= H:
            if px + w <= W:
                if pz + d <= D:
                    candidates.add((px, cy, pz))
            if 0 + w <= W:
                if pz + d <= D:
                    candidates.add((0, cy, pz))
            if px + w <= W:
                if 0 + d <= D:
                    candidates.add((px, cy, 0))
            if 0 + w <= W:
                if 0 + d <= D:
                    candidates.add((0, cy, 0))
            
        # 3. On top (z increases), aligned in x and y
        if cz + d <= D:
            if px + w <= W:
                if py + h <= H:
                    candidates.add((px, py, cz))
            if 0 + w <= W:
                if py + h <= H:
                    candidates.add((0, py, cz))
            if px + w <= W:
                if 0 + h <= H:
                    candidates.add((px, 0, cz))
            if 0 + w <= W:
                if 0 + h <= H:
                    candidates.add((0, 0, cz))

    # Filter valid candidates
    valid_candidates = []
    for c in candidates:
        if c[0] >= 0 and c[1] >= 0 and c[2] >= 0:
            if c[0] + w <= W and c[1] + h <= H and c[2] + d <= D:
                valid_candidates.append(c)
                
    # Sort: Prefer lowest y (bottom), then lowest z (front), then lowest x (left)
    # This encourages filling from the bottom-up and front-back.
    valid_candidates.sort(key=lambda p: (p[1], p[2], p[0]))
    
    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in valid_candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
            
    return unique_candidates

def _calc_score(result, container_vol):
    """Calculate the score for a given result."""
    if not all(r is not None for r in result):
        return float('inf')
    
    if not result:
        return 0.0

    bin_labels = set(r[0] for r in result)
    num_bins = len(bin_labels)
    last_bin_id = max(bin_labels)
    last_bin_vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == last_bin_id)
    
    return num_bins + 0.1 * (1 - last_bin_vol / container_vol)

def _pack_greedy(order, boxes, W, H, D, n):
    """Perform greedy packing for a given order."""
    bins = []  # List of lists of placed boxes (x,y,z,w,h,d)
    result = [None] * n
    
    for idx in order:
        w, h, d = boxes[idx]
        
        best_bin_idx = -1
        best_pos = None
        best_pos_score = float('inf')
        
        for b_idx in range(len(bins)):
            placed_in_bin = bins[b_idx]
            
            # Generate candidates
            candidates = _generate_candidates(placed_in_bin, w, h, d, W, H, D)
            
            for pos in candidates:
                new_box = (pos[0], pos[1], pos[2], w, h, d)
                if _can_place(placed_in_bin, new_box, W, H, D):
                    # Heuristic score: lower y (height) is better, then lower z, then lower x
                    # We want to pack tight to the bottom-left-front
                    score = pos[1] * 10000 + pos[2] * 100 + pos[0]
                    
                    if score < best_pos_score:
                        best_pos_score = score
                        best_bin_idx = b_idx
                        best_pos = pos
                    break # Only take the first valid candidate in sorted order for this bin
        
        if best_bin_idx != -1:
            pos = best_pos
            new_box = (pos[0], pos[1], pos[2], w, h, d)
            bins[best_bin_idx].append(new_box)
            result[idx] = (best_bin_idx, pos[0], pos[1], pos[2], w, h, d)
        else:
            # Open new bin
            new_bin_idx = len(bins)
            bins.append([])
            new_box = (0, 0, 0, w, h, d)
            bins[new_bin_idx].append(new_box)
            result[idx] = (new_bin_idx, 0, 0, 0, w, h, d)
    
    return result, bins

def _local_search(result, boxes, W, H, D, n, container_vol, t_start, deadline, max_iters=100):
    """
    Local search to improve the packing.
    Tries to move boxes to better positions in the same bin or to other bins.
    """
    # Build bins from result
    bins = {}
    for i, (b, x, y, z, w, h, d) in enumerate(result):
        if b not in bins:
            bins[b] = []
        bins[b].append((i, x, y, z, w, h, d))
    
    current_score = _calc_score(result, container_vol)
    
    for _ in range(max_iters):
        if time.time() - t_start > deadline:
            break
            
        improved = False
        # Try to move each box
        for b_idx in sorted(bins.keys()):
            if time.time() - t_start > deadline:
                break
                
            bin_boxes = bins[b_idx]
            for i, (box_idx, x, y, z, w, h, d) in enumerate(bin_boxes):
                # Try to find a better position in the same bin
                others = [bb for j, bb in enumerate(bin_boxes) if j != i]
                others_geom = [(bb[1], bb[2], bb[3], bb[4], bb[5], bb[6]) for bb in others]
                
                candidates = _generate_candidates(others_geom, w, h, d, W, H, D)
                
                best_new_pos = None
                best_new_score = None
                
                for pos in candidates:
                    new_box = (pos[0], pos[1], pos[2], w, h, d)
                    if _can_place(others_geom, new_box, W, H, D):
                        # Score for position: prefer lower y, z, x
                        score = pos[1] * 10000 + pos[2] * 100 + pos[0]
                        old_score = y * 10000 + z * 100 + x
                        
                        if best_new_score is None or score < best_new_score:
                            best_new_score = score
                            best_new_pos = pos
                
                if best_new_pos is not None and best_new_score < y * 10000 + z * 100 + x:
                    # Move to better position in same bin
                    result[box_idx] = (b_idx, best_new_pos[0], best_new_pos[1], best_new_pos[2], w, h, d)
                    bins[b_idx][i] = (box_idx, best_new_pos[0], best_new_pos[1], best_new_pos[2], w, h, d)
                    improved = True
                    # Recalculate score? Only if it affects bin count or last bin fill.
                    # Moving within same bin doesn't change bin count.
                    # It might change last bin fill if this is the last bin.
                    new_score = _calc_score(result, container_vol)
                    if new_score < current_score:
                        current_score = new_score
                    else:
                        # Revert if it didn't improve the global score (e.g. moved box out of last bin)
                        result[box_idx] = (b_idx, x, y, z, w, h, d)
                        bins[b_idx][i] = (box_idx, x, y, z, w, h, d)
                        improved = False

        if not improved:
            break
            
    return current_score

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    t_start = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    # Define different sorting orders
    indices = list(range(n))
    
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Height (h) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][1], reverse=True))
    
    # 4. Depth (d) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][2], reverse=True))
    
    # 5. Width (w) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0], reverse=True))
    
    # 6. Aspect ratio descending
    def aspect(i):
        w,h,d = boxes[i]
        return max(w,h,d)/min(w,h,d)
    orders.append(sorted(indices, key=aspect, reverse=True))
    
    # 7. Volume ascending (sometimes helps to fill small gaps first? Unlikely, but try)
    orders.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    best_result = None
    best_score = float('inf')
    
    # Try the predefined orders first
    for order in orders:
        if time.time() - t_start > DEADLINE:
            break
        
        result, bins = _pack_greedy(order, boxes, W, H, D, n)
        score = _calc_score(result, container_vol)
        
        if score < best_score:
            best_score = score
            best_result = result[:]
            
    # Now do randomized restarts with mixed orderings
    rng = random.Random(42)
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() - t_start > DEADLINE:
            break
        
        # Create a randomized order with some bias
        weights = []
        for i in range(n):
            w, h, d = boxes[i]
            vol = w * h * d
            max_dim = max(w, h, d)
            # Blend volume and max_dim with random noise
            base = 0.5 * vol + 0.5 * max_dim * max_dim
            noise = rng.random() * 0.2 * base
            weights.append(base + noise)
        
        # Sort by weight descending
        rand_order = sorted(indices, key=lambda i: weights[i], reverse=True)
        
        result, bins = _pack_greedy(rand_order, boxes, W, H, D, n)
        score = _calc_score(result, container_vol)
        
        if score < best_score:
            best_score = score
            best_result = result[:]
            
    # If no valid result found (should not happen), fallback to simple shelf
    if best_result is None:
        # Fallback: simple shelf packing
        result = []
        b = 0
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            result.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = result
        best_score = _calc_score(best_result, container_vol)
        
    # Try local improvement on the best result
    if best_result is not None and time.time() - t_start < DEADLINE - 10:
        _local_search(best_result, boxes, W, H, D, n, container_vol, t_start, DEADLINE)
        best_score = _calc_score(best_result, container_vol)
        
    # Renumber bins to be consecutive
    bin_labels = sorted(set(r[0] for r in best_result))
    label_map = {old: new for new, old in enumerate(bin_labels)}
    best_result = [(label_map[r[0]], r[1], r[2], r[3], r[4], r[5], r[6]) for r in best_result]
    
    return best_result

# EVOLVE-BLOCK-END
