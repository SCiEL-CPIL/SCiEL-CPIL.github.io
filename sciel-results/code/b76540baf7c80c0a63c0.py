"""
3D Bin Packing Solver
Strategy:
1. Multiple sorting heuristics for box ordering.
2. Greedy multi-bin packing with candidate positions derived from box corners.
3. Best-fit position selection within each bin.
4. Randomized restarts with mixed ordering weights.
5. Local improvement: try to move boxes to better positions or bins.
6. Deadline guard for time safety.
"""
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
    Positions are derived from the 'corners' of existing boxes and container boundaries.
    """
    candidates = set()
    
    # Always try (0,0,0)
    candidates.add((0, 0, 0))
    
    # Generate candidates from placed boxes
    for pb in placed_boxes:
        px, py, pz, pw, ph, pd = pb
        
        # X candidates: left of pb, right of pb
        cx1 = px - w
        cx2 = px + pw
        # Y candidates: front of pb, back of pb
        cy1 = py - h
        cy2 = py + ph
        # Z candidates: below pb, above pb
        cz1 = pz - d
        cz2 = pz + pd
        
        # Combine these to create a small set of likely spots
        for cx in [cx1, cx2]:
            for cy in [cy1, cy2]:
                for cz in [cz1, cz2]:
                    if cx >= 0 and cy >= 0 and cz >= 0:
                        if cx + w <= W and cy + h <= H and cz + d <= D:
                            candidates.add((cx, cy, cz))
        
        # Also try the "face" alignments
        if px >= 0 and py >= 0 and cz2 >= 0 and px + w <= W and py + h <= H and cz2 + d <= D:
            candidates.add((px, py, cz2))
        if px >= 0 and py >= 0 and cz1 >= 0 and px + w <= W and py + h <= H and cz1 + d <= D:
            candidates.add((px, py, cz1))
        if cx2 >= 0 and py >= 0 and pz >= 0 and cx2 + w <= W and py + h <= H and pz + d <= D:
            candidates.add((cx2, py, pz))
        if cx1 >= 0 and py >= 0 and pz >= 0 and cx1 + w <= W and py + h <= H and pz + d <= D:
            candidates.add((cx1, py, pz))
        if px >= 0 and cy2 >= 0 and pz >= 0 and px + w <= W and cy2 + h <= H and pz + d <= D:
            candidates.add((px, cy2, pz))
        if px >= 0 and cy1 >= 0 and pz >= 0 and px + w <= W and cy1 + h <= H and pz + d <= D:
            candidates.add((px, cy1, pz))

    # Filter valid candidates and sort by heuristic score
    valid_candidates = []
    for c in candidates:
        if c[0] >= 0 and c[1] >= 0 and c[2] >= 0:
            if c[0] + w <= W and c[1] + h <= H and c[2] + d <= D:
                valid_candidates.append(c)
                
    # Sort: Prefer lowest y (bottom), then lowest z (front), then lowest x (left)
    valid_candidates.sort(key=lambda p: (p[1], p[2], p[0]))
    return valid_candidates

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
                    score = pos[1] * 10000 + pos[2] * 100 + pos[0]
                    
                    if score < best_pos_score:
                        best_pos_score = score
                        best_bin_idx = b_idx
                        best_pos = pos
                    break
        
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

def _calc_score(result, container_vol):
    """Calculate the score for a given result."""
    if not all(r is not None for r in result):
        return float('inf')
    
    num_bins = max(r[0] for r in result) + 1
    last_bin_id = max(r[0] for r in result)
    last_bin_vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == last_bin_id)
    
    return num_bins + 0.1 * (1 - last_bin_vol / container_vol)

def _local_improve(result, boxes, W, H, D, n, container_vol, t_start, deadline):
    """
    Try to improve the solution by moving boxes to better positions or bins.
    """
    # Build bins from result
    bins = {}
    for i, (b, x, y, z, w, h, d) in enumerate(result):
        if b not in bins:
            bins[b] = []
        bins[b].append((i, x, y, z, w, h, d))
    
    improved = False
    max_iterations = 500
    
    for _ in range(max_iterations):
        if time.time() - t_start > deadline:
            break
        
        found_improvement = False
        for b_idx in sorted(bins.keys()):
            if time.time() - t_start > deadline:
                break
            
            bin_boxes = bins[b_idx]
            for i, (box_idx, x, y, z, w, h, d) in enumerate(bin_boxes):
                # Try to find a better position in the same bin
                others = [bb for j, bb in enumerate(bin_boxes) if j != i]
                candidates = _generate_candidates([(bb[1], bb[2], bb[3], bb[4], bb[5], bb[6]) for bb in others], w, h, d, W, H, D)
                
                best_new_pos = None
                best_new_score = None
                
                for pos in candidates:
                    new_box = (pos[0], pos[1], pos[2], w, h, d)
                    if _can_place([(bb[1], bb[2], bb[3], bb[4], bb[5], bb[6]) for bb in others], new_box, W, H, D):
                        score = pos[1] * 10000 + pos[2] * 100 + pos[0]
                        old_score = y * 10000 + z * 100 + x
                        
                        if best_new_score is None or score < best_new_score:
                            best_new_score = score
                            best_new_pos = pos
                
                if best_new_pos is not None and best_new_score < y * 10000 + z * 100 + x:
                    # Move to better position
                    result[box_idx] = (b_idx, best_new_pos[0], best_new_pos[1], best_new_pos[2], w, h, d)
                    bins[b_idx][i] = (box_idx, best_new_pos[0], best_new_pos[1], best_new_pos[2], w, h, d)
                    found_improvement = True
                    improved = True
        
        if not found_improvement:
            break
    
    return improved

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
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
    
    # 7. Volume ascending
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
            best_result = result
    
    # Now do randomized restarts with mixed orderings
    rng = random.Random(42)
    max_restarts = 300
    
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
            best_result = result
    
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
    
    # Try local improvement on the best result
    if best_result is not None and time.time() - t_start < DEADLINE - 5:
        _local_improve(best_result, boxes, W, H, D, n, container_vol, t_start, DEADLINE)
        best_score = _calc_score(best_result, container_vol)
    
    # Renumber bins to be consecutive
    bin_labels = sorted(set(r[0] for r in best_result))
    label_map = {old: new for new, old in enumerate(bin_labels)}
    best_result = [(label_map[r[0]], r[1], r[2], r[3], r[4], r[5], r[6]) for r in best_result]
    
    return best_result

# EVOLVE-BLOCK-END
