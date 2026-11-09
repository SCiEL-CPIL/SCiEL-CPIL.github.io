"""
3D Bin Packing Solver
Strategy:
1. Generate multiple sorting orders for boxes (Volume, Max Dim, Height, etc.).
2. For each order, perform a greedy multi-bin packing using a Bottom-Left-Front 
   (BLF) style heuristic with exact overlap checks.
3. Candidate positions are generated from the corners of already placed boxes 
   and container boundaries, ensuring tight packing.
4. Use randomized restarts to explore different packing orders.
5. Select the solution with the best score (lowest B + 0.1*(1 - fill)).
6. Deadline guard ensures we return within time limits.
"""
import time
import sys
import random

# EVOLVE-BLOCK-START

def _boxes_overlap(b1, b2):
    """Check if two boxes (x,y,z,w,h,d) overlap."""
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    
    # Check separation in any axis
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    
    return True

def _can_place(placed_boxes, new_box, W, H, D):
    """Check if new_box (x,y,z,w,h,d) fits in container and doesn't overlap placed_boxes."""
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
    We prefer positions that are lower (y), then further back (z), then further left (x) 
    to maintain a compact, bottom-left-front structure.
    """
    candidates = set()
    
    # Always try (0,0,0)
    candidates.add((0, 0, 0))
    
    # Add boundaries
    if W >= w:
        candidates.add((W - w, 0, 0))
    if H >= h:
        candidates.add((0, H - h, 0))
    if D >= d:
        candidates.add((0, 0, D - d))
    
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
        # Try all combinations of x from {cx1, cx2}, y from {cy1, cy2}, z from {cz1, cz2}
        for cx in [cx1, cx2]:
            for cy in [cy1, cy2]:
                for cz in [cz1, cz2]:
                    if cx >= 0 and cy >= 0 and cz >= 0:
                        if cx + w <= W and cy + h <= H and cz + d <= D:
                            candidates.add((cx, cy, cz))
        
        # Also try the "face" alignments
        # e.g. same x, same y, different z
        if px >= 0 and py >= 0 and cz2 >= 0 and px + w <= W and py + h <= H and cz2 + d <= D:
            candidates.add((px, py, cz2)) # Above
        if px >= 0 and py >= 0 and cz1 >= 0 and px + w <= W and py + h <= H and cz1 + d <= D:
            candidates.add((px, py, cz1)) # Below
        if cx2 >= 0 and py >= 0 and pz >= 0 and cx2 + w <= W and py + h <= H and pz + d <= D:
            candidates.add((cx2, py, pz)) # Right
        if cx1 >= 0 and py >= 0 and pz >= 0 and cx1 + w <= W and py + h <= H and pz + d <= D:
            candidates.add((cx1, py, pz)) # Left
        if px >= 0 and cy2 >= 0 and pz >= 0 and px + w <= W and cy2 + h <= H and pz + d <= D:
            candidates.add((px, cy2, pz)) # Back
        if px >= 0 and cy1 >= 0 and pz >= 0 and px + w <= W and cy1 + h <= H and pz + d <= D:
            candidates.add((px, cy1, pz)) # Front

    # Filter valid candidates and sort by heuristic score
    valid_candidates = []
    for c in candidates:
        if c[0] >= 0 and c[1] >= 0 and c[2] >= 0:
            if c[0] + w <= W and c[1] + h <= H and c[2] + d <= D:
                valid_candidates.append(c)
                
    # Sort: Prefer lowest y (bottom), then lowest z (front), then lowest x (left)
    valid_candidates.sort(key=lambda p: (p[1], p[2], p[0]))
    return valid_candidates

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    t_start = time.time()
    DEADLINE = 110.0  # Safety margin
    
    # Define different sorting orders
    # boxes: list of (w, h, d)
    indices = list(range(n))
    
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Height (h) descending - packs tall boxes first to establish vertical structure
    orders.append(sorted(indices, key=lambda i: boxes[i][1], reverse=True))
    
    # 4. Depth (d) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][2], reverse=True))
    
    # 5. Width (w) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0], reverse=True))
    
    # 6. Aspect ratio (max/min) descending - "square" or "cubic" boxes first?
    def aspect(i):
        w,h,d = boxes[i]
        return max(w,h,d)/min(w,h,d)
    orders.append(sorted(indices, key=aspect, reverse=True))
    
    # 7. Volume ascending (sometimes helps fill gaps? Unlikely for first pass, but cheap to try)
    orders.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    best_result = None
    best_score = float('inf')
    
    def _do_greedy(order):
        """Perform greedy packing for a given order with best-fit position selection."""
        bins = [] # List of lists of placed boxes (x,y,z,w,h,d)
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
                        # Heuristic score: prefer lower y, then lower z, then lower x
                        # Normalize by container dimensions for scale-independence
                        score = (pos[1] / H) * 1000 + (pos[2] / D) * 100 + (pos[0] / W) * 10
                        
                        if score < best_pos_score:
                            best_pos_score = score
                            best_bin_idx = b_idx
                            best_pos = pos
                        break  # Only check first valid position per bin (candidates are sorted)
            
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
        
        return result
    
    def _calc_score(result):
        """Calculate the score for a given result."""
        if not all(r is not None for r in result):
            return float('inf')
        
        num_bins = max(r[0] for r in result) + 1
        last_bin_id = max(r[0] for r in result)
        last_bin_vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == last_bin_id)
        
        return num_bins + 0.1 * (1 - last_bin_vol / container_vol)
    
    # Try the predefined orders first
    for order in orders:
        if time.time() - t_start > DEADLINE:
            break
        
        result = _do_greedy(order)
        score = _calc_score(result)
        
        if score < best_score:
            best_score = score
            best_result = result
    
    # Now do randomized restarts with mixed orderings
    rng = random.Random(42)
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() - t_start > DEADLINE:
            break
        
        # Create a randomized order with varied strategies
        strategy = rng.randint(0, 4)
        
        if strategy == 0:
            # Volume + noise
            weights = []
            for i in range(n):
                w, h, d = boxes[i]
                vol = w * h * d
                noise = rng.random() * 0.3 * vol
                weights.append(vol + noise)
            rand_order = sorted(indices, key=lambda i: weights[i], reverse=True)
        elif strategy == 1:
            # Max dim squared + noise
            weights = []
            for i in range(n):
                w, h, d = boxes[i]
                max_dim = max(w, h, d)
                noise = rng.random() * 0.3 * max_dim * max_dim
                weights.append(max_dim * max_dim + noise)
            rand_order = sorted(indices, key=lambda i: weights[i], reverse=True)
        elif strategy == 2:
            # Surface area + noise
            weights = []
            for i in range(n):
                w, h, d = boxes[i]
                sa = 2*(w*h + h*d + w*d)
                noise = rng.random() * 0.3 * sa
                weights.append(sa + noise)
            rand_order = sorted(indices, key=lambda i: weights[i], reverse=True)
        elif strategy == 3:
            # Blend of volume and max dim with noise
            weights = []
            for i in range(n):
                w, h, d = boxes[i]
                vol = w * h * d
                max_dim = max(w, h, d)
                base = 0.5 * vol + 0.5 * max_dim * max_dim
                noise = rng.random() * 0.25 * base
                weights.append(base + noise)
            rand_order = sorted(indices, key=lambda i: weights[i], reverse=True)
        else:
            # Mostly random with slight volume bias
            weights = []
            for i in range(n):
                w, h, d = boxes[i]
                vol = w * h * d
                base = 0.1 * vol + rng.random()
                weights.append(base)
            rand_order = sorted(indices, key=lambda i: weights[i], reverse=True)
        
        result = _do_greedy(rand_order)
        score = _calc_score(result)
        
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
    
    return best_result

# EVOLVE-BLOCK-END
