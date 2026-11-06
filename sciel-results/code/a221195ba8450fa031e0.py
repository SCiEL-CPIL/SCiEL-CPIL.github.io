import time
import random
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-process boxes: (index, w, h, d, volume)
    box_data = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    container_vol = W * H * D
    
    # Free Space Representation:
    # Each space is a tuple (x, y, z, w, h, d) representing an axis-aligned box
    # where new items can be placed.
    # When an item is placed, it splits existing spaces.
    
    def split_space(space, item):
        """
        Splits a free space by an item placed at (ix, iy, iz) with size (iw, ih, id).
        Returns a list of new free spaces.
        """
        sx, sy, sz, sw, sh, sd = space
        ix, iy, iz, iw, ih, id_ = item
        
        # Check if item is fully inside space
        if ix < sx or ix + iw > sx + sw:
            return [space] # Not fully inside, keep space (shouldn't happen if placement is valid)
        if iy < sy or iy + ih > sy + sh:
            return [space]
        if iz < sz or iz + id_ > sz + sd:
            return [space]
            
        # If item doesn't touch any boundary of the space, it splits into up to 6 spaces
        # Left
        if sx < ix:
            yield (sx, sy, sz, ix - sx, sh, sd)
        # Right
        if ix + iw < sx + sw:
            yield (ix + iw, sy, sz, sx + sw - (ix + iw), sh, sd)
        # Bottom
        if sy < iy:
            yield (sx, sy, sz, sw, iy - sy, sd)
        # Top
        if iy + ih < sy + sh:
            yield (sx, iy + ih, sz, sw, sy + sh - (iy + ih), sd)
        # Front
        if sz < iz:
            yield (sx, sy, sz, sw, sh, iz - sz)
        # Back
        if iz + id_ < sz + sd:
            yield (sx, sy, iz + id_, sw, sh, sz + sd - (iz + id_))
            
        # Note: The above simple splitting is not perfectly disjoint in all corners, 
        # but for packing heuristics, it's often sufficient if we check overlaps 
        # or if we use a more rigorous "maximal free space" update.
        # A simpler and more robust approach for small n is to just check 
        # against placed boxes. However, the prompt suggests free-space is faster.
        # Let's use a simpler "Corner Point" method but with a twist:
        # Instead of global corner points, we maintain a list of "candidate points" 
        # which are corners of placed boxes.
        # Actually, the previous program used corner points. 
        # The "inspiration" mentioned free-space reduces checking.
        # Let's stick to a robust corner-point method but optimize the search.
        
        # Wait, the directive says "STRUCTURAL change". 
        # Corner points with BFD is the current strategy.
        # Let's implement Local Search on the PERMUTATION of boxes.
        
        # To make it fast, I will use a greedy packer as the "fitness function" 
        # for a given permutation.
        # The greedy packer will use the corner-point method (fast for n=40).
        
        pass

    # Revised Strategy:
    # 1. Define a fast greedy packer that takes a list of box indices (order) and returns placements.
    # 2. Use Simulated Annealing to find a good order.
    # 3. Objective: Minimize (NumBins, -LastBinFill)
    
    def greedy_pack(order, time_limit=None):
        """
        Packs boxes in the given order using Best-Fit Decreasing (within bins) 
        and First-Fit (across bins) with corner point heuristic.
        Returns (placements_dict, num_bins, last_bin_vol)
        """
        placements = {}
        bins = []
        
        def new_bin():
            return {
                'placed': [], # (x, y, z, w, h, d)
                'corners': [(0, 0, 0)],
                'vol': 0
            }
        
        bins.append(new_bin())
        
        def has_overlap(px, py, pz, w, h, d, placed_boxes):
            for bx, by, bz, bw, bh, bd in placed_boxes:
                if px + w <= bx or bx + bw <= px:
                    continue
                if py + h <= by or by + bh <= py:
                    continue
                if pz + d <= bz or bz + bd <= pz:
                    continue
                return True
            return False

        def generate_corners(px, py, pz, w, h, d):
            return [
                (px + w, py, pz),
                (px, py + h, pz),
                (px, py, pz + d),
                (px + w, py + h, pz),
                (px + w, py, pz + d),
                (px, py + h, pz + d),
                (px + w, py + h, pz + d)
            ]

        for idx in order:
            w, h, d, vol = box_data[idx][1], box_data[idx][2], box_data[idx][3], box_data[idx][4]
            
            if time_limit and time.time() > time_limit:
                break
                
            best_bin_idx = -1
            best_pos = None
            best_score = float('inf')
            
            # Try to fit in existing bins
            for b_idx in range(len(bins)):
                bin_data = bins[b_idx]
                placed_boxes = bin_data['placed']
                corners = bin_data['corners']
                
                # Optimization: Skip if box is larger than remaining space? 
                # Hard to check quickly. Just iterate corners.
                
                for cx, cy, cz in corners:
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    if has_overlap(cx, cy, cz, w, h, d, placed_boxes):
                        continue
                    
                    # Score: Minimize coordinate sum (compactness)
                    # Tie-breaker: Prefer bins with more volume? Or lower index?
                    # To minimize B, we prefer existing bins. 
                    # To improve last bin fill, we want to pack tightly.
                    score = cx + cy + cz
                    
                    # Additional heuristic: Prefer bins that are "fuller" 
                    # to close them off? No, that might open new bins too early.
                    # Standard BFD: First Fit Decreasing is often good.
                    # Let's use: score = cx + cy + cz + 0.1 * b_idx (slight preference for early bins)
                    
                    if score < best_score:
                        best_score = score
                        best_bin_idx = b_idx
                        best_pos = (cx, cy, cz)
            
            if best_bin_idx == -1:
                # Open new bin
                bins.append(new_bin())
                best_bin_idx = len(bins) - 1
                best_pos = (0, 0, 0)
            
            px, py, pz = best_pos
            bin_data = bins[best_bin_idx]
            placements[idx] = (best_bin_idx, px, py, pz, w, h, d)
            
            bin_data['placed'].append((px, py, pz, w, h, d))
            bin_data['vol'] += vol
            
            # Update corners
            new_pts = generate_corners(px, py, pz, w, h, d)
            existing_set = set(bin_data['corners'])
            for pt in new_pts:
                if pt not in existing_set:
                    bin_data['corners'].append(pt)
                    existing_set.add(pt)
            
            # Prune corners: Remove any that are out of bounds? 
            # They are checked on use, but keeping list small is good.
            # For n=40, list size is < 300. Fine.

        num_bins = len(bins)
        last_bin_vol = bins[-1]['vol'] if bins else 0
        
        # If we didn't place all boxes due to timeout, we need to handle it.
        # But for n=40, greedy is very fast (<10ms). 
        # The SA loop is the bottleneck.
        
        return placements, num_bins, last_bin_vol

    # 1. Initial Solution: Sort by Volume Descending (BFD)
    initial_order = sorted(range(n), key=lambda i: box_data[i][4], reverse=True)
    init_placements, init_bins, init_last_vol = greedy_pack(initial_order)
    
    best_order = initial_order
    best_score = init_bins + 0.1 * (1 - init_last_vol / container_vol)
    best_placements = init_placements
    
    # 2. Simulated Annealing
    current_order = initial_order[:]
    current_placements, current_bins, current_last_vol = greedy_pack(current_order)
    current_score = current_bins + 0.1 * (1 - current_last_vol / container_vol)
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    # We can afford many iterations because greedy_pack is fast.
    # Estimate: 40 boxes * ~50 corners avg * 40 overlap checks = 80k ops per pack.
    # 80k ops is ~1ms in Python? Maybe 5-10ms.
    # 115s / 10ms = 11,500 iterations.
    
    while time.time() < deadline - 5.0:  # Stop 5s early to ensure return
        if T < min_T:
            break
            
        # Pick two random indices to swap
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_placements, new_bins, new_last_vol = greedy_pack(new_order)
        new_score = new_bins + 0.1 * (1 - new_last_vol / container_vol)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            # Accept
            current_order = new_order
            current_placements = new_placements
            current_bins = new_bins
            current_last_vol = new_last_vol
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_order = current_order
                best_placements = current_placements
                
        T *= cooling_rate
        
    # 3. Final Polish: Try a few specific heuristics if time permits
    # e.g., Sort by (Volume, MaxDim)
    if time.time() < deadline - 2.0:
        alt_order = sorted(range(n), key=lambda i: (box_data[i][4], max(box_data[i][1], box_data[i][2], box_data[i][3])), reverse=True)
        alt_pl, alt_bins, alt_vol = greedy_pack(alt_order)
        alt_score = alt_bins + 0.1 * (1 - alt_vol / container_vol)
        if alt_score < best_score:
            best_score = alt_score
            best_placements = alt_pl
            best_order = alt_order

    # Ensure all boxes are placed in the final best solution
    # If best_placements is missing some boxes (due to timeout in greedy?), 
    # we must fill them. But greedy_pack doesn't timeout for n=40.
    # It processes all boxes in order.
    
    # Construct return format
    # placements is dict {idx: (bin, x, y, z, w, h, d)}
    # Need list of tuples in original order? 
    # "one placement ... per box, preserving its original ordered dimensions"
    # The return type is usually a list of tuples corresponding to input order.
    
    result = [None] * n
    for idx, (b, x, y, z, w, h, d) in best_placements.items():
        result[idx] = (b, x, y, z, w, h, d)
        
    # Fallback for any missing (should not happen)
    for i in range(n):
        if result[i] is None:
            # Place in bin 0 at 0,0,0? No, might overlap.
            # Find first valid spot in bin 0 or create new bin.
            # Since best_placements came from a full greedy run, this shouldn't happen.
            # But to be safe:
            w, h, d = boxes[i]
            # Simple check
            placed_in_bin0 = [r for r in result if r is not None and r[0] == 0]
            # Just throw it in a new bin with high index to be safe?
            # No, that hurts score.
            # Let's re-run greedy for the whole set if we are missing boxes?
            # Actually, greedy_pack returns placements for all indices in order.
            # So result should be complete.
            pass

    return result
