"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box can be placed at position (x,y,z) in the given bin."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_spot(bin_placements, w, h, d):
        """Find the best spot for a box in a bin using full grid of candidate positions."""
        if not bin_placements:
            return (0, 0, 0)
        
        # Generate candidate positions from all edges
        xs = {0}
        ys = {0}
        zs = {0}
        for (px, py, pz, pw, ph, pd) in bin_placements:
            if px + pw <= W:
                xs.add(px + pw)
            if py + ph <= H:
                ys.add(py + ph)
            if pz + pd <= D:
                zs.add(pz + pd)
        
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        best_pos = None
        best_score = float('inf')
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_placements, x, y, z, w, h, d):
                        # Prefer positions that minimize wasted space
                        score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        return best_pos
    
    def pack_with_order(order):
        """Pack boxes in given order using first-fit with best-fit placement."""
        bins = []  # list of lists of placements
        placements = [None] * n
        
        for orig_idx, w, h, d in order:
            placed = False
            best_bin = -1
            best_pos = None
            best_waste = float('inf')
            
            for b_idx, bin_placements in enumerate(bins):
                spot = find_spot(bin_placements, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    waste = z * (H + 1) * (W + 1) + y * (W + 1) + x
                    if waste < best_waste:
                        best_waste = waste
                        best_bin = b_idx
                        best_pos = spot
            
            if best_bin >= 0:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                placements[orig_idx] = (best_bin, x, y, z, w, h, d)
            else:
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
        
        return placements, bins
    
    def score_placements(placements):
        """Calculate the score for a set of placements."""
        if not placements or all(p is None for p in placements):
            return float('inf')
        
        bin_ids = set(p[0] for p in placements if p is not None)
        B = len(bin_ids)
        last_bin = max(bin_ids)
        
        last_bin_vol = 0
        for p in placements:
            if p is not None and p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        
        container_vol = W * H * D
        if container_vol == 0:
            return float(B)
        
        last_bin_fraction = last_bin_vol / container_vol
        return B + 0.1 * (1 - last_bin_fraction)
    
    def local_search(placements, bins, max_iters=500):
        """Local search: try moving or swapping boxes between bins to improve score."""
        current_score = score_placements(placements)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try moving boxes to other bins
            box_list = list(range(n))
            random.shuffle(box_list)
            
            for i in box_list:
                if time.time() > deadline - 5:
                    return placements, current_score
                
                if placements[i] is None:
                    continue
                
                orig_bin, x, y, z, w, h, d = placements[i]
                
                # Try moving to another bin
                for target_bin in range(len(bins)):
                    if target_bin == orig_bin:
                        continue
                    
                    # Get placements in target bin
                    target_placements = list(bins[target_bin])
                    
                    spot = find_spot(target_placements, w, h, d)
                    if spot is not None:
                        # Tentatively move
                        old_placement = placements[i]
                        old_bin_items = list(bins[orig_bin])
                        
                        # Remove from old bin
                        bins[orig_bin] = [p for p in bins[orig_bin] 
                                         if not (p[0] == x and p[1] == y and p[2] == z 
                                                 and p[3] == w and p[4] == h and p[5] == d)]
                        
                        # Add to new bin
                        bins[target_bin].append((spot[0], spot[1], spot[2], w, h, d))
                        placements[i] = (target_bin, spot[0], spot[1], spot[2], w, h, d)
                        
                        new_score = score_placements(placements)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            # Revert
                            bins[target_bin].remove((spot[0], spot[1], spot[2], w, h, d))
                            bins[orig_bin] = old_bin_items
                            placements[i] = old_placement
                        
                        if improved:
                            break
            
            # Try swapping boxes between bins
            if not improved:
                box_list = list(range(n))
                random.shuffle(box_list)
                
                for i in box_list:
                    if time.time() > deadline - 5:
                        return placements, current_score
                    if placements[i] is None:
                        continue
                    
                    for j in box_list:
                        if time.time() > deadline - 5:
                            return placements, current_score
                        if i == j or placements[j] is None:
                            continue
                        
                        bin_i, xi, yi, zi, wi, hi, di = placements[i]
                        bin_j, xj, yj, zj, wj, hj, dj = placements[j]
                        
                        if bin_i == bin_j:
                            continue
                        
                        # Get placements in each bin without the boxes
                        bin_i_placements = [p for p in bins[bin_i] 
                                           if not (p[0] == xi and p[1] == yi and p[2] == zi 
                                                   and p[3] == wi and p[4] == hi and p[5] == di)]
                        bin_j_placements = [p for p in bins[bin_j] 
                                           if not (p[0] == xj and p[1] == yj and p[2] == zj 
                                                   and p[3] == wj and p[4] == hj and p[5] == dj)]
                        
                        # Try placing box i in bin_j and box j in bin_i
                        spot_i = find_spot(bin_j_placements, wi, hi, di)
                        if spot_i is None:
                            continue
                        
                        spot_j = find_spot(bin_i_placements, wj, hj, dj)
                        if spot_j is None:
                            continue
                        
                        # Tentatively swap
                        old_i = placements[i]
                        old_j = placements[j]
                        old_bin_i = list(bins[bin_i])
                        old_bin_j = list(bins[bin_j])
                        
                        bins[bin_i] = bin_i_placements + [(spot_j[0], spot_j[1], spot_j[2], wj, hj, dj)]
                        bins[bin_j] = bin_j_placements + [(spot_i[0], spot_i[1], spot_i[2], wi, hi, di)]
                        placements[i] = (bin_j, spot_i[0], spot_i[1], spot_i[2], wi, hi, di)
                        placements[j] = (bin_i, spot_j[0], spot_j[1], spot_j[2], wj, hj, dj)
                        
                        new_score = score_placements(placements)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            # Revert
                            bins[bin_i] = old_bin_i
                            bins[bin_j] = old_bin_j
                            placements[i] = old_i
                            placements[j] = old_j
                        
                        if improved:
                            break
                    
                    if improved:
                        break
        
        return placements, current_score
    
    def is_valid(placements):
        """Verify placements are valid."""
        if len(placements) != n:
            return False
        for p in placements:
            if p is None:
                return False
            b, x, y, z, w, h, d = p
            if b < 0 or x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
        # Check for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                bi, xi, yi, zi, wi, hi, di = placements[i]
                bj, xj, yj, zj, wj, hj, dj = placements[j]
                if bi == bj:
                    if (xi < xj + wj and xi + wi > xj and
                        yi < yj + hj and yi + hi > yj and
                        zi < zj + dj and zi + di > zj):
                        return False
        return True
    
    # Try multiple deterministic orderings
    orderings = []
    
    # 1. Volume descending
    order1 = sorted(indexed_boxes, key=lambda x: x[4], reverse=True)
    orderings.append([(i, w, h, d) for i, w, h, d, vol in order1])
    
    # 2. Max dimension descending
    order2 = sorted(indexed_boxes, key=lambda x: max(x[1], x[2], x[3]), reverse=True)
    orderings.append([(i, w, h, d) for i, w, h, d, vol in order2])
    
    # 3. Volume * max_dim descending
    order3 = sorted(indexed_boxes, key=lambda x: x[4] * max(x[1], x[2], x[3]), reverse=True)
    orderings.append([(i, w, h, d) for i, w, h, d, vol in order3])
    
    # 4. Min side ascending (smaller min side first for better shelf packing)
    order4 = sorted(indexed_boxes, key=lambda x: min(x[1], x[2], x[3]))
    orderings.append([(i, w, h, d) for i, w, h, d, vol in order4])
    
    # 5-12. Randomized restarts with perturbation
    for restart in range(8):
        random.seed(restart * 12345 + 67890)
        perturbed = []
        for i, w, h, d, vol in indexed_boxes:
            # Perturb volume for sorting
            pert_vol = vol * random.uniform(0.7, 1.3)
            perturbed.append((i, w, h, d, pert_vol))
        perturbed.sort(key=lambda x: x[4], reverse=True)
        orderings.append([(i, w, h, d) for i, w, h, d, vol in perturbed])
    
    # Try all orderings and keep the best
    best_placements = None
    best_score = float('inf')
    best_bins = None
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        
        placements, bins = pack_with_order(order)
        score = score_placements(placements)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            best_bins = bins
    
    # Local search on the best solution
    if best_placements is not None and time.time() < deadline - 5:
        best_placements, best_score = local_search(best_placements, best_bins, max_iters=800)
    
    # Verify validity
    if best_placements is None or not is_valid(best_placements):
        # Fallback to simple shelf packing
        best_placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            best_placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    return best_placements
# EVOLVE-BLOCK-END
