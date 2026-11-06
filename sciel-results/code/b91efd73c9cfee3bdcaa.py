"""Rewritten 3D bin packing with stronger heuristics."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin within 120s
    
    container_vol = W * H * D
    if container_vol <= 0:
        return [(0, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Precompute box volumes
    volumes = [w * h * d for w, h, d in boxes]
    
    def get_candidate_positions(bin_placements, w, h, d):
        """Generate candidate positions for a box in a bin."""
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Positions along each axis offset by existing boxes
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            # Corner combinations
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            # Also try positions aligned with box faces
            candidates.add((px, py, pz))
        
        # Filter valid positions
        valid = []
        for (x, y, z) in candidates:
            if x >= 0 and y >= 0 and z >= 0:
                if x + w <= W and y + h <= H and z + d <= D:
                    valid.append((x, y, z))
        
        return valid
    
    def check_overlap(bin_placements, x, y, z, w, h, d):
        """Check if a box at (x,y,z) overlaps with any existing placement."""
        for (px, py, pz, pw, ph, pd) in bin_placements:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return True
        return False
    
    def find_best_position(bin_placements, w, h, d):
        """Find the best position for a box in a bin using scoring."""
        candidates = get_candidate_positions(bin_placements, w, h, d)
        
        if not candidates:
            return None
        
        best_pos = None
        best_score = float('inf')
        
        for (x, y, z) in candidates:
            if check_overlap(bin_placements, x, y, z, w, h, d):
                continue
            
            # Score: prefer lower z, then lower y, then lower x
            # Also prefer positions that maximize contact with existing boxes
            score = z * 1000 + y * 100 + x
            
            # Bonus for being close to existing boxes (tighter packing)
            for (px, py, pz, pw, ph, pd) in bin_placements:
                # Check if this position is adjacent to an existing box
                if (x + w == px or px + pw == x) and \
                   (y + h >= py and y <= py + ph) and \
                   (z + d >= pz and z <= pz + pd):
                    score -= 10
                if (y + h == py or py + ph == y) and \
                   (x + w >= px and x <= px + pw) and \
                   (z + d >= pz and z <= pz + pd):
                    score -= 10
                if (z + d == pz or pz + pd == z) and \
                   (x + w >= px and x <= px + pw) and \
                   (y + h >= py and y <= py + ph):
                    score -= 10
            
            if score < best_score:
                best_score = score
                best_pos = (x, y, z)
        
        return best_pos
    
    def pack_with_strategy(sort_key, random_seed=None):
        """Pack boxes using a specific sorting strategy."""
        if random_seed is not None:
            random.seed(random_seed)
        
        indexed_boxes = [(i, w, h, d, volumes[i]) for i, (w, h, d) in enumerate(boxes)]
        
        if sort_key == 'volume_desc':
            indexed_boxes.sort(key=lambda x: x[4], reverse=True)
        elif sort_key == 'max_dim_desc':
            indexed_boxes.sort(key=lambda x: max(x[1], x[2], x[3]), reverse=True)
        elif sort_key == 'surface_area_desc':
            indexed_boxes.sort(key=lambda x: 2*(x[1]*x[2] + x[2]*x[3] + x[1]*x[3]), reverse=True)
        elif sort_key == 'aspect_ratio_desc':
            indexed_boxes.sort(key=lambda x: max(x[1], x[2], x[3]) / max(1, min(x[1], x[2], x[3])), reverse=True)
        elif sort_key == 'volume_then_max':
            indexed_boxes.sort(key=lambda x: (x[4], max(x[1], x[2], x[3])), reverse=True)
        elif sort_key == 'random':
            random.shuffle(indexed_boxes)
        else:
            indexed_boxes.sort(key=lambda x: x[4], reverse=True)
        
        bins = []  # List of lists of (x, y, z, w, h, d)
        placements = [None] * n
        
        for (orig_idx, w, h, d, vol) in indexed_boxes:
            placed = False
            
            for b_idx in range(len(bins)):
                pos = find_best_position(bins[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((x, y, z, w, h, d))
                    placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
        
        return placements, bins
    
    def compute_score(placements):
        """Compute the score for a placement."""
        if not placements or any(p is None for p in placements):
            return float('inf')
        
        occupied_bins = set(p[0] for p in placements)
        B = len(occupied_bins)
        
        if B == 0:
            return 0.0
        
        last_bin = max(occupied_bins)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
        last_bin_fraction = last_bin_vol / container_vol
        
        return B + 0.1 * (1 - last_bin_fraction)
    
    def local_search(placements, bins, time_limit):
        """Improve packing via local search."""
        best_placements = [p[:] for p in placements]
        best_bins = [[list(p) for p in b] for b in bins]
        best_score = compute_score(best_placements)
        
        def try_move(box_idx, target_bin):
            """Try to move a box to a different bin."""
            b, x, y, z, w, h, d = best_placements[box_idx]
            if target_bin == b:
                return False
            
            # Remove from current bin
            current_bin = best_bins[b]
            to_remove = (x, y, z, w, h, d)
            if to_remove not in current_bin:
                return False
            current_bin.remove(to_remove)
            
            # Try to place in target bin
            target = best_bins[target_bin]
            pos = find_best_position(target, w, h, d)
            if pos is not None:
                tx, ty, tz = pos
                target.append((tx, ty, tz, w, h, d))
                best_placements[box_idx] = (target_bin, tx, ty, tz, w, h, d)
                return True
            else:
                # Revert
                current_bin.append((x, y, z, w, h, d))
                best_placements[box_idx] = (b, x, y, z, w, h, d)
                return False
        
        def try_swap(box_i, box_j):
            """Try to swap two boxes between bins."""
            bi, xi, yi, zi, wi, hi, di = best_placements[box_i]
            bj, xj, yj, zj, wj, hj, dj = best_placements[box_j]
            
            if bi == bj:
                return False
            
            # Remove both
            bin_i = best_bins[bi]
            bin_j = best_bins[bj]
            
            item_i = (xi, yi, zi, wi, hi, di)
            item_j = (xj, yj, zj, wj, hj, dj)
            
            if item_i not in bin_i or item_j not in bin_j:
                return False
            
            bin_i.remove(item_i)
            bin_j.remove(item_j)
            
            # Try to place box_i in bin_j
            pos_i = find_best_position(bin_j, wi, hi, di)
            if pos_i is None:
                # Revert
                bin_i.append(item_i)
                bin_j.append(item_j)
                return False
            
            # Try to place box_j in bin_i
            pos_j = find_best_position(bin_i, wj, hj, dj)
            if pos_j is None:
                # Revert
                bin_i.append(item_i)
                bin_j.append(item_j)
                return False
            
            # Apply swap
            xi2, yi2, zi2 = pos_i
            xj2, yj2, zj2 = pos_j
            bin_j.append((xi2, yi2, zi2, wi, hi, di))
            bin_i.append((xj2, yj2, zj2, wj, hj, dj))
            best_placements[box_i] = (bj, xi2, yi2, zi2, wi, hi, di)
            best_placements[box_j] = (bi, xj2, yj2, zj2, wj, hj, dj)
            return True
        
        # Local search iterations
        max_iters = 200
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            improved = False
            
            # Try moves
            for i in range(n):
                if time.time() > time_limit:
                    break
                orig_bin = best_placements[i][0]
                for target in range(len(best_bins)):
                    if target == orig_bin:
                        continue
                    if try_move(i, target):
                        new_score = compute_score(best_placements)
                        if new_score < best_score - 1e-9:
                            best_score = new_score
                            improved = True
                            break
                        else:
                            # Revert
                            try_move(i, orig_bin)
                if improved:
                    break
            
            # Try swaps if no move improved
            if not improved:
                for i in range(n):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, n):
                        if time.time() > time_limit:
                            break
                        if try_swap(i, j):
                            new_score = compute_score(best_placements)
                            if new_score < best_score - 1e-9:
                                best_score = new_score
                                improved = True
                                break
                            else:
                                # Revert swap
                                try_swap(i, j)
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved and it > 50:
                break
        
        return best_placements, best_score
    
    # Try multiple strategies and keep the best
    strategies = [
        'volume_desc',
        'max_dim_desc',
        'surface_area_desc',
        'volume_then_max',
    ]
    
    best_overall = None
    best_overall_score = float('inf')
    
    # First pass: deterministic strategies
    for strat in strategies:
        if time.time() > deadline - 20:
            break
        
        placements, bins = pack_with_strategy(strat)
        score = compute_score(placements)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall = (placements, bins)
    
    # Local search on the best
    if best_overall is not None and time.time() < deadline - 10:
        placements, bins = best_overall
        ls_placements, ls_score = local_search(placements, bins, deadline - 5)
        if ls_score < best_overall_score:
            best_overall = (ls_placements, None)
            best_overall_score = ls_score
    
    # If we have time, try random restarts
    if time.time() < deadline - 30:
        for restart in range(10):
            if time.time() > deadline - 15:
                break
            
            seed = restart * 12345 + 67890
            placements, bins = pack_with_strategy('random', random_seed=seed)
            score = compute_score(placements)
            
            if score < best_overall_score:
                best_overall_score = score
                best_overall = (placements, bins)
                
                # Local search on this
                if time.time() < deadline - 10:
                    ls_placements, ls_score = local_search(placements, bins, deadline - 5)
                    if ls_score < best_overall_score:
                        best_overall = (ls_placements, None)
                        best_overall_score = ls_score
    
    # Return the best placements
    if best_overall is None:
        # Fallback: simple shelf packing
        result = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
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
        return result
    
    return best_overall[0]
# EVOLVE-BLOCK-END
