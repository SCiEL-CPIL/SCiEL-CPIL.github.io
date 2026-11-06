"""3D Container Loading: multi-strategy with improved heuristics and local search."""
import time
import math
import random
import numpy as np
from itertools import product

def load(boxes, container):
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin vs 150s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    
    # Precompute orientations for each box
    box_data = []
    for idx, (w, h, d) in enumerate(boxes):
        # Generate all unique orientations
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        
        # Filter to only orientations that fit in container
        valid_orients = [(bw, bh, bd) for bw, bh, bd in orients if bw <= W and bh <= H and bd <= D]
        if not valid_orients:
            valid_orients = [(w, h, d)]
        
        # Sort orientations: prefer those with smaller max dimension (more "flat")
        valid_orients.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        
        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))
    
    # Sort boxes by various criteria
    box_data_vol = sorted(box_data, key=lambda x: x[2], reverse=True)
    box_data_maxdim = sorted(box_data, key=lambda x: max(x[3], x[4], x[5]), reverse=True)
    box_data_ratio = sorted(box_data, key=lambda x: max(x[3], x[4], x[5]) / max(1, min(x[3], x[4], x[5])), reverse=True)
    
    def try_place_greedy(boxes_to_place, W, H, D, deadline, candidate_order='zyx', use_skyline=True):
        """Try to place boxes using a candidate position approach with occupancy grid."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(boxes_to_place)
        
        total_placed_volume = 0
        
        # Use a 3D numpy array for fast overlap checking
        occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.uint8)
        
        for idx, orients, volume, ow, oh, od in boxes_to_place:
            if time.time() > deadline:
                # Time out - mark remaining as not placed
                for j in range(idx, len(boxes_to_place)):
                    if results[j] is None:
                        oj, oorient, ov, ow2, oh2, od2 = boxes_to_place[j]
                        results[oj] = (-1, 0, 0, 0, ow2, oh2, od2)
                # Fill any that are still None
                for i in range(len(boxes_to_place)):
                    if results[i] is None:
                        w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                        results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
                break
            
            # Get candidate positions from placed boxes and origin
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                cx_set.add(px)
                cx_set.add(px + pw)
                cy_set.add(py)
                cy_set.add(py + ph)
                cz_set.add(pz)
                cz_set.add(pz + pd)
            cx_list = sorted(cx_set)
            cy_list = sorted(cy_set)
            cz_list = sorted(cz_set)
            
            found = False
            # Generate candidates and sort by priority
            candidates = []
            if candidate_order == 'zyx':
                for cz in cz_list:
                    for cy in cy_list:
                        for cx in cx_list:
                            candidates.append((cx, cy, cz))
                candidates.sort(key=lambda c: (c[2], c[1], c[0]))  # z, y, x
            elif candidate_order == 'xyz':
                for cx in cx_list:
                    for cy in cy_list:
                        for cz in cz_list:
                            candidates.append((cx, cy, cz))
                candidates.sort(key=lambda c: (c[0], c[1], c[2]))  # x, y, z
            elif candidate_order == 'yxz':
                for cy in cy_list:
                    for yx in cx_list:
                        for cz in cz_list:
                            candidates.append((yx, cy, cz))
                candidates.sort(key=lambda c: (c[1], c[0], c[2]))  # y, x, z
            elif candidate_order == 'zxy':
                for cz in cz_list:
                    for cx in cx_list:
                        for cy in cy_list:
                            candidates.append((cx, cy, cz))
                candidates.sort(key=lambda c: (c[2], c[0], c[1]))  # z, x, y
            elif candidate_order == 'random':
                all_cands = list(product(cx_list, cy_list, cz_list))
                random.shuffle(all_cands)
                candidates = all_cands
            else:
                # default zyx
                for cz in cz_list:
                    for cy in cy_list:
                        for cx in cx_list:
                            candidates.append((cx, cy, cz))
                candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            
            for (cx, cy, cz) in candidates:
                # Try each orientation
                for (bw, bh, bd) in orients:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Check overlap using occupancy grid - fast vectorized check
                    region = occ[cx:cx+bw, cy:cy+bh, cz:cz+bd]
                    if np.any(region > 0):
                        continue
                    
                    # Place this box - update occupancy grid
                    occ[cx:cx+bw, cy:cy+bh, cz:cz+bd] = 1
                    placed.append((cx, cy, cz, bw, bh, bd))
                    results[idx] = (0, cx, cy, cz, bw, bh, bd)
                    total_placed_volume += volume
                    found = True
                    break
                
                if found:
                    break
            
            if not found:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        # Fill any remaining None entries
        for i in range(len(boxes_to_place)):
            if results[i] is None:
                w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
        
        return results, total_placed_volume
    
    def try_place_skyline(boxes_to_place, W, H, D, deadline):
        """Skyline-based placement: maintain a skyline and place boxes at the highest available position."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(boxes_to_place)
        
        total_placed_volume = 0
        
        # Use a 3D numpy array for fast overlap checking
        occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.uint8)
        
        for idx, orients, volume, ow, oh, od in boxes_to_place:
            if time.time() > deadline:
                # Time out - mark remaining as not placed
                for j in range(idx, len(boxes_to_place)):
                    if results[j] is None:
                        oj, oorient, ov, ow2, oh2, od2 = boxes_to_place[j]
                        results[oj] = (-1, 0, 0, 0, ow2, oh2, od2)
                # Fill any that are still None
                for i in range(len(boxes_to_place)):
                    if results[i] is None:
                        w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                        results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
                break
            
            # Generate candidate positions from placed boxes
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                cx_set.add(px)
                cx_set.add(px + pw)
                cy_set.add(py)
                cy_set.add(py + ph)
                cz_set.add(pz)
                cz_set.add(pz + pd)
            cx_list = sorted(cx_set)
            cy_list = sorted(cy_set)
            cz_list = sorted(cz_set)
            
            found = False
            best_pos = None
            best_score = -1
            
            # For each candidate position, try all orientations and score by how "tight" the fit is
            for cx in cx_list:
                for cy in cy_list:
                    for cz in cz_list:
                        for (bw, bh, bd) in orients:
                            # Check bounds
                            if cx + bw > W or cy + bh > H or cz + bd > D:
                                continue
                            
                            # Check overlap using occupancy grid
                            region = occ[cx:cx+bw, cy:cy+bh, cz:cz+bd]
                            if np.any(region > 0):
                                continue
                            
                            # Score: prefer positions that are lower (smaller z), then smaller y, then smaller x
                            # Also prefer orientations that are more compact
                            score = -(cz * 1000 + cy * 100 + cx) - (bw + bh + bd) * 0.1
                            if score > best_score:
                                best_score = score
                                best_pos = (cx, cy, cz, bw, bh, bd)
                        
                        # Early exit if we found a position at origin
                        if best_pos and best_pos[0] == 0 and best_pos[1] == 0 and best_pos[2] == 0:
                            break
                    if best_pos and best_pos[0] == 0 and best_pos[1] == 0 and best_pos[2] == 0:
                        break
                if best_pos and best_pos[0] == 0 and best_pos[1] == 0 and best_pos[2] == 0:
                    break
            
            if best_pos:
                cx, cy, cz, bw, bh, bd = best_pos
                occ[cx:cx+bw, cy:cy+bh, cz:cz+bd] = 1
                placed.append((cx, cy, cz, bw, bh, bd))
                results[idx] = (0, cx, cy, cz, bw, bh, bd)
                total_placed_volume += volume
                found = True
            
            if not found:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        # Fill any remaining None entries
        for i in range(len(boxes_to_place)):
            if results[i] is None:
                w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
        
        return results, total_placed_volume
    
    def local_search(results, boxes_to_place, W, H, D, deadline, max_iters=500):
        """Try to improve the solution by swapping or rotating placed boxes."""
        placed_boxes = {}  # idx -> (x, y, z, bw, bh, bd)
        unplaced = []
        
        for i, res in enumerate(results):
            idx, x, y, z, bw, bh, bd = res
            if idx == 0:
                placed_boxes[i] = (x, y, z, bw, bh, bd)
            else:
                unplaced.append(i)
        
        if not unplaced:
            return results, sum(boxes_to_place[i][2] for i in range(len(boxes_to_place)))
        
        # Rebuild occupancy from placed boxes
        occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.uint8)
        for i, (x, y, z, bw, bh, bd) in placed_boxes.items():
            occ[x:x+bw, y:y+bh, z:z+bd] = 1
        
        best_results = results[:]
        best_volume = sum(boxes_to_place[i][2] for i in placed_boxes.keys())
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to place one of the unplaced boxes
            if not unplaced:
                break
            
            ui = unplaced[0]
            idx, orients, volume, ow, oh, od = boxes_to_place[ui]
            
            # Generate candidate positions
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (x, y, z, bw, bh, bd) in placed_boxes.values():
                cx_set.add(x)
                cx_set.add(x + bw)
                cy_set.add(y)
                cy_set.add(y + bh)
                cz_set.add(z)
                cz_set.add(z + bd)
            cx_list = sorted(cx_set)
            cy_list = sorted(cy_set)
            cz_list = sorted(cz_set)
            
            placed_this = False
            for cx in cx_list:
                for cy in cy_list:
                    for cz in cz_list:
                        for (bw, bh, bd) in orients:
                            if cx + bw > W or cy + bh > H or cz + bd > D:
                                continue
                            
                            region = occ[cx:cx+bw, cy:cy+bh, cz:cz+bd]
                            if np.any(region > 0):
                                continue
                            
                            # Place it
                            occ[cx:cx+bw, cy:cy+bh, cz:cz+bd] = 1
                            placed_boxes[ui] = (cx, cy, cz, bw, bh, bd)
                            unplaced.pop(0)
                            placed_this = True
                            
                            # Update results
                            best_results[ui] = (0, cx, cy, cz, bw, bh, bd)
                            best_volume += volume
                            break
                        if placed_this:
                            break
                    if placed_this:
                        break
                if placed_this:
                    break
            
            if not placed_this:
                # Try to remove a placed box to make room
                if len(placed_boxes) > 1:
                    # Remove a small box to see if we can place the unplaced one
                    removed = None
                    for ri in list(placed_boxes.keys()):
                        rv = boxes_to_place[ri][2]
                        if rv < volume:  # only remove smaller boxes
                            x, y, z, bw, bh, bd = placed_boxes[ri]
                            occ[x:x+bw, y:y+bh, z:z+bd] = 0
                            removed = (ri, (x, y, z, bw, bh, bd))
                            break
                    
                    if removed:
                        ri, rpos = removed
                        del placed_boxes[ri]
                        best_results[ri] = (-1, 0, 0, 0, boxes_to_place[ri][3], boxes_to_place[ri][4], boxes_to_place[ri][5])
                        best_volume -= boxes_to_place[ri][2]
                        unplaced.append(ri)
        
        return best_results, best_volume
    
    # Run multiple strategies
    all_results = []
    
    # Strategy 1: Volume-descending with ZYX candidate ordering
    results1, vol1 = try_place_greedy(box_data_vol, W, H, D, deadline, 'zyx')
    all_results.append((vol1, results1))
    
    # Strategy 2: Max-dimension-descending with ZYX
    results2, vol2 = try_place_greedy(box_data_maxdim, W, H, D, deadline, 'zyx')
    all_results.append((vol2, results2))
    
    # Strategy 3: Volume-descending with XYZ candidate ordering
    results3, vol3 = try_place_greedy(box_data_vol, W, H, D, deadline, 'xyz')
    all_results.append((vol3, results3))
    
    # Strategy 4: Volume-descending with YXZ candidate ordering
    results4, vol4 = try_place_greedy(box_data_vol, W, H, D, deadline, 'yxz')
    all_results.append((vol4, results4))
    
    # Strategy 5: Volume-descending with ZXY candidate ordering
    results5, vol5 = try_place_greedy(box_data_vol, W, H, D, deadline, 'zxy')
    all_results.append((vol5, results5))
    
    # Strategy 6: Ratio-descending with ZYX
    results6, vol6 = try_place_greedy(box_data_ratio, W, H, D, deadline, 'zyx')
    all_results.append((vol6, results6))
    
    # Strategy 7: Skyline with volume-descending
    if time.time() < deadline - 10:
        results7, vol7 = try_place_skyline(box_data_vol, W, H, D, deadline)
        all_results.append((vol7, results7))
    
    # Choose the best result based on placed volume
    best_results = max(all_results, key=lambda x: x[0])
    
    # Try local search on the best result if we have time
    if time.time() < deadline - 20:
        best_results_ls, best_vol_ls = local_search(best_results[1], box_data, W, H, D, deadline, max_iters=200)
        if best_vol_ls > best_results[0]:
            best_results = (best_vol_ls, best_results_ls)
    
    return best_results[1]
