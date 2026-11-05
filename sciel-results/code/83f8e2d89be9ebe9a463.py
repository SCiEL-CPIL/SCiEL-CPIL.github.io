"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def compute_score(bins, bin_vols):
        """Compute the objective: B + 0.1*(1 - last_bin_vol/container_vol)"""
        if not bins:
            return float('inf')
        B = len(bins)
        last_bin_vol = bin_vols[-1]
        return B + 0.1 * (1 - last_bin_vol / container_vol)
    
    def can_place(placements, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) doesn't overlap with any placement."""
        for _, px, py, pz, pw, ph, pd in placements:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_positions(bin_placements, w, h, d):
        """Find all valid positions, sorted by (z, y, x). Returns list of (x,y,z)."""
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for _, px, py, pz, pw, ph, pd in bin_placements:
            if px + pw <= W:
                x_candidates.add(px + pw)
            if py + ph <= H:
                y_candidates.add(py + ph)
            if pz + pd <= D:
                z_candidates.add(pz + pd)
        
        x_candidates = sorted([x for x in x_candidates if x + w <= W])
        y_candidates = sorted([y for y in y_candidates if y + h <= H])
        z_candidates = sorted([z for z in z_candidates if z + d <= D])
        
        valid_positions = []
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if can_place(bin_placements, x, y, z, w, h, d):
                        valid_positions.append((x, y, z))
        
        # Sort by (z, y, x) for best-fit preference
        valid_positions.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid_positions
    
    def greedy_pack(boxes, container, heuristic='volume_desc', random_seed=None):
        """Greedy packing with different heuristics."""
        W, H, D = container
        n = len(boxes)
        container_vol = W * H * D
        
        if random_seed is not None:
            random.seed(random_seed)
        
        indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
        
        if heuristic == 'volume_desc':
            indexed_boxes.sort(key=lambda x: x[4], reverse=True)
        elif heuristic == 'max_dim_desc':
            indexed_boxes.sort(key=lambda x: max(x[1], x[2], x[3]), reverse=True)
        elif heuristic == 'surface_desc':
            indexed_boxes.sort(key=lambda x: 2*(x[1]*x[2] + x[1]*x[3] + x[2]*x[3]), reverse=True)
        elif heuristic == 'random':
            random.shuffle(indexed_boxes)
        else:
            indexed_boxes.sort(key=lambda x: x[4], reverse=True)
        
        bins = []
        bin_vols = []
        
        for orig_i, w, h, d, vol in indexed_boxes:
            if time.time() > deadline:
                break
            
            best_bin = -1
            best_pos = None
            best_score = None
            
            for b_idx, bin_placements in enumerate(bins):
                positions = find_positions(bin_placements, w, h, d)
                if positions:
                    # Best fit: prefer lowest z, then y, then x
                    pos = positions[0]
                    score = pos[2]  # z-coordinate
                    if best_score is None or score < best_score:
                        best_score = score
                        best_bin = b_idx
                        best_pos = pos
            
            if best_bin >= 0:
                x, y, z = best_pos
                bins[best_bin].append((orig_i, x, y, z, w, h, d))
                bin_vols[best_bin] += vol
            else:
                bins.append([(orig_i, 0, 0, 0, w, h, d)])
                bin_vols.append(vol)
        
        # Build result
        result = [None] * n
        for b_idx, bin_placements in enumerate(bins):
            for orig_i, x, y, z, w, h, d in bin_placements:
                result[orig_i] = (b_idx, x, y, z, w, h, d)
        
        # Fallback for any unplaced boxes
        next_bin = len(bins)
        for i in range(n):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (next_bin, 0, 0, 0, w, h, d)
                next_bin += 1
        
        return result, bins, bin_vols
    
    def local_search(result, bins, bin_vols, max_iterations=1000):
        """Try to move boxes from last bin to earlier bins to improve score."""
        W, H, D = container
        n = len(boxes)
        container_vol = W * H * D
        
        # Work with copies
        bins_copy = [list(b) for b in bins]
        bin_vols_copy = list(bin_vols)
        result_copy = [None] * n
        
        for b_idx, bin_placements in enumerate(bins_copy):
            for orig_i, x, y, z, w, h, d in bin_placements:
                result_copy[orig_i] = (b_idx, x, y, z, w, h, d)
        
        best_score = compute_score(bins_copy, bin_vols_copy)
        best_result = [tuple(r) for r in result_copy]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            if len(bins_copy) < 2:
                break
            
            last_bin_idx = len(bins_copy) - 1
            last_bin = bins_copy[last_bin_idx]
            
            if not last_bin:
                break
            
            moved_any = False
            for li in range(len(last_bin) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                
                orig_i, x, y, z, w, h, d = last_bin[li]
                vol = w * h * d
                
                best_target_bin = -1
                best_pos = None
                best_target_score = None
                
                for b_idx in range(last_bin_idx):
                    if time.time() > deadline - 5.0:
                        break
                    
                    positions = find_positions(bins_copy[b_idx], w, h, d)
                    if positions:
                        pos = positions[0]
                        new_last_vol = bin_vols_copy[last_bin_idx] - vol
                        
                        if new_last_vol == 0:
                            new_B = len(bins_copy) - 1
                            if new_B > 0:
                                new_last_vol = bin_vols_copy[new_B - 1]
                            else:
                                new_last_vol = 0
                        else:
                            new_B = len(bins_copy)
                        
                        new_score = new_B + 0.1 * (1 - new_last_vol / container_vol)
                        
                        if best_target_score is None or new_score < best_target_score:
                            best_target_score = new_score
                            best_target_bin = b_idx
                            best_pos = pos
                
                if best_target_bin >= 0:
                    last_bin.pop(li)
                    bin_vols_copy[last_bin_idx] -= vol
                    
                    if bin_vols_copy[last_bin_idx] == 0:
                        bins_copy.pop(last_bin_idx)
                        bin_vols_copy.pop(last_bin_idx)
                        last_bin_idx = len(bins_copy) - 1
                        if last_bin_idx < 0:
                            break
                    
                    x, y, z = best_pos
                    bins_copy[best_target_bin].append((orig_i, x, y, z, w, h, d))
                    bin_vols_copy[best_target_bin] += vol
                    
                    result_copy[orig_i] = (best_target_bin, x, y, z, w, h, d)
                    moved_any = True
            
            if not moved_any:
                break
            
            current_score = compute_score(bins_copy, bin_vols_copy)
            if current_score < best_score:
                best_score = current_score
                best_result = [tuple(r) for r in result_copy]
        
        return best_result, best_score
    
    # Phase 1: Try multiple heuristics and pick the best
    heuristics = ['volume_desc', 'max_dim_desc', 'surface_desc']
    best_overall_result = None
    best_overall_score = float('inf')
    
    for h in heuristics:
        if time.time() > deadline - 30.0:
            break
        
        result, bins, bin_vols = greedy_pack(boxes, container, heuristic=h)
        score = compute_score(bins, bin_vols)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_result = result
        
        # Apply local search to improve
        improved_result, improved_score = local_search(result, bins, bin_vols, max_iterations=500)
        
        if improved_score < best_overall_score:
            best_overall_score = improved_score
            best_overall_result = improved_result
    
    # Phase 2: Try random restarts with different orderings
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        result, bins, bin_vols = greedy_pack(boxes, container, heuristic='random', random_seed=r)
        score = compute_score(bins, bin_vols)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_result = result
        
        improved_result, improved_score = local_search(result, bins, bin_vols, max_iterations=300)
        
        if improved_score < best_overall_score:
            best_overall_score = improved_score
            best_overall_result = improved_result
    
    # Final fallback: ensure all boxes are placed
    if best_overall_result is None:
        best_overall_result = [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    return best_overall_result
# EVOLVE-BLOCK-END
