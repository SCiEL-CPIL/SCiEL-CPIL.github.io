import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Data structures for bins
    # Each bin is a list of placed boxes: (x, y, z, w, h, d, original_index)
    bins = []
    # placements[i] = (bin_idx, x, y, z, w, h, d)
    placements = [None] * n
    
    def check_overlap(x, y, z, w, h, d, bin_boxes):
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return True
        return False

    def find_best_position(w, h, d, bin_boxes):
        """
        Find the best position in a bin.
        'Best' is defined by a cost function that prefers tight packing.
        """
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            x_candidates.add(bx)
            x_candidates.add(bx + bw)
            y_candidates.add(by)
            y_candidates.add(by + bh)
            z_candidates.add(bz)
            z_candidates.add(bz + bd)
            
        x_candidates = [x for x in x_candidates if x + w <= W]
        y_candidates = [y for y in y_candidates if y + h <= H]
        z_candidates = [z for z in z_candidates if z + d <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None, float('inf')

        best_pos = None
        best_cost = float('inf')
        
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if check_overlap(x, y, z, w, h, d, bin_boxes):
                        continue
                    
                    # Calculate cost
                    cost = x + y + z
                    touches = False
                    for bx, by, bz, bw, bh, bd, _ in bin_boxes:
                        if (x + w == bx or bx + bw == x or 
                            y + h == by or by + bh == y or 
                            z + d == bz or bz + bd == z):
                            touches = True
                            break
                    if touches:
                        cost -= 50
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        
        return best_pos, best_cost

    def try_place_in_bin(w, h, d, bin_boxes):
        pos, cost = find_best_position(w, h, d, bin_boxes)
        if pos is not None:
            return pos, cost
        return None, float('inf')

    def calculate_bin_volume(bin_boxes):
        vol = 0
        for _, _, _, w, h, d, _ in bin_boxes:
            vol += w * h * d
        return vol

    def score_solution(placements, bins):
        occupied_bins = set()
        for i in range(n):
            if placements[i] is not None:
                occupied_bins.add(placements[i][0])
        B = len(occupied_bins)
        
        if B == 0:
            return 0.0
        max_bin_idx = max(occupied_bins)
        last_vol = calculate_bin_volume(bins[max_bin_idx])
        container_vol = W * H * D
        fraction = last_vol / container_vol if container_vol > 0 else 1.0
        return B + 0.1 * (1 - fraction)

    def greedy_construct():
        """
        Phase 1: Greedy Best-Fit Decreasing construction.
        """
        bins = []
        placements = [None] * n
        
        for i, w, h, d in indexed_boxes:
            if time.time() > deadline:
                break
                
            best_bin_idx = -1
            best_pos = None
            best_cost = float('inf')
            
            for bin_idx in range(len(bins)):
                pos, cost = try_place_in_bin(w, h, d, bins[bin_idx])
                if pos is not None and cost < best_cost:
                    best_cost = cost
                    best_bin_idx = bin_idx
                    best_pos = pos
            
            if best_bin_idx != -1:
                x, y, z = best_pos
                bins[best_bin_idx].append((x, y, z, w, h, d, i))
                placements[i] = (best_bin_idx, x, y, z, w, h, d)
            else:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d, i)])
                placements[i] = (new_bin_idx, 0, 0, 0, w, h, d)
        
        # Ensure all boxes are placed (safety)
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d, i)])
                placements[i] = (new_bin_idx, 0, 0, 0, w, h, d)
        
        return placements, bins

    def local_search(placements, bins):
        """
        Phase 2: Simulated Annealing local search.
        Moves: Relocate a box from one bin to another.
        Objective: Minimize B + 0.1 * (1 - last_bin_vol_fraction).
        """
        current_score = score_solution(placements, bins)
        best_score = current_score
        best_placements = [p[:] for p in placements]
        best_bins = [list(b) for b in bins]
        
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        # Precompute container volume for faster scoring
        container_vol = W * H * D
        
        def fast_score(placements, bins):
            occupied_bins = set()
            for i in range(n):
                if placements[i] is not None:
                    occupied_bins.add(placements[i][0])
            B = len(occupied_bins)
            if B == 0:
                return 0.0
            max_bin_idx = max(occupied_bins)
            last_vol = calculate_bin_volume(bins[max_bin_idx])
            fraction = last_vol / container_vol if container_vol > 0 else 1.0
            return B + 0.1 * (1 - fraction)
        
        while T > min_T and time.time() < deadline - 5:
            # Pick a random box from a random occupied bin
            occupied_indices = [b for b in range(len(bins)) if bins[b]]
            if not occupied_indices:
                break
            
            src_bin_idx = random.choice(occupied_indices)
            box_pos_in_bin = random.randint(0, len(bins[src_bin_idx]) - 1)
            box_entry = bins[src_bin_idx][box_pos_in_bin]
            orig_idx = box_entry[6]
            w, h, d = box_entry[3], box_entry[4], box_entry[5]
            
            # Remove from current bin
            bins[src_bin_idx].remove(box_entry)
            placements[orig_idx] = None
            
            # Try to place in a random other bin
            target_candidates = [b for b in range(len(bins)) if b != src_bin_idx and bins[b]]
            if not target_candidates:
                # Put back
                bins[src_bin_idx].append(box_entry)
                placements[orig_idx] = (src_bin_idx, box_entry[0], box_entry[1], box_entry[2], w, h, d)
                T *= cooling
                continue
                
            target_bin_idx = random.choice(target_candidates)
            pos, cost = try_place_in_bin(w, h, d, bins[target_bin_idx])
            
            if pos is not None:
                x, y, z = pos
                bins[target_bin_idx].append((x, y, z, w, h, d, orig_idx))
                placements[orig_idx] = (target_bin_idx, x, y, z, w, h, d)
                
                new_score = fast_score(placements, bins)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_placements = [p[:] for p in placements]
                        best_bins = [list(b) for b in bins]
                else:
                    # Revert
                    bins[target_bin_idx].remove((x, y, z, w, h, d, orig_idx))
                    placements[orig_idx] = None
                    bins[src_bin_idx].append(box_entry)
                    placements[orig_idx] = (src_bin_idx, box_entry[0], box_entry[1], box_entry[2], w, h, d)
            
            else:
                # Couldn't place, put back
                bins[src_bin_idx].append(box_entry)
                placements[orig_idx] = (src_bin_idx, box_entry[0], box_entry[1], box_entry[2], w, h, d)
            
            T *= cooling
        
        return best_placements, best_bins, best_score

    # --- Main Loop: Multi-Restart Simulated Annealing ---
    best_overall_score = float('inf')
    best_overall_placements = None
    best_overall_bins = None
    
    # Run multiple restarts
    while time.time() < deadline - 10:
        # Greedy construction
        placements, bins = greedy_construct()
        
        # Local search (Simulated Annealing)
        placements, bins, score = local_search(placements, bins)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_placements = [p[:] for p in placements]
            best_overall_bins = [list(b) for b in bins]
            
        # If we've achieved a very good score (e.g., B=1), we can break early
        if best_overall_score < 1.1:
            break

    # Final cleanup: Renumber bins to be 0..B-1
    if best_overall_placements is None:
        # Fallback if no valid solution found (should not happen)
        placements, bins = greedy_construct()
        best_overall_placements = placements
        best_overall_bins = bins

    placements = best_overall_placements
    bins = best_overall_bins

    occupied_old_indices = sorted([p[0] for p in placements if p is not None])
    
    if not occupied_old_indices:
        return []
        
    old_to_new = {}
    for new_idx, old_idx in enumerate(occupied_old_indices):
        old_to_new[old_idx] = new_idx
        
    final_bins = [[] for _ in range(len(occupied_old_indices))]
    new_placements = [None] * n
    
    for i in range(n):
        if placements[i] is None:
            continue
        old_bin_idx = placements[i][0]
        new_bin_idx = old_to_new[old_bin_idx]
        x, y, z, w, h, d = placements[i][1], placements[i][2], placements[i][3], placements[i][4], placements[i][5], placements[i][6]
        
        final_bins[new_bin_idx].append((x, y, z, w, h, d, i))
        new_placements[i] = (new_bin_idx, x, y, z, w, h, d)
        
    return new_placements
# EVOLVE-BLOCK-END
