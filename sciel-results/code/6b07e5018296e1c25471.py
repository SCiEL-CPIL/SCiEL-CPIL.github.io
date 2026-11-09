import time
import math
import random
import numpy as np
from collections import defaultdict
import itertools

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    if container_vol == 0:
        return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    # Sort indices by volume descending (Best Fit Decreasing)
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    
    # Generate a few different orderings for diversification
    orderings = [vol_sorted]
    # Add a few random perturbations to the sorted list
    for _ in range(5):
        perturbed = vol_sorted[:]
        # Shuffle a small portion to break ties or slightly reorder
        for _ in range(n // 2):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        orderings.append(perturbed)

    def compute_score(placements):
        if len(placements) != n:
            return float('inf')
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        if B == 0:
            return 0
        max_bin = max(p[0] for p in placements)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == max_bin)
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - occupied_frac)

    def renumber_bins(placements):
        """Compress bin labels to 0..B-1 in order of first use.
        The new 'last bin' (label B-1) is the bin whose first box appears
        latest in the placement list. This can improve the 0.1*(1-fill) term
        if that bin is fuller than the old max-label bin."""
        if len(placements) == 0:
            return placements
        mapping = {}
        next_label = 0
        new_placements = []
        for p in placements:
            old = p[0]
            if old not in mapping:
                mapping[old] = next_label
                next_label += 1
            new_placements.append((mapping[old], p[1], p[2], p[3], p[4], p[5], p[6]))
        return new_placements

    def validate_placement(placements):
        if len(placements) != n:
            return False
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            bins[b].append((x, y, z, w, h, d))
        for bin_idx, box_list in bins.items():
            for i in range(len(box_list)):
                for j in range(i + 1, len(box_list)):
                    x1, y1, z1, w1, h1, d1 = box_list[i]
                    x2, y2, z2, w2, h2, d2 = box_list[j]
                    if (x1 < x2 + w2 and x2 < x1 + w1 and
                        y1 < y2 + h2 and y2 < y1 + h1 and
                        z1 < z2 + d2 and z2 < z1 + d1):
                        return False
        return True

    class Bin:
        def __init__(self):
            self.boxes = [] # (x, y, z, w, h, d)
            self.eps = [(0, 0, 0)] # Extreme Points
            self.volume = 0
            # Grid-based occupancy for fast overlap checks
            # Using a slightly larger grid to handle edge cases if needed, but standard is enough
            # We cap the grid size to avoid memory issues with very large containers
            if W * H * D <= 1000000:
                self.grid = np.zeros((W, H, D), dtype=bool)
            else:
                self.grid = None

        def is_dominated(self, p1, p2):
            return (p1[0] <= p2[0] and p1[1] <= p2[1] and p1[2] <= p2[2] and 
                    (p1[0] < p2[0] or p1[1] < p2[1] or p1[2] < p2[2]))

        def add_box(self, x, y, z, w, h, d):
            # Check bounds
            if x + w > W or y + h > H or z + d > D:
                return False
            
            # Check overlap
            if self.grid is not None:
                if self.grid[x:x+w, y:y+h, z:z+d].any():
                    return False
            else:
                # Fallback to O(n) check
                for bx, by, bz, bw, bh, bd in self.boxes:
                    if (x < bx + bw and bx < x + w and
                        y < by + bh and by < y + h and
                        z < bz + bd and bz < z + d):
                        return False
            
            # Place box
            self.boxes.append((x, y, z, w, h, d))
            self.volume += w * h * d
            if self.grid is not None:
                self.grid[x:x+w, y:y+h, z:z+d] = True
            
            # Update Extreme Points
            # Remove EPs inside the new box
            filtered_eps = []
            for p in self.eps:
                if not (x <= p[0] < x + w and y <= p[1] < y + h and z <= p[2] < z + d):
                    filtered_eps.append(p)
            
            # Add new candidates (all corners of the new box)
            candidates = [
                (x + w, y, z),
                (x, y + h, z),
                (x, y, z + d),
                (x + w, y + h, z),
                (x + w, y, z + d),
                (x, y + h, z + d),
                (x + w, y + h, z + d)
            ]
            
            for c in candidates:
                if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D:
                    filtered_eps.append(c)
            
            # Prune dominated points
            # For efficiency, we can use a set-like logic or just O(N^2) since N is small
            final_eps = []
            for p in filtered_eps:
                is_dom = False
                for q in filtered_eps:
                    if p == q:
                        continue
                    if self.is_dominated(q, p):
                        is_dom = True
                        break
                if not is_dom:
                    final_eps.append(p)
            
            self.eps = final_eps
            return True

        def try_place(self, w, h, d):
            """Try to place a box in this bin using EPs. Returns (x,y,z) or None."""
            if not self.eps:
                return None
            
            # Sort EPs lexicographically to prefer "tight" packing towards origin
            sorted_eps = sorted(self.eps)
            
            for x, y, z in sorted_eps:
                if self.is_valid_pos(x, y, z, w, h, d):
                    return (x, y, z)
            
            return None

        def is_valid_pos(self, x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            # Fast overlap check using grid
            if self.grid is not None:
                if self.grid[x:x+w, y:y+h, z:z+d].any():
                    return False
            else:
                # Fallback to O(n) check
                for bx, by, bz, bw, bh, bd in self.boxes:
                    if (x < bx + bw and bx < x + w and
                        y < by + bh and by < y + h and
                        z < bz + bd and bz < z + d):
                        return False
            return True

    def greedy_pack(ordering):
        bins = []
        placements = [None] * n
        
        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False
            
            best_bin_idx = -1
            best_pos = None
            best_score = float('inf')
            
            for i, bin_obj in enumerate(bins):
                pos = bin_obj.try_place(w, h, d)
                if pos is not None:
                    # Best Fit: select bin with max current volume (fullest)
                    # We could also check if this is the "tightest" fit, but volume is a good proxy
                    if best_bin_idx == -1 or bins[i].volume > bins[best_bin_idx].volume:
                        best_bin_idx = i
                        best_pos = pos
            
            if best_bin_idx != -1:
                x, y, z = best_pos
                bins[best_bin_idx].add_box(x, y, z, w, h, d)
                placements[box_idx] = (best_bin_idx, x, y, z, w, h, d)
            else:
                # Open new bin
                new_bin = Bin()
                new_bin.add_box(0, 0, 0, w, h, d)
                bins.append(new_bin)
                placements[box_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
        
        return placements

    def local_search(placements):
        """
        Try to move boxes from the last bin to previous bins to reduce B or fill last bin.
        """
        if time.time() > deadline - 10:
            return placements
            
        bins_dict = defaultdict(list)
        for i, p in enumerate(placements):
            bins_dict[p[0]].append(i)
            
        bin_indices = sorted(bins_dict.keys())
        if len(bin_indices) <= 1:
            return placements
            
        last_bin_idx = bin_indices[-1]
        boxes_to_move = bins_dict[last_bin_idx][:]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Iterate over a copy of the list to handle modifications
            for box_idx in boxes_to_move[:]:
                if time.time() > deadline - 5:
                    break
                if box_idx not in bins_dict.get(last_bin_idx, []):
                    continue # Already moved or removed
                
                p = placements[box_idx]
                w, h, d = p[4], p[5], p[6]
                
                moved = False
                # Try to move to other bins
                # Prefer bins with higher volume? Or just any bin that fits?
                # To minimize B, we just need to fit it somewhere else.
                # To maximize last bin fill, we might want to keep some boxes in last bin?
                # No, score is B + 0.1*(1 - fill). Reducing B by 1 is worth 10 bins of fill.
                # So primary goal: reduce B.
                
                for other_bin_idx in bin_indices[:-1]:
                    if other_bin_idx not in bins_dict or not bins_dict[other_bin_idx]:
                        continue
                    
                    other_box_indices = bins_dict[other_bin_idx]
                    other_boxes = [(placements[j][1], placements[j][2], placements[j][3], placements[j][4], placements[j][5], placements[j][6]) for j in other_box_indices]
                    
                    # Generate candidate positions using Extreme Points of the target bin
                    # We need to reconstruct or maintain EPs for local search bins.
                    # Since we don't have Bin objects here, we generate candidates from box faces.
                    x_cands = {0}
                    y_cands = {0}
                    z_cands = {0}
                    for bx, by, bz, bw, bh, bd in other_boxes:
                        x_cands.add(bx)
                        x_cands.add(bx + bw)
                        y_cands.add(by)
                        y_cands.add(by + bh)
                        z_cands.add(bz)
                        z_cands.add(bz + bd)
                    
                    # Sort candidates to prefer compact positions
                    x_cands = sorted(x_cands)
                    y_cands = sorted(y_cands)
                    z_cands = sorted(z_cands)
                    
                    # Limit candidates to avoid O(N^3) explosion
                    x_cands = x_cands[:20]
                    y_cands = y_cands[:20]
                    z_cands = z_cands[:20]
                    
                    for cx in x_cands:
                        if moved: break
                        for cy in y_cands:
                            if moved: break
                            for cz in z_cands:
                                if cx + w > W or cy + h > H or cz + d > D:
                                    continue
                                overlap = False
                                for bx, by, bz, bw, bh, bd in other_boxes:
                                    if (cx < bx + bw and bx < cx + w and
                                        cy < by + bh and by < cy + h and
                                        cz < bz + bd and bz < cz + d):
                                        overlap = True
                                        break
                                if not overlap:
                                    # Move box
                                    placements[box_idx] = (other_bin_idx, cx, cy, cz, w, h, d)
                                    bins_dict[last_bin_idx].remove(box_idx)
                                    if not bins_dict[last_bin_idx]:
                                        del bins_dict[last_bin_idx]
                                        bin_indices = sorted(bins_dict.keys())
                                        if len(bin_indices) <= 1:
                                            return placements
                                        last_bin_idx = bin_indices[-1]
                                        boxes_to_move = bins_dict[last_bin_idx][:]
                                    bins_dict[other_bin_idx].append(box_idx)
                                    moved = True
                                    improved = True
                                    break
                if moved:
                    if box_idx in boxes_to_move:
                        boxes_to_move.remove(box_idx)

        return placements

    best_result = None
    best_score = float('inf')

    for order in orderings:
        if time.time() > deadline - 20:
            break
        
        try:
            placements = greedy_pack(order)
            if not validate_placement(placements):
                continue
            
            # Renumber bins to compress labels
            placements = renumber_bins(placements)
            score = compute_score(placements)
            
            if time.time() < deadline - 10:
                improved_placements = local_search(placements)
                if validate_placement(improved_placements):
                    improved_placements = renumber_bins(improved_placements)
                    placements = improved_placements
                    score = compute_score(placements)
            
            if score < best_score:
                best_score = score
                best_result = placements[:]
                
        except Exception as e:
            continue

    if best_result is not None:
        return best_result

    return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]
