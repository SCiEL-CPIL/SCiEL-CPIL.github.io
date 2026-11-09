import time
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
import time
import math
import random
from collections import defaultdict

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placements = [None] * n
    bins = []  # List of lists, each containing (x, y, z, w, h, d, original_index)
    current_bin = []
    
    def check_overlap(x, y, z, w, h, d, bin_boxes):
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return True
        return False
    
    def find_best_position(w, h, d, bin_boxes):
        # Generate candidate positions using corners of existing boxes
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
            
        # Filter candidates to those within bounds
        x_candidates = [x for x in x_candidates if x + w <= W]
        y_candidates = [y for y in y_candidates if y + h <= H]
        z_candidates = [z for z in z_candidates if z + d <= D]
        
        best_pos = None
        best_cost = float('inf')
        
        # Use a more sophisticated cost function:
        # 1. Primary: minimize max coordinate (encourages filling from bottom-left-front)
        # 2. Secondary: minimize sum of coordinates
        # 3. Tertiary: prefer positions that create less fragmentation
        
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if check_overlap(x, y, z, w, h, d, bin_boxes):
                        continue
                    # Cost: weighted sum with emphasis on compactness
                    # Use max as primary to encourage filling the "corner"
                    max_coord = max(x, y, z)
                    sum_coord = x + y + z
                    # Combine: prioritize lower max, then lower sum
                    cost = max_coord * 1000 + sum_coord
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        
        return best_pos
    
    def try_place_in_bin(w, h, d, bin_boxes):
        """Try to place a box in the given bin. Returns position or None."""
        pos = find_best_position(w, h, d, bin_boxes)
        if pos is not None:
            return pos
        return None
    
    # First pass: greedy placement
    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback
            break
            
        placed = False
        # Try current bin first
        pos = try_place_in_bin(w, h, d, current_bin)
        if pos is not None:
            x, y, z = pos
            current_bin.append((x, y, z, w, h, d, i))
            placements[i] = (len(bins), x, y, z, w, h, d)
            placed = True
        else:
            # Try other existing bins
            found = False
            for bin_idx in range(len(bins)):
                pos = try_place_in_bin(w, h, d, bins[bin_idx])
                if pos is not None:
                    x, y, z = pos
                    bins[bin_idx].append((x, y, z, w, h, d, i))
                    placements[i] = (bin_idx, x, y, z, w, h, d)
                    found = True
                    break
            if not found:
                # Open a new bin
                if current_bin:
                    bins.append(current_bin)
                current_bin = [(0, 0, 0, w, h, d, i)]
                placements[i] = (len(bins), 0, 0, 0, w, h, d)
                placed = True
    
    if current_bin:
        bins.append(current_bin)
    
    # Check if all boxes are placed
    if any(p is None for p in placements):
        # Fallback: place remaining boxes in new bins
        bin_count = len(bins)
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                # Find a position in a new bin
                bins.append([(0, 0, 0, w, h, d, i)])
                placements[i] = (bin_count, 0, 0, 0, w, h, d)
                bin_count += 1
    
    # Second pass: try to improve by moving boxes to reduce bin count
    # This is a local search step
    if time.time() < deadline - 10:
        # Try to consolidate bins by moving boxes
        max_improvements = 50
        improvements = 0
        for _ in range(max_improvements):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try moving each box to an earlier bin if possible
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                w, h, d = boxes[i]
                current_bin_idx = placements[i][0]
                if current_bin_idx == 0:
                    continue
                # Remove from current bin
                current_bin_list = bins[current_bin_idx]
                box_entry = None
                for j, entry in enumerate(current_bin_list):
                    if entry[6] == i:
                        box_entry = entry
                        break
                if box_entry is None:
                    continue
                current_bin_list.remove(box_entry)
                
                # Try to place in an earlier bin
                placed_elsewhere = False
                for target_bin_idx in range(current_bin_idx):
                    pos = try_place_in_bin(w, h, d, bins[target_bin_idx])
                    if pos is not None:
                        x, y, z = pos
                        bins[target_bin_idx].append((x, y, z, w, h, d, i))
                        placements[i] = (target_bin_idx, x, y, z, w, h, d)
                        placed_elsewhere = True
                        improved = True
                        improvements += 1
                        break
                
                if not placed_elsewhere:
                    # Put back in original bin
                    current_bin_list.append(box_entry)
            
            if not improved:
                break
    
    # Remove empty bins and renumber
    final_bins = [b for b in bins if len(b) > 0]
    bin_mapping = {}
    for new_idx, old_bin in enumerate(final_bins):
        for entry in old_bin:
            old_idx = entry[6]
            # Find which old bin index this was
            for old_idx_check in range(len(bins)):
                if bins[old_idx_check] is old_bin:
                    bin_mapping[old_idx_check] = new_idx
                    break
    
    # Rebuild placements with new bin indices
    final_placements = [None] * n
    for i in range(n):
        old_bin_idx = placements[i][0]
        new_bin_idx = bin_mapping.get(old_bin_idx, old_bin_idx)
        x, y, z, w, h, d = placements[i][1], placements[i][2], placements[i][3], placements[i][4], placements[i][5], placements[i][6]
        final_placements[i] = (new_bin_idx, x, y, z, w, h, d)
    
    return final_placements
# EVOLVE-BLOCK-END
