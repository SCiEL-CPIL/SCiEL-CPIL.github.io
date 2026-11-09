"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we aim to finish logic by 145s
    deadline = start_time + 145.0 

    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # Sort boxes by volume descending initially for stability
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    class Bin:
        def __init__(self):
            self.boxes = []  # list of (idx, x, y, z, w, h, d)
            self.volume = 0
            # Skyline points: corners of existing boxes
            # We maintain a set of candidate positions derived from box corners
            # Initially just (0,0,0)
            self.candidate_points = {(0, 0, 0)}

        def can_place(self, x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Check overlap with all existing boxes
            # Optimization: Sort boxes by x to break early? 
            # For n=40, O(N^2) per placement is fine (40*40=1600 ops, 40 boxes -> 64k ops per bin)
            for (_, bx, by, bz, bw, bh, bd) in self.boxes:
                if x < bx + bw and x + w > bx and \
                   y < by + bh and y + h > by and \
                   z < bz + bd and z + d > bz:
                    return False
            return True

        def get_valid_candidates(self, w, h, d):
            """
            Generate valid placement positions.
            Strategy: Check all existing corner points (skyline).
            If none work, try (0,0,0) just in case (though it should be in the set).
            """
            valid = []
            # Iterate over current candidate points
            # To make this faster, we could filter points by x, y, z bounds, but with small N it's okay.
            for p in self.candidate_points:
                x, y, z = p
                if self.can_place(x, y, z, w, h, d):
                    valid.append(p)
            
            # If no candidate points worked, try (0,0,0) explicitly (shouldn't happen if set is maintained correctly)
            if not valid and self.can_place(0, 0, 0, w, h, d):
                valid.append((0, 0, 0))
                
            return valid

        def place(self, idx, x, y, z, w, h, d):
            self.boxes.append((idx, x, y, z, w, h, d))
            self.volume += w * h * d
            # Update candidate points: add all 8 corners of the new box
            for dx in [0, w]:
                for dy in [0, h]:
                    for dz in [0, d]:
                        self.candidate_points.add((x + dx, y + dy, z + dz))
            
            # Optional: Prune candidate points that are strictly inside another box?
            # Or just keep them all. With 40 boxes, max 40*8=320 points. 
            # Checking 320 points * 40 boxes * 40 boxes (overlap check) = 512k ops. 
            # This is fast in Python if optimized, but might be tight.
            # Let's add a simple prune: remove points that are inside the new box's footprint? No, that's complex.
            # Just keep the set.

        def remove(self, idx):
            # For local search, we need to remove boxes.
            # Rebuilding candidate points is expensive. 
            # Alternative: Don't do complex removals. Just use the initial greedy fill.
            # If we do local search, we might need to rebuild the Bin from scratch for the affected bin.
            pass

    def score_placement(assignments):
        """
        Calculate the score for a given assignment.
        assignments: dict {box_idx: bin_id}
        bins: list of Bin objects corresponding to bin_id
        """
        if not assignments:
            return float('inf')
        
        # Determine B and last bin fill
        bin_vols = {}
        for idx, b_id in assignments.items():
            w, h, d = boxes[idx]
            if b_id not in bin_vols:
                bin_vols[b_id] = 0
            bin_vols[b_id] += w * h * d
        
        if not bin_vols:
            return float('inf')
            
        B = len(bin_vols)
        last_bin_id = max(bin_vols.keys())
        last_vol = bin_vols[last_bin_id]
        
        # Score: B + 0.1 * (1 - last_vol / container_vol)
        return B + 0.1 * (1 - last_vol / container_vol)

    def run_greedy(order):
        """
        Run a greedy packing with a specific box order.
        Returns (score, assignments)
        """
        bins = []
        assignments = {}
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            if bins:
                # Try to place in existing bins
                # Heuristic: Try bins with most volume first? Or least?
                # "Best Fit" usually means trying to fill the most full bin to keep others open.
                # Let's sort bins by volume descending (most full first)
                bin_indices = sorted(range(len(bins)), key=lambda i: -bins[i].volume)
                
                for b_id in bin_indices:
                    b = bins[b_id]
                    cands = b.get_valid_candidates(w, h, d)
                    if cands:
                        # Pick the candidate that minimizes "distance" to origin? 
                        # Or just the first one? 
                        # To be deterministic and fast, pick (0,0,0) if available, else first.
                        # Actually, picking the one with smallest (x+y+z) tends to compact the bin.
                        cands.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
                        cx, cy, cz = cands[0]
                        b.place(idx, cx, cy, cz, w, h, d)
                        assignments[idx] = b_id
                        placed = True
                        break
            
            if not placed:
                # Open new bin
                b_id = len(bins)
                new_bin = Bin()
                new_bin.place(idx, 0, 0, 0, w, h, d)
                bins.append(new_bin)
                assignments[idx] = b_id
        
        # Post-processing: Try to move boxes from the last bin to earlier bins to improve last-bin fill?
        # Or move boxes from high-numbered bins to lower-numbered bins to reduce B?
        # Simple Local Search:
        if time.time() < deadline - 1.0:
            improve_local(bins, assignments)
            
        s = score_placement(assignments)
        return s, assignments

    def improve_local(bins, assignments):
        """
        Simple local search to reduce B and improve last bin fill.
        1. Try to move boxes from the highest numbered bin to lower numbered bins.
        2. If that fails, try to move boxes from other bins INTO the highest numbered bin (to increase its fill).
        """
        if not bins:
            return
            
        # Get list of occupied bins
        occupied = [b_id for b_id in range(len(bins)) if bins[b_id].boxes]
        if not occupied:
            return
            
        last_bin_id = max(occupied)
        
        # Pass 1: Try to empty the last bin by moving its boxes to earlier bins
        # Sort boxes in last bin by volume descending (larger ones are harder to place)
        last_bin_boxes = list(bins[last_bin_id].boxes)
        last_bin_boxes.sort(key=lambda t: -(t[4]*t[5]*t[6]))
        
        for box_tuple in last_bin_boxes:
            idx, x, y, z, w, h, d = box_tuple
            # Check if box is still in last bin (it might have been moved)
            if assignments.get(idx) != last_bin_id:
                continue
                
            # Try to move to other bins
            # Sort other bins by volume descending (most full first, to avoid fragmenting empty bins)
            other_bins = [b_id for b_id in occupied if b_id != last_bin_id]
            other_bins.sort(key=lambda b: -bins[b].volume)
            
            moved = False
            for target_b in other_bins:
                b = bins[target_b]
                cands = b.get_valid_candidates(w, h, d)
                if cands:
                    cands.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
                    cx, cy, cz = cands[0]
                    # Perform move
                    # Remove from last_bin
                    # We need a remove method. Let's implement a simple remove for Bin.
                    # Since we don't have a robust remove, we will rebuild the bin's boxes list.
                    
                    # Remove from last_bin.boxes
                    last_bin.boxes = [bt for bt in last_bin.boxes if bt[0] != idx]
                    last_bin.volume -= w * h * d
                    # Rebuild candidate points for last_bin? 
                    # This is expensive. 
                    # Alternative: Just mark it as moved and don't worry about candidate points for the rest of the pass?
                    # No, candidate points are needed for subsequent moves.
                    # Let's rebuild candidate points for last_bin.
                    last_bin.candidate_points = {(0,0,0)}
                    for (_, bx, by, bz, bw, bh, bd) in last_bin.boxes:
                        for dx in [0, bw]:
                            for dy in [0, bh]:
                                for dz in [0, bd]:
                                    last_bin.candidate_points.add((bx + dx, by + dy, bz + dz))
                                    
                    # Place in target_bin
                    b.place(idx, cx, cy, cz, w, h, d)
                    assignments[idx] = target_b
                    moved = True
                    break
                    
            if moved:
                # Check if last bin is now empty
                if not bins[last_bin_id].boxes:
                    # Bin is empty. We could remove it, but bin IDs are fixed.
                    # The score function uses max(bin_vols.keys()). If last_bin is empty, 
                    # it won't be in bin_vols, so max will be lower. 
                    # So B decreases automatically.
                    pass

        # Pass 2: Try to fill the last bin more by moving boxes FROM other bins INTO it
        # This helps if B is fixed, but last bin is underfilled.
        # Actually, if we moved boxes OUT of last bin in Pass 1, last bin might be empty.
        # If last bin is NOT empty, we want to maximize its volume.
        
        # Re-identify last bin
        occupied = [b_id for b_id in range(len(bins)) if bins[b_id].boxes]
        if not occupied:
            return
        last_bin_id = max(occupied)
        
        # Try to move boxes from other bins into last_bin
        other_bins = [b_id for b_id in occupied if b_id != last_bin_id]
        # Sort other bins by volume descending? Or just iterate?
        # Let's try to move smaller boxes first to fill gaps? 
        # Or larger boxes? 
        # Let's just iterate through all boxes in other bins.
        
        for b_id in other_bins:
            # Make a copy of boxes to iterate
            boxes_in_bin = list(bins[b_id].boxes)
            # Sort by volume ascending (smaller boxes are easier to fit into gaps)
            boxes_in_bin.sort(key=lambda t: t[4]*t[5]*t[6])
            
            for box_tuple in boxes_in_bin:
                idx, x, y, z, w, h, d = box_tuple
                if assignments.get(idx) != b_id:
                    continue
                    
                last_bin = bins[last_bin_id]
                cands = last_bin.get_valid_candidates(w, h, d)
                if cands:
                    cands.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
                    cx, cy, cz = cands[0]
                    
                    # Move
                    # Remove from b_id
                    bins[b_id].boxes = [bt for bt in bins[b_id].boxes if bt[0] != idx]
                    bins[b_id].volume -= w * h * d
                    # Rebuild candidates for b_id
                    bins[b_id].candidate_points = {(0,0,0)}
                    for (_, bx, by, bz, bw, bh, bd) in bins[b_id].boxes:
                        for dx in [0, bw]:
                            for dy in [0, bh]:
                                for dz in [0, bd]:
                                    bins[b_id].candidate_points.add((bx + dx, by + dy, bz + dz))
                                    
                    # Place in last_bin
                    last_bin.place(idx, cx, cy, cz, w, h, d)
                    assignments[idx] = last_bin_id

    # --- Main Execution Loop ---
    
    best_score = float('inf')
    best_assignments = None
    
    # Strategy: Multi-start
    # 1. Try sorted by volume descending (Standard FFD)
    # 2. Try random permutations if time permits
    
    orders = []
    # Order 1: Volume Desc
    orders.append([idx for idx, _ in indexed_boxes])
    
    # Order 2: Volume Asc (Worst Fit Decreasing? No, usually not good, but maybe for last bin fill?)
    # Let's stick to variations of Descending Volume with slight noise.
    
    # Generate a few random orders
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        # Random shuffle, but keep large items likely to be first?
        # Or just pure random.
        rand_order = [idx for idx, _ in indexed_boxes]
        random.shuffle(rand_order)
        orders.append(rand_order)
        
        # Also try "Heuristic Random": Sort by volume, but swap some adjacent items
        # This creates diverse but structured orders.
        if i % 2 == 0 and i > 0:
            semi_rand = list(orders[0])
            for _ in range(5):
                if len(semi_rand) > 1:
                    i1 = random.randint(0, len(semi_rand)-2)
                    i2 = i1 + 1
                    semi_rand[i1], semi_rand[i2] = semi_rand[i2], semi_rand[i1]
            orders.append(semi_rand)

    for order in orders:
        if time.time() > deadline - 1.0:
            break
            
        s, assignments = run_greedy(order)
        
        if s < best_score:
            best_score = s
            best_assignments = assignments.copy()
            
        # If we achieved a perfect score (or close), we can stop?
        # Min score is 1.0 (1 bin, full). 
        # If best_score < 1.5, it's pretty good.
        # But we don't know the theoretical min.
        # Let's just run all starts if time permits.

    # Fallback if no valid assignment found (shouldn't happen)
    if best_assignments is None:
        # Simple shelf packing fallback
        placements = [None] * n
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for idx, (w, h, d) in indexed_boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements[idx] = (b, x, y, z, w, h, d)
        return placements

    # Construct final result from best_assignments
    # We need to recover the exact positions. 
    # The `run_greedy` function modifies `bins` which are local to it. 
    # We only returned `assignments`. 
    # We need to re-run the placement logic for the best order to get coordinates?
    # Or we can store the bins in the best case.
    
    # Let's modify run_greedy to return bins as well, or just re-run the best order.
    # Re-running is safer and simple.
    
    # Find the order that produced best_assignments? 
    # We didn't store which order produced it. 
    # Let's just re-run the best order found in the loop? 
    # Actually, `run_greedy` returns `assignments`. 
    # To get coordinates, we need the `bins` state.
    
    # Let's refactor: `run_greedy` returns (score, assignments, bins)
    
    # Let's re-implement the loop to keep the bins.
    
    best_score = float('inf')
    best_result = None
    
    for order in orders:
        if time.time() > deadline - 1.0:
            break
            
        # Re-run greedy to get bins
        bins = []
        assignments = {}
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            if bins:
                bin_indices = sorted(range(len(bins)), key=lambda i: -bins[i].volume)
                for b_id in bin_indices:
                    b = bins[b_id]
                    cands = b.get_valid_candidates(w, h, d)
                    if cands:
                        cands.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
                        cx, cy, cz = cands[0]
                        b.place(idx, cx, cy, cz, w, h, d)
                        assignments[idx] = b_id
                        placed = True
                        break
            if not placed:
                b_id = len(bins)
                new_bin = Bin()
                new_bin.place(idx, 0, 0, 0, w, h, d)
                bins.append(new_bin)
                assignments[idx] = b_id
        
        # Local search
        if time.time() < deadline - 1.0:
            improve_local(bins, assignments)
            
        # Calculate score
        s = score_placement(assignments)
        
        if s < best_score:
            best_score = s
            # Construct result
            result = [None] * n
            for idx in range(n):
                if idx in assignments:
                    b_id = assignments[idx]
                    # Find position in bin
                    pos = None
                    for (b_idx, x, y, z, w, h, d) in bins[b_id].boxes:
                        if b_idx == idx:
                            pos = (b_id, x, y, z, w, h, d)
                            break
                    if pos:
                        result[idx] = pos
                    else:
                        # Should not happen
                        result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                else:
                    result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            
            # Verify validity
            valid = True
            bins_boxes = {}
            for idx in range(n):
                b, x, y, z, w, h, d = result[idx]
                if b not in bins_boxes:
                    bins_boxes[b] = []
                bins_boxes[b].append((x, y, z, w, h, d))
                
            for b, bl in bins_boxes.items():
                for i in range(len(bl)):
                    for j in range(i+1, len(bl)):
                        if is_overlap(bl[i][0], bl[i][1], bl[i][2], bl[i][3], bl[i][4], bl[i][5],
                                      bl[j][0], bl[j][1], bl[j][2], bl[j][3], bl[j][4], bl[j][5]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_result = result
            else:
                # If invalid, don't update best_result, but maybe we should? 
                # Better to keep the last valid one.
                pass

    if best_result is None:
        # Fallback to simple shelf pack
        placements = [None] * n
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for idx, (w, h, d) in indexed_boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements[idx] = (b, x, y, z, w, h, d)
        return placements

    return best_result

# EVOLVE-BLOCK-END
