"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def pack(boxes, container):
    """
    3D bin packing with randomized restarts and local search.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    # Precompute box volumes and max dims
    box_vols = [w * h * d for w, h, d in boxes]
    box_max_dim = [max(w, h, d) for w, h, d in boxes]
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) in bin without overlap and within bounds."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_best_spot(bin_boxes, w, h, d):
        """
        Find the best (x, y, z) position for a box in a given bin.
        Minimizes max(x, y, z) to pack tightly toward origin.
        Returns (x, y, z) or None if no valid position found.
        """
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Build candidate coordinate sets from box edges
        x_vals = {0}
        y_vals = {0}
        z_vals = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                x_vals.add(px + pw)
            if py + ph <= H:
                y_vals.add(py + ph)
            if pz + pd <= D:
                z_vals.add(pz + pd)
        
        x_vals = sorted(x for x in x_vals if x + w <= W)
        y_vals = sorted(y for y in y_vals if y + h <= H)
        z_vals = sorted(z for z in z_vals if z + d <= D)
        
        if not x_vals or not y_vals or not z_vals:
            return None
        
        # Try positions sorted by max coordinate (compactness)
        best_pos = None
        best_max_coord = float('inf')
        
        for z in z_vals:
            if z >= best_max_coord:
                continue
            for y in y_vals:
                if y >= best_max_coord:
                    continue
                for x in x_vals:
                    max_c = max(x, y, z)
                    if max_c >= best_max_coord:
                        continue
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        best_max_coord = max_c
                        best_pos = (x, y, z)
                        if max_c == 0:
                            return (0, 0, 0)
        
        return best_pos
    
    def compute_score(bins):
        """Compute score: B + 0.1*(1 - last_bin_fill_fraction)."""
        if not bins:
            return 0.0
        B = len(bins)
        last_fill = sum(w * h * d for (_, _, _, w, h, d) in bins[-1])
        last_frac = last_fill / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1.0 - last_frac)
    
    def build_result(placement, boxes):
        """Build result in original order."""
        result = []
        for i in range(len(boxes)):
            b_idx, x, y, z = placement[i]
            w_i, h_i, d_i = boxes[i]
            result.append((b_idx, x, y, z, w_i, h_i, d_i))
        return result
    
    def validate_placement(bins):
        """Validate that no boxes overlap within each bin and all are within bounds."""
        for bin_boxes in bins:
            for i in range(len(bin_boxes)):
                x1, y1, z1, w1, h1, d1 = bin_boxes[i]
                if x1 + w1 > W or y1 + h1 > H or z1 + d1 > D:
                    return False
                for j in range(i + 1, len(bin_boxes)):
                    x2, y2, z2, w2, h2, d2 = bin_boxes[j]
                    if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                            y1 + h1 <= y2 or y2 + h2 <= y1 or
                            z1 + d1 <= z2 or z2 + d2 <= z1):
                        return False
        return True
    
    def greedy_pack(order):
        """
        Greedy best-fit decreasing pack using the given order of box indices.
        Returns (bins, placement) or None if timeout.
        """
        bins = []
        placement = {}
        
        for idx in order:
            w, h, d = boxes[idx]
            
            # Try to place in existing bins
            best_bin = -1
            best_pos = None
            best_max_coord = float('inf')
            
            for b_idx in range(len(bins)):
                spot = find_best_spot(bins[b_idx], w, h, d)
                if spot is not None:
                    x, y, z = spot
                    max_c = max(x, y, z)
                    if max_c < best_max_coord:
                        best_max_coord = max_c
                        best_bin = b_idx
                        best_pos = spot
            
            if best_bin == -1:
                # Create new bin
                if w <= W and h <= H and d <= D:
                    bins.append([])
                    best_bin = len(bins) - 1
                    best_pos = (0, 0, 0)
                else:
                    # Box doesn't fit in any bin - shouldn't happen
                    return None
            
            bins[best_bin].append(best_pos + (w, h, d))
            placement[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2])
        
        return bins, placement
    
    def local_search(bins, placement, order):
        """
        Try to eliminate bins by moving boxes from last bin to earlier bins.
        Returns improved (bins, placement) or original if no improvement.
        """
        if len(bins) <= 1:
            return bins, placement
        
        # Try to eliminate the last bin
        last_bin_idx = len(bins) - 1
        last_boxes = list(bins[last_bin_idx])
        
        if not last_boxes:
            bins.pop()
            return bins, placement
        
        # Try to move each box from last bin to earlier bins
        saved_bins = [list(b) for b in bins]
        saved_placement = dict(placement)
        
        bins[last_bin_idx] = []
        
        success = True
        for box in last_boxes:
            x, y, z, w, h, d = box
            placed = False
            
            for b_idx in range(len(bins) - 1):
                spot = find_best_spot(bins[b_idx], w, h, d)
                if spot is not None:
                    bx, by, bz = spot
                    bins[b_idx].append((bx, by, bz, w, h, d))
                    
                    # Find and update placement for this box
                    for idx in range(n):
                        if placement[idx] == (last_bin_idx, x, y, z):
                            placement[idx] = (b_idx, bx, by, bz)
                            break
                    placed = True
                    break
            
            if not placed:
                # Restore state
                bins[:] = [list(b) for b in saved_bins]
                placement.clear()
                placement.update(saved_placement)
                success = False
                break
        
        if success:
            bins.pop()
            # Update placements that referenced old last bin index
            for idx in placement:
                if placement[idx][0] > last_bin_idx:
                    placement[idx] = (placement[idx][0] - 1, placement[idx][1], placement[idx][2], placement[idx][3])
        
        return bins, placement
    
    # Best solution tracking
    best_bins = None
    best_placement = None
    best_score = float('inf')
    
    # Base ordering: volume descending, then max dim descending
    base_order = list(range(n))
    base_order.sort(key=lambda i: (-box_vols[i], -box_max_dim[i]))
    
    # Multiple restarts with perturbed orderings
    num_restarts = 8
    per_restart_time = min(15.0, (deadline - t_start - 5.0) / max(1, num_restarts))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Create perturbed ordering
        if restart == 0:
            order = list(base_order)
        else:
            order = list(base_order)
            # Randomly swap some adjacent pairs to perturb
            num_swaps = random.randint(2, min(8, n // 2))
            for _ in range(num_swaps):
                pos = random.randint(0, len(order) - 2)
                order[pos], order[pos + 1] = order[pos + 1], order[pos]
        
        # Greedy pack
        result = greedy_pack(order)
        if result is None:
            continue
        
        bins, placement = result
        
        # Local search
        bins, placement = local_search(bins, placement, order)
        
        # Validate
        if not validate_placement(bins):
            continue
        
        score = compute_score(bins)
        if score < best_score:
            best_score = score
            best_bins = [list(b) for b in bins]
            best_placement = dict(placement)
    
    # If we have a valid best solution, return it
    if best_bins is not None and best_placement is not None:
        return build_result(best_placement, boxes)
    
    # Fallback: simple sequential packing
    bins = []
    placement = {}
    for i, (w, h, d) in enumerate(boxes):
        placed = False
        for b_idx in range(len(bins)):
            if can_place(bins[b_idx], 0, 0, 0, w, h, d):
                bins[b_idx].append((0, 0, 0, w, h, d))
                placement[i] = (b_idx, 0, 0, 0)
                placed = True
                break
        if not placed:
            bins.append([(0, 0, 0, w, h, d)])
            placement[i] = (len(bins) - 1, 0, 0, 0)
    
    return build_result(placement, boxes)
# EVOLVE-BLOCK-END
