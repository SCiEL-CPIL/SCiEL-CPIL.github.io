"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import math
import random
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 145s to be safe.
    # The prompt says 120s in the text but the directive says 150s wall-clock kill.
    # Let's use 145s as the hard deadline for computation.
    deadline = start_time + 145.0
    
    W, H, D = container
    container_vol = W * H * D
    n_boxes = len(boxes)
    
    if n_boxes == 0:
        return []
        
    # Pre-compute box dimensions and volumes
    box_dims = np.array(boxes, dtype=np.float64)
    box_vols = box_dims[:, 0] * box_dims[:, 1] * box_dims[:, 2]
    
    # Check if any box is too big
    # If a box doesn't fit, the problem is technically unsolvable in a single bin, 
    # but we must place them. The contract implies valid inputs, but let's be robust.
    # If a box is larger than the container, it cannot be placed in a standard bin.
    # However, the problem states "places every box... into numbered bins".
    # Usually, this implies the boxes fit. If not, we just place them in their own bin?
    # Actually, if w > W, it can't be placed. 
    # The contract says "valid inputs", so we assume boxes fit.
    
    def get_candidate_positions(bin_items, w, h, d):
        """
        Generate candidate (x, y, z) positions for a box of size (w, h, d)
        in a bin containing bin_items.
        bin_items: list of (x, y, z, w, h, d)
        """
        if not bin_items:
            return [(0, 0, 0)]
            
        xs = {0}
        ys = {0}
        zs = {0}
        
        for ex, ey, ez, ew, eh, ed in bin_items:
            # Add coordinates to the right, above, and behind each item
            # Only add if the resulting position fits within the container bounds
            # for the new box
            x_new = ex + ew
            if x_new + w <= W:
                xs.add(x_new)
                
            y_new = ey + eh
            if y_new + h <= H:
                ys.add(y_new)
                
            z_new = ez + ed
            if z_new + d <= D:
                zs.add(z_new)
                
        # Also, it is sometimes beneficial to try positions aligned with the 
        # "inner" corners, but the "outer" corners (ex+ew) are the standard 
        # Bottom-Left-Far (BLF) heuristic positions.
        
        # Convert to lists for iteration
        xs = list(xs)
        ys = list(ys)
        zs = list(zs)
        
        # Sort to prioritize "bottom-left-far" (min x, then min y, then min z)
        # This helps in finding a feasible spot quickly and often leads to good packing.
        # However, for local search, we might want to evaluate more.
        # For the construction phase, BLF is good.
        
        candidates = []
        for x in xs:
            for y in ys:
                for z in zs:
                    # Quick bounds check
                    if x + w > W or y + h > H or z + d > D:
                        continue
                    candidates.append((x, y, z))
                    
        return candidates

    def check_overlap(x, y, z, w, h, d, bin_items):
        """
        Check if a box at (x,y,z) with dims (w,h,d) overlaps with any item in bin_items.
        """
        for ex, ey, ez, ew, eh, ed in bin_items:
            # Check separation in all 3 dimensions
            # If they are separated in ANY dimension, they do not overlap
            if x + w <= ex or ex + ew <= x:
                continue
            if y + h <= ey or ey + eh <= y:
                continue
            if z + d <= ez or ez + ed <= z:
                continue
            # If we reach here, they overlap in all 3 dimensions
            return True
        return False

    def construct_packing(order):
        """
        Construct a packing given a specific order of box indices.
        Returns:
        - placements: list of (bin_idx, x, y, z, w, h, d)
        - bins: list of lists of (x, y, z, w, h, d)
        - success: bool (True if all boxes placed)
        """
        bins = []
        placements = [None] * n_boxes
        
        for idx in order:
            w, h, d = box_dims[idx]
            placed = False
            
            # Try existing bins
            for b_idx in range(len(bins)):
                bin_items = bins[b_idx]
                # Generate candidates
                candidates = get_candidate_positions(bin_items, w, h, d)
                
                # Sort candidates by BLF heuristic (x, then y, then z)
                # This is deterministic for a given set of candidates
                candidates.sort(key=lambda p: (p[0], p[1], p[2]))
                
                for x, y, z in candidates:
                    if not check_overlap(x, y, z, w, h, d, bin_items):
                        # Place it
                        bins[b_idx].append((x, y, z, w, h, d))
                        placements[idx] = (b_idx, x, y, z, w, h, d)
                        placed = True
                        break
                
                if placed:
                    break
            
            if not placed:
                # Try to place in a new bin
                # A new bin is always feasible if the box fits in the container
                if w <= W and h <= H and d <= D:
                    bins.append([(0, 0, 0, w, h, d)])
                    placements[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
                    placed = True
                else:
                    # Box doesn't fit in container? Should not happen.
                    # Return failure
                    return None, None, False
        
        return placements, bins, True

    def calculate_score(placements, bins):
        """
        Calculate the score: mean B + 0.1 * (1 - last-bin occupied volume fraction)
        B is the number of occupied bins.
        """
        if not bins:
            return float('inf')
            
        # Count occupied bins
        # Note: bin labels are indices in the list.
        # B is the count of bins that have at least one box.
        # Since we only add boxes to bins, all bins in the list are occupied.
        B = len(bins)
        
        # Identify the last bin
        # "The largest label defines the last bin"
        # Our labels are 0..len(bins)-1. So the last bin is bins[-1].
        # But wait, the score formula says "last-bin occupied volume fraction".
        # Does "last bin" mean the bin with the highest index? Yes.
        # And "occupied volume fraction" is volume of boxes in that bin / container volume.
        
        last_bin_idx = B - 1
        last_bin_items = bins[last_bin_idx]
        
        last_bin_vol = sum(item[3] * item[4] * item[5] for item in last_bin_items)
        
        if container_vol == 0:
            return float('inf')
            
        fraction = last_bin_vol / container_vol
        # The term is 0.1 * (1 - fraction)
        # We want to MINIMIZE the score.
        # So we want to MINIMIZE B and MAXIMIZE the fraction of the last bin.
        # Maximizing the fraction of the last bin means filling the last bin as much as possible.
        # This is interesting: it penalizes a nearly empty last bin.
        
        score = B + 0.1 * (1 - fraction)
        return score

    # --- Main Search Logic ---
    
    best_score = float('inf')
    best_placements = None
    best_bins = None
    
    # Generate a few different sorting orders
    # 1. Volume Descending (FFD)
    order_vol_desc = list(np.argsort(-box_vols))
    
    # 2. Surface Area Descending
    surface_areas = 2 * (box_dims[:, 0]*box_dims[:, 1] + box_dims[:, 1]*box_dims[:, 2] + box_dims[:, 0]*box_dims[:, 2])
    order_surf_desc = list(np.argsort(-surface_areas))
    
    # 3. Max Dimension Descending
    max_dims = np.max(box_dims, axis=1)
    order_max_dim_desc = list(np.argsort(-max_dims))
    
    # 4. Random Shuffle (multiple times)
    random_orders = []
    for _ in range(10):
        random_order = list(range(n_boxes))
        random.shuffle(random_order)
        random_orders.append(random_order)
        
    # 5. Volume Ascending (LFD - Last Fit Decreasing? No, Last Fit Decreasing is LFD, but sorting ascending is LFD's opposite? No, FFD sorts descending. LFD sorts descending but tries last bin first. Here we are doing First Fit construction. Sorting ascending is "First Fit Increasing")
    order_vol_asc = list(np.argsort(box_vols))
    
    # Combine all orders
    all_orders = [order_vol_desc, order_surf_desc, order_max_dim_desc, order_vol_asc] + random_orders
    
    # We can also add a "hybrid" order: sort by volume, but with some noise
    # Or just rely on the random shuffles.
    
    # Iterate through orders
    for order in all_orders:
        if time.time() > deadline:
            break
            
        placements, bins, success = construct_packing(order)
        
        if not success:
            continue
            
        score = calculate_score(placements, bins)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            best_bins = bins
            
            # If we found a perfect packing (1 bin, full), we can stop?
            # Score would be 1 + 0.1*(1-1) = 1.0.
            # Can we do better?
            # B is at least 1.
            # The term 0.1*(1-fraction) is >= 0.
            # So minimum possible score is 1.0.
            if best_score <= 1.0 + 1e-9:
                break

    # If we didn't find any valid packing (should not happen), return empty or fallback
    if best_placements is None:
        # Fallback: place each box in its own bin
        placements = []
        for i in range(n_boxes):
            w, h, d = box_dims[i]
            placements.append((i, 0, 0, 0, w, h, d))
        return placements

    return best_placements
# EVOLVE-BLOCK-END
