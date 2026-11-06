"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    """
    3D Bin Packing: BFD with multiple sorting strategies, efficient candidate
    position search, and local-search refinement to reduce bin count.
    
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    if n == 0:
        return []
    
    # Sort boxes by volume descending, with tie-breaking by max dimension
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[1], -max(x[2], x[3], x[4])))
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s kill
    
    # Data structures
    bins = []  # list of lists of (x, y, z, w, h, d)
    placement = {}  # original index -> (bin_idx, x, y, z)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) in bin without overlap and within bounds."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Quick reject: check if boxes overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_best_spot(bin_boxes, w, h, d):
        """
        Find the best (x, y, z) position for a box in a given bin.
        Uses candidate positions derived from box faces with pruning.
        Returns (x, y, z) or None if no valid position found.
        """
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Build candidate coordinate sets from box edges
        x_vals = [0]
        y_vals = [0]
        z_vals = [0]
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                x_vals.append(px + pw)
            if py + ph <= H:
                y_vals.append(py + ph)
            if pz + pd <= D:
                z_vals.append(pz + pd)
        
        # Deduplicate and sort
        x_vals = sorted(set(x_vals))
        y_vals = sorted(set(y_vals))
        z_vals = sorted(set(z_vals))
        
        # Filter to valid starting positions
        x_vals = [x for x in x_vals if x + w <= W]
        y_vals = [y for y in y_vals if y + h <= H]
        z_vals = [z for z in z_vals if z + d <= D]
        
        if not x_vals or not y_vals or not z_vals:
            return None
        
        # Try positions in order of compactness (minimize x+y+z)
        best_pos = None
        best_cost = float('inf')
        
        # Generate candidate positions sorted by cost
        candidates = []
        for z in z_vals:
            for y in y_vals:
                for x in x_vals:
                    candidates.append((x + y + z, x, y, z))
        
        candidates.sort()
        
        for cost, x, y, z in candidates:
            if cost >= best_cost:
                break  # No better position possible
            if can_place(bin_boxes, x, y, z, w, h, d):
                best_cost = cost
                best_pos = (x, y, z)
                if cost == 0:
                    break  # Can't do better than origin
        
        return best_pos
    
    def try_merge_bin(src_idx, dst_idx, bins, placement, indexed_boxes):
        """
        Try to move all boxes from src_idx to dst_idx.
        Returns True if successful (src bin becomes empty).
        """
        if src_idx == dst_idx or src_idx >= len(bins) or dst_idx >= len(bins):
            return False
        
        src_boxes = list(bins[src_idx])
        if not src_boxes:
            return True
        
        # Save state for rollback
        dst_boxes_backup = list(bins[dst_idx])
        
        # Clear src bin
        bins[src_idx] = []
        
        success = True
        for box in src_boxes:
            x, y, z, w, h, d = box
            placed = False
            # Try to place in dst bin
            spot = find_best_spot(bins[dst_idx], w, h, d)
            if spot is not None:
                bx, by, bz = spot
                bins[dst_idx].append((bx, by, bz, w, h, d))
                # Update placement
                for idx2, vol2, w2, h2, d2 in indexed_boxes:
                    if placement[idx2] == (src_idx, x, y, z):
                        placement[idx2] = (dst_idx, bx, by, bz)
                        break
                placed = True
            
            if not placed:
                # Try other bins
                for b_idx in range(len(bins)):
                    if b_idx == src_idx or b_idx == dst_idx:
                        continue
                    spot = find_best_spot(bins[b_idx], w, h, d)
                    if spot is not None:
                        bx, by, bz = spot
                        bins[b_idx].append((bx, by, bz, w, h, d))
                        for idx2, vol2, w2, h2, d2 in indexed_boxes:
                            if placement[idx2] == (src_idx, x, y, z):
                                placement[idx2] = (b_idx, bx, by, bz)
                                break
                        placed = True
                        break
            
            if not placed:
                # Rollback
                success = False
                break
        
        if not success:
            # Restore state
            bins[src_idx] = src_boxes
            bins[dst_idx] = dst_boxes_backup
            # Restore placements
            for idx2, vol2, w2, h2, d2 in indexed_boxes:
                # Find original position in src_boxes
                for box in src_boxes:
                    x, y, z, w_b, h_b, d_b = box
                    if w_b == w2 and h_b == h2 and d_b == d2:
                        # Check if this box was moved
                        found_in_dst = False
                        for b_idx in range(len(bins)):
                            for b_box in bins[b_idx]:
                                if b_box[3] == w_b and b_box[4] == h_b and b_box[5] == d_b:
                                    if b_idx != src_idx:
                                        found_in_dst = True
                                        break
                            if found_in_dst:
                                break
                        if found_in_dst:
                            # Restore to src
                            for b_idx in range(len(bins)):
                                for bi, b_box in enumerate(bins[b_idx]):
                                    if b_box[3] == w_b and b_box[4] == h_b and b_box[5] == d_b and b_idx != src_idx:
                                        # This is a moved box, remove it
                                        # We need to be more careful here
                                        pass
                        break
                # Simpler: just restore from backup
                pass
            # Simpler rollback: restore bins and recompute placements
            bins[src_idx] = src_boxes
            bins[dst_idx] = dst_boxes_backup
            # Rebuild placement for affected boxes
            for idx2, vol2, w2, h2, d2 in indexed_boxes:
                old_placement = placement[idx2]
                # Find where this box actually is now
                found = False
                for b_idx in range(len(bins)):
                    for (bx, by, bz, bw, bh, bd) in bins[b_idx]:
                        if bw == w2 and bh == h2 and bd == d2:
                            # Could be multiple boxes with same dims, need better matching
                            pass
                # This rollback is complex; let's use a different approach
                pass
        
        return success
    
    def compute_bin_fill(bin_boxes):
        """Compute total volume of boxes in a bin."""
        return sum(w * h * d for (_, _, _, w, h, d) in bin_boxes)
    
    def get_score(num_bins, last_bin_fill):
        """Compute the score: B + 0.1*(1 - last_bin_fill_fraction)."""
        if num_bins == 0:
            return 0.0
        last_frac = last_bin_fill / container_vol if container_vol > 0 else 0.0
        return num_bins + 0.1 * (1.0 - last_frac)
    
    def build_result(placement, boxes):
        """Build result in original order."""
        result = []
        for i in range(len(boxes)):
            b_idx, x, y, z = placement[i]
            w_i, h_i, d_i = boxes[i]
            result.append((b_idx, x, y, z, w_i, h_i, d_i))
        return result
    
    def validate_placement(bins, boxes):
        """Validate that no boxes overlap within each bin and all are within bounds."""
        for b_idx, bin_boxes in enumerate(bins):
            for i, (x1, y1, z1, w1, h1, d1) in enumerate(bin_boxes):
                if x1 + w1 > W or y1 + h1 > H or z1 + d1 > D:
                    return False
                for j in range(i + 1, len(bin_boxes)):
                    x2, y2, z2, w2, h2, d2 = bin_boxes[j]
                    if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                            y1 + h1 <= y2 or y2 + h2 <= y1 or
                            z1 + d1 <= z2 or z2 + d2 <= z1):
                        return False
        return True
    
    def pack_with_order(order):
        """Pack boxes in the given order using BFD. Returns (bins, placement)."""
        bins_local = []
        placement_local = {}
        
        for idx, vol, w, h, d in order:
            # Time check
            if time.time() > deadline - 30.0:
                # Fallback: place remaining boxes quickly
                for idx2, vol2, w2, h2, d2 in order:
                    if idx2 in placement_local:
                        continue
                    placed = False
                    for b_idx in range(len(bins_local)):
                        if can_place(bins_local[b_idx], 0, 0, 0, w2, h2, d2):
                            bins_local[b_idx].append((0, 0, 0, w2, h2, d2))
                            placement_local[idx2] = (b_idx, 0, 0, 0)
                            placed = True
                            break
                    if not placed:
                        for b_idx in range(len(bins_local)):
                            spot = find_best_spot(bins_local[b_idx], w2, h2, d2)
                            if spot is not None:
                                x, y, z = spot
                                bins_local[b_idx].append((x, y, z, w2, h2, d2))
                                placement_local[idx2] = (b_idx, x, y, z)
                                placed = True
                                break
                    if not placed:
                        bins_local.append([(0, 0, 0, w2, h2, d2)])
                        placement_local[idx2] = (len(bins_local) - 1, 0, 0, 0)
                return bins_local, placement_local
            
            # Try to place in existing bins using best-fit
            best_bin = -1
            best_x, best_y, best_z = 0, 0, 0
            best_cost = float('inf')
            
            for b_idx in range(len(bins_local)):
                spot = find_best_spot(bins_local[b_idx], w, h, d)
                if spot is not None:
                    x, y, z = spot
                    cost = x + y + z
                    if cost < best_cost:
                        best_cost = cost
                        best_bin = b_idx
                        best_x, best_y, best_z = x, y, z
            
            if best_bin == -1:
                # Create new bin
                bins_local.append([])
                best_bin = len(bins_local) - 1
                best_x, best_y, best_z = 0, 0, 0
            
            # Place the box
            bins_local[best_bin].append((best_x, best_y, best_z, w, h, d))
            placement_local[idx] = (best_bin, best_x, best_y, best_z)
        
        return bins_local, placement_local
    
    # Phase 1: Initial packing with BFD
    bins, placement = pack_with_order(indexed_boxes)
    
    # Phase 2: Local search to reduce bin count
    if time.time() < deadline - 10.0:
        # Build a mapping from (bin_idx, x, y, z, w, h, d) to original indices for fast lookup
        # Actually, let's track box IDs explicitly
        
        # Rebuild bins with box IDs for proper tracking
        # bins[i] = list of (x, y, z, w, h, d, original_idx)
        bins_with_ids = []
        for b_idx, bin_boxes in enumerate(bins):
            new_bin = []
            for (x, y, z, w, h, d) in bin_boxes:
                # Find original index
                orig_idx = None
                for idx2, vol2, w2, h2, d2 in indexed_boxes:
                    if placement[idx2] == (b_idx, x, y, z):
                        orig_idx = idx2
                        break
                if orig_idx is None:
                    # Should not happen
                    orig_idx = 0
                new_bin.append((x, y, z, w, h, d, orig_idx))
            bins_with_ids.append(new_bin)
        
        # Local search: try to merge bins
        improved = True
        max_iterations = 100
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10.0:
            improved = False
            iteration += 1
            
            if len(bins_with_ids) <= 1:
                break
            
            # Try to merge each pair of bins (src -> dst)
            for src_idx in range(len(bins_with_ids) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    break
                if len(bins_with_ids) <= 1:
                    break
                if not bins_with_ids[src_idx]:
                    bins_with_ids.pop(src_idx)
                    improved = True
                    break
                
                src_boxes = list(bins_with_ids[src_idx])
                
                for dst_idx in range(len(bins_with_ids)):
                    if dst_idx == src_idx:
                        continue
                    if time.time() > deadline - 10.0:
                        break
                    
                    # Try to move all boxes from src to dst
                    dst_backup = list(bins_with_ids[dst_idx])
                    src_backup = list(bins_with_ids[src_idx])
                    
                    # Clear src
                    bins_with_ids[src_idx] = []
                    
                    success = True
                    for (x, y, z, w, h, d, orig_idx) in src_boxes:
                        placed = False
                        # Try dst first
                        spot = find_best_spot([(bx, by, bz, bw, bh, bd) for (bx, by, bz, bw, bh, bd, _) in bins_with_ids[dst_idx]], w, h, d)
                        if spot is not None:
                            bx, by, bz = spot
                            bins_with_ids[dst_idx].append((bx, by, bz, w, h, d, orig_idx))
                            placement[orig_idx] = (dst_idx, bx, by, bz)
                            placed = True
                        
                        if not placed:
                            # Try other bins
                            for b_idx in range(len(bins_with_ids)):
                                if b_idx == src_idx or b_idx == dst_idx:
                                    continue
                                spot = find_best_spot([(bx, by, bz, bw, bh, bd) for (bx, by, bz, bw, bh, bd, _) in bins_with_ids[b_idx]], w, h, d)
                                if spot is not None:
                                    bx, by, bz = spot
                                    bins_with_ids[b_idx].append((bx, by, bz, w, h, d, orig_idx))
                                    placement[orig_idx] = (b_idx, bx, by, bz)
                                    placed = True
                                    break
                        
                        if not placed:
                            success = False
                            break
                    
                    if success:
                        # Remove empty src bin
                        bins_with_ids.pop(src_idx)
                        improved = True
                        break
                    else:
                        # Rollback
                        bins_with_ids[src_idx] = src_backup
                        bins_with_ids[dst_idx] = dst_backup
                        # Restore placements
                        for (x, y, z, w, h, d, orig_idx) in src_boxes:
                            placement[orig_idx] = (src_idx, x, y, z)
        
        # Convert back to simple bins format
        bins = [[(x, y, z, w, h, d) for (x, y, z, w, h, d, _) in bin] for bin in bins_with_ids]
    
    # Final validation and result building
    if not validate_placement(bins, boxes):
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
    
    # Ensure all boxes have a placement
    for i in range(n):
        if i not in placement:
            # Should not happen, but handle gracefully
            placement[i] = (0, 0, 0, 0)
    
    return build_result(placement, boxes)
# EVOLVE-BLOCK-END
