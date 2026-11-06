"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s
    
    # Pre-compute box properties: (index, volume, w, h, d)
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, w * h * d, w, h, d))
    
    # Sort indices by volume descending for initial packing
    sorted_indices = sorted(range(n), key=lambda i: box_data[i][1], reverse=True)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for _, bx, by, bz, bw, bh, bd in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return False
        return True

    def find_spot(bin_boxes, w, h, d):
        candidates = [(0, 0, 0)]
        for _, bx, by, bz, bw, bh, bd in bin_boxes:
            # Front points
            candidates.append((bx, by, bz + bd))
            candidates.append((bx + bw, by, bz))
            candidates.append((bx, by + bh, bz))
            candidates.append((bx + bw, by + bh, bz))
            candidates.append((bx + bw, by, bz + bd))
            candidates.append((bx, by + bh, bz + bd))
            candidates.append((bx + bw, by + bh, bz + bd))
            
        unique_candidates = list(set(candidates))
        unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for cx, cy, cz in unique_candidates:
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        return None

    def initial_pack():
        bins = []  # List of lists of (index, x, y, z, w, h, d)
        placements = [None] * n
        
        for idx in sorted_indices:
            i, vol, w, h, d = box_data[idx]
            
            best_bin_idx = -1
            best_pos = None
            
            for b_idx, bin_boxes in enumerate(bins):
                spot = find_spot(bin_boxes, w, h, d)
                if spot is not None:
                    best_bin_idx = b_idx
                    best_pos = spot
                    break # First Fit Decreasing (FFD) for speed
            
            if best_bin_idx == -1:
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            
            bins[best_bin_idx].append((i, best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placements[i] = (best_bin_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            
        return bins, placements

    def local_search(bins, placements):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
                
            last_bin_idx = num_bins - 1
            last_bin_boxes = bins[last_bin_idx]
            
            if not last_bin_boxes:
                # Last bin is empty, remove it
                bins.pop()
                # Update placements for boxes that were in this bin? 
                # If it's empty, no boxes were in it. 
                # But if we just emptied it, the boxes moved to other bins.
                # Wait, if last bin is empty, it means we successfully moved all boxes out.
                # We should remove the empty bin to reduce B.
                # But we need to update the bin indices in placements for all boxes in bins > last_bin_idx?
                # Since last_bin_idx is the max, no bins have higher index.
                # So just popping is fine.
                improved = True
                continue

            # Try to move each box from last bin to an earlier bin
            # Sort boxes in last bin by volume descending (hardest to place first)
            # We need to be careful with indices if we modify the list.
            # We iterate over a snapshot of the boxes in the last bin.
            
            # Get list of (index, x, y, z, w, h, d) in last bin
            boxes_in_last = list(last_bin_boxes)
            # Sort by volume desc
            boxes_in_last.sort(key=lambda b: b[4]*b[5]*b[6], reverse=True)
            
            for box_info in boxes_in_last:
                i, bx, by, bz, w, h, d = box_info
                
                # Check if this box is still in the last bin
                # It might have been moved by a previous iteration in this while loop?
                # No, we only move one box and then `improved=True` breaks the inner loop 
                # and restarts the while loop. 
                # So `boxes_in_last` is a snapshot at the start of the while iteration.
                # If the box was moved in a previous *while* iteration, it wouldn't be in the new snapshot.
                # If it was moved in a previous *inner* iteration? We break inner loop on success.
                # So it's safe.
                
                # Remove from last bin
                if box_info not in bins[last_bin_idx]:
                    continue
                bins[last_bin_idx].remove(box_info)
                
                moved = False
                # Try to place in bins 0 to last_bin_idx - 1
                for b_idx in range(last_bin_idx):
                    spot = find_spot(bins[b_idx], w, h, d)
                    if spot is not None:
                        # Move to b_idx
                        bins[b_idx].append((i, spot[0], spot[1], spot[2], w, h, d))
                        placements[i] = (b_idx, spot[0], spot[1], spot[2], w, h, d)
                        moved = True
                        improved = True
                        break
                
                if not moved:
                    # Put back in last bin
                    bins[last_bin_idx].append(box_info)
                    # Placements[i] remains unchanged (still points to last_bin_idx, bx, by, bz)
                    # But wait, if we put it back, its position should be the original one?
                    # Yes, we just reversed the move.
                    # However, if we want to try a different position in the last bin?
                    # For now, just put it back.
                    
                if moved:
                    break # Restart while loop with new configuration
        
        return bins, placements

    # Run initial pack
    bins, placements = initial_pack()
    
    # Run local search
    bins, placements = local_search(bins, placements)
    
    # Final check: Ensure all placements are valid and within deadline
    # The local search should have maintained validity.
    
    return placements

# EVOLVE-BLOCK-END
