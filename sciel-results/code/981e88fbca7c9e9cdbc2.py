"""3D Bin Packing: Best-fit decreasing with tightness-aware positioning and consolidation polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin for 150s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, then by max dimension for tie-breaking
    indexed_boxes = [(i, w, h, d, w*h*d, max(w,h,d)) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[4], x[5], x[1], x[2], x[3]))
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits within container and doesn't overlap with existing boxes."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def position_tightness(bin_boxes, x, y, z, w, h, d):
        """
        Score how tightly a position is packed against existing boxes.
        Lower score = tighter fit. Rewards contact with existing boxes on multiple faces.
        """
        if not bin_boxes:
            return 0.0
        
        # Base score: distance from origin (prefer compact packing)
        score = 0.5 * (x + y + z)
        
        # Reward contact with existing boxes (tighter packing)
        contact_score = 0.0
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            touches = 0
            # Check adjacency
            if abs(x - (px + pw)) < 1e-9 or abs(x + w - px) < 1e-9:
                touches += 1
            if abs(y - (py + ph)) < 1e-9 or abs(y + h - py) < 1e-9:
                touches += 1
            if abs(z - (pz + pd)) < 1e-9 or abs(z + d - pz) < 1e-9:
                touches += 1
            if touches > 0:
                contact_score += touches
        
        # More contacts = better (lower score)
        score -= 0.1 * contact_score
        
        return score
    
    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best position for a box in a bin.
        Returns (x, y, z) or None if no valid position exists.
        Uses candidate points from box faces and scores by tightness.
        """
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Generate candidate coordinates from existing box faces
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                xs.add(px + pw)
            if py + ph <= H:
                ys.add(py + ph)
            if pz + pd <= D:
                zs.add(pz + pd)
        
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        # Limit candidates for performance if bin gets very dense
        # With 40 boxes, max unique coords is 41, so 41^3 = 68921 checks max.
        # This is acceptable.
        
        best_score = float('inf')
        best_pos = None
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        score = position_tightness(bin_boxes, x, y, z, w, h, d)
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_utilization(bin_boxes):
        """Calculate the volume utilization of a bin."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / (W * H * D)
    
    def bin_fragmentation(bin_boxes):
        """
        Estimate fragmentation: how spread out the boxes are.
        Lower = more compact. Uses bounding box volume / actual volume ratio.
        """
        if not bin_boxes:
            return 0.0
        if len(bin_boxes) == 1:
            return 0.0
        
        min_x = min(b[0] for b in bin_boxes)
        min_y = min(b[1] for b in bin_boxes)
        min_z = min(b[2] for b in bin_boxes)
        max_x = max(b[0]+b[3] for b in bin_boxes)
        max_y = max(b[1]+b[4] for b in bin_boxes)
        max_z = max(b[2]+b[5] for b in bin_boxes)
        
        bbox_vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        actual_vol = sum(b[3]*b[4]*b[5] for b in bin_boxes)
        
        if actual_vol == 0:
            return 0.0
        return bbox_vol / actual_vol - 1.0  # 0 = perfect, higher = more fragmented
    
    def place_box(w, h, d):
        """
        Place a box in the best existing bin or create a new one.
        Returns (bin_index, x, y, z).
        """
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        # Try to fit in existing bins first
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_best_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Combined score: prefer bins with higher utilization, lower fragmentation, tighter positions
                util = bin_utilization(bin_boxes)
                frag = bin_fragmentation(bin_boxes)
                pos_tight = position_tightness(bin_boxes, x, y, z, w, h, d)
                
                # We want: high util, low frag, low pos_tight
                # Score to minimize: pos_tight + 0.5*frag - 0.3*util
                score = pos_tight + 0.5 * frag - 0.3 * util
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            # Create a new bin
            bins.append([])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol, maxdim in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in new bins (valid but suboptimal)
            for i in range(idx, n):
                _, w2, h2, d2, _, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Final polish: Consolidate boxes to reduce bin count (B) and improve last-bin fill
    if time.time() < deadline - 10 and len(bins) > 1:
        
        # Helper to update placements after a move
        def update_placement(old_bin, old_pos, old_dims, new_bin, new_pos):
            for p_idx, p in enumerate(placements):
                if p is not None and p[0] == old_bin and p[1] == old_pos[0] and p[2] == old_pos[1] and p[3] == old_pos[2]:
                    # Verify it's the right box by dims
                    if p[4] == old_dims[0] and p[5] == old_dims[1] and p[6] == old_dims[2]:
                        placements[p_idx] = (new_bin, new_pos[0], new_pos[1], new_pos[2], old_dims[0], old_dims[1], old_dims[2])
                        return True
            return False

        # Strategy 1: Try to empty the last bin (highest index) by moving its boxes to other bins
        # This reduces B if successful.
        # We identify the last bin as the one with the max index that has boxes.
        # Note: bins list might have empty slots if we don't remove them, but we append sequentially.
        # So max index is len(bins)-1 if it's not empty.
        
        # Re-identify active bins
        active_bin_indices = [i for i in range(len(bins)) if bins[i]]
        if active_bin_indices:
            last_bin_idx = max(active_bin_indices)
            
            # Try to move each box from last_bin to any other bin
            # We iterate backwards to remove items safely
            boxes_to_move = list(enumerate(bins[last_bin_idx]))
            for j, (bx, by, bz, bw, bh, bd) in reversed(boxes_to_move):
                if time.time() > deadline - 5:
                    break
                
                # Try to move this box to any other bin
                moved = False
                for other_bin_idx in range(last_bin_idx):
                    if not bins[other_bin_idx]:
                        continue
                    
                    # Check if it fits
                    pos = find_best_position(bins[other_bin_idx], bw, bh, bd)
                    if pos is not None:
                        # Move it
                        # Remove from last_bin
                        # Since we are iterating reversed, we can pop by index if we are careful,
                        # but 'boxes_to_move' is a snapshot. We need to remove the specific item.
                        # Easier: mark as moved and clean up later, or use a set of moved indices.
                        
                        # Let's just do it directly.
                        # Remove from bins[last_bin_idx]
                        # Find the index in the current list
                        # Since we are in a reversed loop over a snapshot, we need to be careful.
                        # Better approach: Try moves in a separate loop and apply them.
                        
                        pass # We'll implement a safer move logic below

            # Safer Move Logic:
            # We will try to move boxes from last_bin to other bins one by one.
            # We keep a list of "moves" to apply.
            
            # Actually, let's just try to move the *smallest* boxes from the last bin first,
            # as they are most likely to fit in gaps in other bins.
            
            last_bin_boxes = bins[last_bin_idx]
            if last_bin_boxes:
                # Sort indices of boxes in last bin by volume (smallest first)
                sorted_indices = sorted(range(len(last_bin_boxes)), 
                                        key=lambda i: last_bin_boxes[i][3]*last_bin_boxes[i][4]*last_bin_boxes[i][5])
                
                for i_idx in sorted_indices:
                    if time.time() > deadline - 5:
                        break
                    
                    if i_idx >= len(bins[last_bin_idx]):
                        continue # Already moved
                    
                    bx, by, bz, bw, bh, bd = bins[last_bin_idx][i_idx]
                    
                    # Try to move to other bins
                    moved = False
                    for other_bin_idx in range(last_bin_idx):
                        if not bins[other_bin_idx]:
                            continue
                        
                        pos = find_best_position(bins[other_bin_idx], bw, bh, bd)
                        if pos is not None:
                            # Move it
                            # Remove from last_bin
                            bins[last_bin_idx].pop(i_idx)
                            # Add to other_bin
                            bins[other_bin_idx].append((pos[0], pos[1], pos[2], bw, bh, bd))
                            # Update placement
                            update_placement(last_bin_idx, (bx, by, bz), (bw, bh, bd), other_bin_idx, pos)
                            moved = True
                            break
                    
                    if not moved:
                        # Try next box
                        continue

            # After moving boxes, check if last_bin is empty.
            # If so, the "last bin" effectively becomes the next highest non-empty bin.
            # This reduces B.
            
            # Strategy 2: If we didn't reduce B (last bin still has boxes), try to increase its fill.
            # We can try to move boxes FROM other bins TO the last bin.
            # This increases the fill fraction of the last bin, improving the 0.1*(1-fill) term.
            # But we must ensure we don't empty the other bin in a way that increases B?
            # No, moving a box from bin A to bin B (where B > A) doesn't change the max index if B is already the max.
            # It just changes the distribution.
            
            active_bin_indices = [i for i in range(len(bins)) if bins[i]]
            if active_bin_indices:
                last_bin_idx = max(active_bin_indices)
                
                # Try to pull boxes from other bins into the last bin
                # We prefer to pull boxes that fit "well" (tightness)
                
                # Get boxes from other bins
                candidate_boxes = []
                for b_idx in range(last_bin_idx):
                    if bins[b_idx]:
                        for j_idx, (bx, by, bz, bw, bh, bd) in enumerate(bins[b_idx]):
                            candidate_boxes.append((b_idx, j_idx, bw, bh, bd, bw*bh*bd))
                
                # Sort by volume descending? Or try to fit large ones first?
                # Let's try to fit boxes that fit tightly.
                
                # To avoid O(N^2) explosion in polish, we limit attempts.
                attempts = 0
                max_attempts = 50
                
                # We need to be careful: moving a box changes the state.
                # We will try moves one by one and apply if valid.
                
                while attempts < max_attempts and time.time() < deadline - 5:
                    attempts += 1
                    best_move = None
                    best_score = float('inf')
                    
                    # Iterate through other bins
                    for b_idx in range(last_bin_idx):
                        if not bins[b_idx]:
                            continue
                        
                        # Snapshot of boxes in this bin
                        for j_idx, (bx, by, bz, bw, bh, bd) in enumerate(list(bins[b_idx])):
                            # Check if it fits in last bin
                            pos = find_best_position(bins[last_bin_idx], bw, bh, bd)
                            if pos is not None:
                                # Calculate a "benefit" score
                                # Benefit: increase in last bin fill
                                # Cost: potential fragmentation increase in last bin?
                                # Let's just accept if it fits.
                                # We prefer moving smaller boxes? Or larger?
                                # Moving larger boxes increases fill fraction more.
                                # Let's prioritize by volume.
                                score = - (bw*bh*bd) # Maximize volume added
                                if score < best_score:
                                    best_score = score
                                    best_move = (b_idx, j_idx, bw, bh, bd, pos)
                    
                    if best_move:
                        b_idx, j_idx, bw, bh, bd, pos = best_move
                        # Perform move
                        # Remove from b_idx
                        # Note: j_idx is from a snapshot, so it's valid if we haven't modified bins[b_idx] in this iteration
                        # But we are modifying bins[last_bin_idx].
                        # We need to remove from bins[b_idx].
                        # The index j_idx might be stale if we removed items from bins[b_idx] in a previous iteration of this while loop.
                        # However, we are only removing from bins[b_idx] and adding to bins[last_bin_idx].
                        # So indices in bins[b_idx] shift down if we remove from the front.
                        # To be safe, we should identify the box by its content or re-scan.
                        
                        # Let's find the box again to get the current index
                        current_idx = None
                        for k, (cx, cy, cz, cw, ch, cd) in enumerate(bins[b_idx]):
                            if cw == bw and ch == bh and cd == bd and cx == bx and cy == by and cz == bz:
                                current_idx = k
                                break
                        if current_idx is None:
                            continue # Box not found (shouldn't happen)
                        
                        bx, by, bz, _, _, _ = bins[b_idx][current_idx]
                        
                        # Remove
                        bins[b_idx].pop(current_idx)
                        # Add to last bin
                        bins[last_bin_idx].append((pos[0], pos[1], pos[2], bw, bh, bd))
                        # Update placement
                        update_placement(b_idx, (bx, by, bz), (bw, bh, bd), last_bin_idx, pos)
                    else:
                        break # No more moves found

    return placements
# EVOLVE-BLOCK-END
