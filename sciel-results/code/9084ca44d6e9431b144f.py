import time
import itertools

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe
    
    W, H, D = container
    container_vol = W * H * D
    
    # If container is too small for any box, or boxes are empty
    if not boxes:
        return []
    
    # Check if any box is larger than container
    for w, h, d in boxes:
        if w > W or h > H or d > D:
            # This instance is unsolvable as per problem constraints? 
            # The task says "places every box... into numbered bins". 
            # If a box doesn't fit, it's an invalid instance or we must assume they fit.
            # Assuming they fit.
            pass

    # We will try multiple sorting strategies and return the best result
    # Each strategy returns a list of placements: (bin, x, y, z, w, h, d)
    
    def score_placements(placements):
        if not placements:
            return float('inf')
        
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        
        # Find the last bin (largest label)
        if not bins_used:
            return float('inf')
        
        last_bin = max(bins_used)
        
        # Calculate volume in last bin
        last_bin_vol = 0
        for p in placements:
            if p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        
        # Score: B + 0.1 * (1 - last_bin_vol / container_vol)
        # If container_vol is 0, avoid div by zero
        if container_vol == 0:
            return float('inf')
            
        return B + 0.1 * (1 - last_bin_vol / container_vol)

    def try_heuristic(sorted_boxes):
        """
        Attempts to pack the given list of boxes using the Best-Fit Holes heuristic.
        Returns a list of placements or None if failed (shouldn't happen if boxes fit).
        """
        # Holes are tuples: (x, y, z, w, h, d) representing an empty rectangular region
        # We maintain a list of holes for the current bin.
        
        # Initialize first bin with the container as the only hole
        current_holes = [(0, 0, 0, W, H, D)]
        placements = []
        bin_id = 0
        
        # Sort holes initially? Not necessary as we search all.
        
        for i, (w, h, d) in enumerate(sorted_boxes):
            # Check deadline
            if time.time() > deadline:
                return None # Signal timeout
            
            placed = False
            
            # Find the best hole to place the box
            # We want to minimize the "waste". 
            # A common metric is the remaining volume of the hole after placement,
            # or the position (min x, then y, then z) to pack densely.
            # Let's use a combined score: prioritize lower z, then y, then x, 
            # and then smaller remaining volume.
            # Actually, a robust metric is minimizing the max coordinate or 
            # minimizing the sum of coordinates, or just "Best Fit" by volume.
            
            best_hole_idx = -1
            best_score = float('inf')
            best_pos = None # (x, y, z)
            
            for idx, hole in enumerate(current_holes):
                hx, hy, hz, hw, hh, hd = hole
                if w <= hw and h <= hh and d <= hd:
                    # The box fits in this hole
                    # Calculate a score for this placement
                    # We want to pack as low (z), then deep (y), then left (x) as possible
                    # This helps in stacking.
                    # Let's try: score = hz * 10000 + hy * 100 + hx + (hw*hh*hd - w*h*d)*0.001
                    # This prioritizes position over volume waste slightly.
                    
                    # Alternative: Minimize the volume of the largest remaining hole?
                    # Let's stick to a simple heuristic: 
                    # 1. Minimize z (height)
                    # 2. Minimize y (depth)
                    # 3. Minimize x (width)
                    # 4. Minimize remaining volume
                    
                    pos_score = (hz, hy, hx)
                    rem_vol = (hw * hh * hd) - (w * h * d)
                    
                    # Combine into a single comparable tuple
                    # Python compares tuples lexicographically
                    current_score = (hz, hy, hx, rem_vol)
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_hole_idx = idx
                        best_pos = (hx, hy, hz)
            
            if best_hole_idx == -1:
                # No hole in current bin fits the box. Start a new bin.
                bin_id += 1
                current_holes = [(0, 0, 0, W, H, D)]
                # Retry placement in the new bin
                # Since it's a fresh container, it must fit if the box fits the container
                if w > W or h > H or d > D:
                    return None # Invalid box size
                best_hole_idx = 0
                best_pos = (0, 0, 0)
                # Update score for consistency, though it doesn't matter for placement
                best_score = (0, 0, 0, W*H*D - w*h*d)

            # Place the box
            px, py, pz = best_pos
            placements.append((bin_id, px, py, pz, w, h, d))
            
            # Update holes
            # Remove the used hole
            hx, hy, hz, hw, hh, hd = current_holes[best_hole_idx]
            current_holes.pop(best_hole_idx)
            
            # Generate new holes
            # 1. Right of the box (in x)
            if px + w < hx + hw:
                current_holes.append((px + w, hy, hz, hx + hw - (px + w), hh, hd))
            
            # 2. Above the box (in y)
            if py + h < hy + hh:
                current_holes.append((px, py + h, hz, hw, hy + hh - (py + h), hd))
            
            # 3. Behind the box (in z)
            if pz + d < hz + hd:
                current_holes.append((px, hy, hz + d, hw, hh, hz + hd - (pz + d)))
            
            # Optional: Prune holes that are fully contained within others
            # For 40 boxes, the number of holes might grow, but usually stays manageable.
            # Let's add a simple pruning step if the list gets too large
            if len(current_holes) > 200:
                current_holes = prune_holes(current_holes)

        return placements

    def prune_holes(holes):
        """Remove holes that are completely contained within another hole."""
        # This is O(N^2) but N is small after pruning
        # We only keep holes that are not a subset of any other
        # Note: (x1,y1,z1,w1,h1,d1) contains (x2,y2,z2,w2,h2,d2) if
        # x1 <= x2, y1 <= y2, z1 <= z2, x1+w1 >= x2+w2, y1+h1 >= y2+h2, z1+d1 >= z2+d2
        
        n = len(holes)
        is_contained = [False] * n
        
        for i in range(n):
            if is_contained[i]:
                continue
            x1, y1, z1, w1, h1, d1 = holes[i]
            for j in range(n):
                if i == j or is_contained[j]:
                    continue
                x2, y2, z2, w2, h2, d2 = holes[j]
                # Check if i contains j
                if (x1 <= x2 and y1 <= y2 and z1 <= z2 and
                    x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2 and z1 + d1 >= z2 + d2):
                    is_contained[j] = True
                    # If they are identical, we can remove both or one. 
                    # If i is strictly larger, j is removed.
                    # If identical, we should remove one to avoid duplicates.
                    if (x1, y1, z1, w1, h1, d1) == (x2, y2, z2, w2, h2, d2):
                         # Mark i as contained too if we want unique holes, 
                         # but typically we just want to remove duplicates.
                         # Let's mark j as contained. If i was also contained by k, i is gone.
                         pass
        
        # Also remove duplicates
        unique_holes = []
        seen = set()
        for i in range(n):
            if not is_contained[i]:
                h_tuple = holes[i]
                if h_tuple not in seen:
                    seen.add(h_tuple)
                    unique_holes.append(h_tuple)
        
        return unique_holes

    # Define sorting strategies
    # 1. Volume descending
    boxes_sorted_vol = sorted(boxes, key=lambda b: b[0]*b[1]*b[2], reverse=True)
    # 2. Max dimension descending
    boxes_sorted_max = sorted(boxes, key=lambda b: max(b), reverse=True)
    # 3. Height descending
    boxes_sorted_h = sorted(boxes, key=lambda b: b[1], reverse=True)
    # 4. Width descending
    boxes_sorted_w = sorted(boxes, key=lambda b: b[0], reverse=True)
    # 5. Depth descending
    boxes_sorted_d = sorted(boxes, key=lambda b: b[2], reverse=True)
    # 6. Original order
    boxes_sorted_orig = list(boxes)
    
    strategies = [
        boxes_sorted_vol,
        boxes_sorted_max,
        boxes_sorted_h,
        boxes_sorted_w,
        boxes_sorted_d,
        boxes_sorted_orig
    ]
    
    best_placements = None
    best_score = float('inf')
    
    for i, strategy_boxes in enumerate(strategies):
        if time.time() > deadline:
            break
            
        placements = try_heuristic(strategy_boxes)
        
        if placements is not None:
            s = score_placements(placements)
            if s < best_score:
                best_score = s
                best_placements = placements
                
    if best_placements is None:
        # Fallback: Just try the first strategy again or return empty?
        # If timeout, we might have no valid placement. 
        # But 40 boxes is small. try_heuristic is fast.
        # If it returned None due to timeout, we should return whatever we have?
        # The function must return a placement for every box.
        # If we timed out in the middle of a heuristic, we can't return partial.
        # So we must ensure we return a valid full packing.
        # The heuristic is O(N * H) where H is holes. N=40. H < 200.
        # 40 * 200 = 8000 operations per box? No, 40 boxes * 200 holes = 8000 checks.
        # This is extremely fast. < 0.1s per strategy.
        # So timeout is unlikely unless pruning is expensive or logic is wrong.
        
        # If best_placements is still None, something went wrong.
        # Return a trivial (but likely invalid if boxes don't fit) or best effort.
        # Let's try one more time with original order without pruning to see if it works.
        # Or just return an empty list? No, that's invalid.
        
        # Given the speed, this shouldn't happen. But to be safe:
        # Return the first strategy's result if it exists, otherwise empty.
        # Actually, if try_heuristic returns None, it means timeout.
        # We should have kept track of the last complete valid packing.
        
        # Since we assume it's fast, let's just return the best found.
        # If best_placements is None, it means all strategies timed out or failed.
        # This is highly unlikely.
        
        # Fallback: Return empty list (will score 0/invalid) or try to force a placement?
        # Let's assume it works.
        
        return []
        
    return best_placements
