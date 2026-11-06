"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe within 120s
    
    # Sort indices by volume descending for greedy initialization
    vol = [w * h * d for w, h, d in boxes]
    indices = list(range(n))
    indices.sort(key=lambda i: vol[i], reverse=True)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return False
        return True

    def find_spot(bin_boxes, w, h, d):
        # Generate candidate positions based on "corner" points of existing boxes
        # This is a standard heuristic for 3D packing to find valid spots efficiently
        candidates = set()
        candidates.add((0, 0, 0))
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Points on the "top" face of existing boxes
            candidates.add((bx, by, bz + bd))
            # Points on the "side" faces (x-direction)
            candidates.add((bx + bw, by, bz))
            # Points on the "side" faces (y-direction)
            candidates.add((bx, by + bh, bz))
            
            # Combinations of these offsets often fill gaps
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            
            # Also try the "far" corners if they might be valid (less common but helps with weird shapes)
            candidates.add((bx + bw, by + bh, bz + bd))

        # Sort candidates to prefer "lowest" position (minimize z, then y, then x)
        # This tends to pack items towards the origin, which is generally good for stability and filling
        # But for pure bin count reduction, any valid spot is okay. 
        # However, compactness helps future items fit.
        # Let's sort by (z, y, x) to fill bottom-up, left-to-right, front-to-back.
        valid_candidates = []
        for cx, cy, cz in candidates:
            if cx < W and cy < H and cz < D:
                valid_candidates.append((cx, cy, cz))
        
        valid_candidates.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        
        for cx, cy, cz in valid_candidates:
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        
        return None

    def initialize():
        """Greedy First-Fit-Decreasing initialization."""
        bins = [] # List of lists of (x, y, z, w, h, d)
        placements = [None] * n
        
        for i in indices:
            w, h, d = boxes[i]
            placed = False
            
            for b_idx in range(len(bins)):
                spot = find_spot(bins[b_idx], w, h, d)
                if spot is not None:
                    bx, by, bz = spot
                    bins[b_idx].append((bx, by, bz, w, h, d))
                    placements[i] = (b_idx, bx, by, bz, w, h, d)
                    placed = True
                    break
            
            if not placed:
                # Open new bin
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[i] = (b_idx, 0, 0, 0, w, h, d)
                
        return bins, placements

    def get_score(bins, placements):
        """Calculate the task score. Lower is better."""
        if not bins:
            return 0.0
        num_bins = len(bins)
        # Find the last bin (highest index label used)
        # In our representation, bins are 0..N-1. 
        # The problem says "largest label defines the last bin". 
        # Since we use consecutive labels 0..N-1, the last bin is the one with the highest index.
        last_bin_idx = num_bins - 1
        last_bin = bins[last_bin_idx]
        
        last_bin_vol = sum(w * h * d for _, _, _, w, h, d in last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            fill_ratio = 1.0
        else:
            fill_ratio = last_bin_vol / container_vol
            
        score = num_bins + 0.1 * (1 - fill_ratio)
        return score

    def local_search(bins, placements, time_limit):
        """
        Perform local search to reduce bin count.
        1. Try to move a box from a non-last bin to the last bin? No, that increases last bin volume, which is good for the 0.1 term, but doesn't reduce B.
        Actually, reducing B is the primary goal.
        Strategy: 
        - Identify the "last" bin (highest index).
        - Try to move boxes FROM the last bin TO other bins to empty it completely.
        - If a bin is emptied, remove it. This reduces B by 1.
        - If we can't empty the last bin, try to move boxes from other bins to the last bin? 
          No, that just shifts volume. 
          The key to reducing B is consolidating boxes.
        
        Improved Strategy:
        1. While time remains:
           a. Pick the bin with the highest index (last_bin).
           b. For each box in last_bin (largest first? smallest first?):
              - Try to relocate it to any other bin (0 to last_bin-1).
              - If it fits, move it.
           c. If last_bin becomes empty, remove it.
           d. If no box from last_bin can be moved, break? 
              Or try a "swap" heuristic? 
              If we can't move a box out, maybe we can swap a box from another bin into the last bin 
              and move the box we wanted to move into that other bin?
              This is complex. Let's stick to Relocation first.
              
        Additional Heuristic: 
        If we are stuck, try to "compact" the last bin? No, we want to EMPTY it.
        
        What if the last bin has a very large box that doesn't fit anywhere else?
        Then we are stuck with that bin. We should focus on making sure the OTHER bins are full 
        so that future boxes (if we were doing online) or just the current set 
        are packed such that the last bin is as "full" as possible IF we can't reduce B.
        
        But the score is B + 0.1*(1-fill). 
        Reducing B by 1 saves 1.0.
        Increasing fill of last bin from 0 to 1 saves 0.1.
        So reducing B is 10x more important.
        
        Therefore, the primary goal of local search is to eliminate bins.
        
        Algorithm:
        1. Sort bins by index descending.
        2. Let target_bin be the highest index bin.
        3. For each box in target_bin:
             Try to move it to any bin j < target_bin.
             If it fits, move it.
        4. If target_bin is empty, remove it.
        5. Repeat.
        
        If no box in target_bin can be moved, we are stuck with B bins.
        In this case, we should try to MAXIMIZE the volume of the last bin.
        How? 
        - Take boxes from OTHER bins and move them INTO the last bin?
        - This doesn't change B, but increases the fill of the last bin.
        - We should prioritize moving boxes that fit "tightly" or just any box that fits.
        - Actually, moving a box from bin i to last_bin increases last_bin fill.
        - We should do this for ALL boxes in ALL other bins if they fit in the last bin.
        - But wait, if we move a box from bin i to last_bin, bin i might become empty? 
          If bin i becomes empty, B decreases! That's great.
          So the "move to last bin" strategy is also a way to empty other bins.
        
        Refined Local Search:
        Loop:
          1. Try to empty the highest-index bin (last_bin) by moving its contents to lower-index bins.
          2. If successful (bin emptied), remove it and continue.
          3. If not successful (cannot empty last_bin):
             a. Try to move boxes from lower-index bins INTO the last_bin.
             b. This increases the fill of the last_bin (good for score).
             c. If a lower-index bin becomes empty, remove it (reduces B, even better).
             d. If no moves are possible, stop.
        """
        best_score = get_score(bins, placements)
        
        while time.time() < time_limit:
            moved_any = False
            last_bin_idx = len(bins) - 1
            
            # Phase 1: Try to empty the last bin
            if last_bin_idx > 0:
                # Iterate boxes in last bin
                # We need to be careful: if we remove a box, indices change.
                # Let's copy the list of boxes to try to move.
                boxes_to_move = list(bins[last_bin_idx])
                
                for box_data in boxes_to_move:
                    x, y, z, w, h, d = box_data
                    # Find the original index of this box?
                    # We need to update placements.
                    # Let's identify which box index this is.
                    # We can store box indices in the bins structure or search placements.
                    # Searching placements is O(N). N=40, so it's fine.
                    box_idx = -1
                    for i in range(n):
                        if placements[i] is not None:
                            p_bin, px, py, pz, pw, ph, pd = placements[i]
                            if p_bin == last_bin_idx and px == x and py == y and pz == z and pw == w and ph == h and pd == d:
                                box_idx = i
                                break
                    if box_idx == -1:
                        continue # Should not happen, but safety check
                    
                    # Try to move box_idx to any other bin
                    moved = False
                    for b_idx in range(last_bin_idx):
                        # Check if it fits in b_idx
                        # Temporarily remove from last_bin to check? No, can_place checks against existing.
                        # We need to check against bins[b_idx] without the box (which isn't there).
                        # So just check against bins[b_idx].
                        spot = find_spot(bins[b_idx], w, h, d)
                        if spot is not None:
                            # Move it!
                            # 1. Remove from last_bin
                            bins[last_bin_idx].remove(box_data)
                            # 2. Add to b_idx
                            bx, by, bz = spot
                            bins[b_idx].append((bx, by, bz, w, h, d))
                            # 3. Update placement
                            placements[box_idx] = (b_idx, bx, by, bz, w, h, d)
                            moved = True
                            moved_any = True
                            break
                    if moved:
                        continue # Next box in last_bin (list was copied, so safe)
                    
                    # If we couldn't move this box, we might be stuck with the last bin.
                    # But we should still try other boxes in the last bin.
            
            # Check if last bin is empty
            if last_bin_idx >= 0 and len(bins[last_bin_idx]) == 0:
                # Remove the bin
                bins.pop(last_bin_idx)
                # We don't need to re-index placements because the bin index just disappeared.
                # Wait, if we remove bin k, all bins j > k shift down.
                # But we only remove the LAST bin, so no other bins shift.
                moved_any = True
                continue # Loop again to check new last bin
            
            # Phase 2: If we couldn't empty the last bin, try to fill it
            # or empty other bins by moving boxes INTO the last bin.
            if last_bin_idx > 0:
                # Try to move boxes from other bins to the last bin
                # We want to maximize volume in last bin.
                # Let's iterate through all boxes in other bins.
                # To avoid O(N^2) overhead in Python loops, we can just iterate.
                
                # Get a list of (box_idx, bin_idx) for all boxes NOT in last bin
                candidates = []
                for i in range(n):
                    if placements[i] is not None:
                        b_idx, _, _, _, _, _, _ = placements[i]
                        if b_idx != last_bin_idx:
                            candidates.append(i)
                
                # Sort by volume descending? Or random?
                # Moving larger boxes first might fill the bin faster.
                candidates.sort(key=lambda i: vol[i], reverse=True)
                
                for box_idx in candidates:
                    b_idx, x, y, z, w, h, d = placements[box_idx]
                    # Check if it fits in last_bin
                    spot = find_spot(bins[last_bin_idx], w, h, d)
                    if spot is not None:
                        # Move it
                        # Remove from b_idx
                        box_data = (x, y, z, w, h, d)
                        # We need to find and remove the exact tuple.
                        # Tuples are immutable, so we can just remove it.
                        try:
                            bins[b_idx].remove(box_data)
                        except ValueError:
                            # Should not happen
                            continue
                        
                        # Add to last_bin
                        bx, by, bz = spot
                        bins[last_bin_idx].append((bx, by, bz, w, h, d))
                        
                        # Update placement
                        placements[box_idx] = (last_bin_idx, bx, by, bz, w, h, d)
                        moved_any = True
                        
                        # Check if b_idx became empty
                        if len(bins[b_idx]) == 0:
                            # Remove bin b_idx
                            # This shifts all bins j > b_idx down by 1.
                            # We must update placements for all boxes in bins j > b_idx.
                            # This is expensive. 
                            # Alternative: Keep a list of active bins? 
                            # Or just re-index everything?
                            
                            # Let's re-index. N is small (40).
                            bins.pop(b_idx)
                            # Update placements
                            for i in range(n):
                                if placements[i] is not None:
                                    pb, px, py, pz, pw, ph, pd = placements[i]
                                    if pb > b_idx:
                                        placements[i] = (pb - 1, px, py, pz, pw, ph, pd)
                                    elif pb == b_idx:
                                        # This box was just moved, so it should be in last_bin (now shifted)
                                        # Wait, if we removed b_idx, and the box was in b_idx, 
                                        # it's no longer in b_idx. It's in last_bin.
                                        # But we already updated placements[box_idx] to last_bin.
                                        # If last_bin > b_idx, its index changed.
                                        # Let's handle this carefully.
                                        pass
                            
                            # Actually, this re-indexing logic is prone to bugs.
                            # Let's avoid removing bins during the "fill" phase.
                            # If a bin becomes empty, we can clean it up at the end.
                            # Or, we can just leave empty bins in the list?
                            # The score calculation uses `len(bins)`. 
                            # If we have empty bins, `len(bins)` is high, which is bad.
                            # So we MUST remove empty bins.
                            
                            # Let's do a full cleanup and re-indexing step if any bin becomes empty.
                            # Since this is a rare event (only if we empty a whole bin), the cost is acceptable.
                            pass
                            
                        # If we moved a box, we should probably break and restart the loop 
                        # because the state has changed significantly.
                        # Or continue to try to move more boxes into the last bin?
                        # Moving more boxes into the last bin increases its fill, which is good.
                        # Let's continue.
            
            if not moved_any:
                break
            
            # Calculate current score to see if we improved
            current_score = get_score(bins, placements)
            if current_score < best_score:
                best_score = current_score
            else:
                # If we didn't improve, we might be cycling.
                # Add a simple check: if no score improvement in last K iterations, break?
                # For now, just break if no moves were made.
                pass

        # Final cleanup: remove any empty bins that might have been created
        new_bins = []
        bin_map = {} # old_idx -> new_idx
        for i, b in enumerate(bins):
            if len(b) > 0:
                new_bins.append(b)
                bin_map[i] = len(new_bins) - 1
        
        # Update placements
        for i in range(n):
            if placements[i] is not None:
                old_b, x, y, z, w, h, d = placements[i]
                if old_b in bin_map:
                    new_b = bin_map[old_b]
                    placements[i] = (new_b, x, y, z, w, h, d)
                else:
                    # Box is in an empty bin? Should not happen.
                    pass
        
        bins = new_bins
        return bins, placements

    # 1. Initialize
    bins, placements = initialize()
    
    # 2. Local Search
    # Use a time limit for local search
    ls_deadline = start_time + 110 # Leave 5s for final processing
    bins, placements = local_search(bins, placements, ls_deadline)
    
    # 3. Final Polish: Try to maximize fill of the last bin one last time?
    # The local search already does this.
    
    # Ensure all placements are valid and non-None
    # If any placement is None, it means we failed to place a box.
    # This shouldn't happen with the greedy init, but just in case.
    for i in range(n):
        if placements[i] is None:
            # Fallback: put in a new bin
            w, h, d = boxes[i]
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[i] = (b_idx, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
