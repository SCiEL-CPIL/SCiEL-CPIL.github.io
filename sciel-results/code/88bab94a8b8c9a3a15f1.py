import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # State: List of bins. Each bin is a list of free spaces (rectangular prisms).
    # Free space: (x, y, z, w, h, d)
    # Initially, each bin has one free space covering the whole container.
    bins = []
    placements = [None] * n
    
    def new_bin():
        return [(0, 0, 0, W, H, D)]
    
    bins.append(new_bin())
    
    # Helper: Check if a box fits in a specific free space
    def fits_in_space(box, space):
        _, _, _, bw, bh, bd = box
        _, _, _, sw, sh, sd = space
        return bw <= sw and bh <= sh and bd <= sd

    # Helper: Split a free space into up to 6 new free spaces after placing a box
    # Box at (px, py, pz) with size (bw, bh, bd) inside space (sx, sy, sz, sw, sh, sd)
    def split_space(space, box):
        sx, sy, sz, sw, sh, sd = space
        px, py, pz, bw, bh, bd = box
        
        new_spaces = []
        
        # 1. Space above the box (if any)
        if py + bh < sy + sh:
            new_spaces.append((sx, py + bh, sz, sw, sy + sh - (py + bh), sd))
        
        # 2. Space to the right of the box (if any)
        if px + bw < sx + sw:
            new_spaces.append((px + bw, sy, sz, sx + sw - (px + bw), sh, sd))
        
        # 3. Space in front of the box (if any)
        if pz + bd < sz + sd:
            new_spaces.append((sx, sy, pz + bd, sw, sh, sz + sd - (pz + bd)))
            
        # 4. Space above and to the right (corner)
        # This is implicitly covered by the combination of splits if we are careful, 
        # but standard orthogonal partitioning creates disjoint regions.
        # Let's use a simpler, robust split:
        # We place the box at (px, py, pz).
        # The remaining free space is the union of disjoint boxes.
        
        # Let's restart the split logic to ensure disjointness and completeness.
        # The box occupies [px, px+bw] x [py, py+bh] x [pz, pz+bd].
        
        # Region 1: x < px (Left)
        if px > sx:
            new_spaces.append((sx, sy, sz, px - sx, sh, sd))
            
        # Region 2: x > px+bw (Right)
        if px + bw < sx + sw:
            new_spaces.append((px + bw, sy, sz, sx + sw - (px + bw), sh, sd))
            
        # Now consider the slice where sx+px <= x < sx+px+bw (the box's x-range within the space)
        # Actually, the box is placed at px, which is >= sx.
        # The slice is [px, px+bw] in x.
        
        # Within this x-slice, we have y and z to consider.
        # The box occupies [py, py+bh] in y and [pz, pz+bd] in z.
        
        # Sub-region 2a: y < py (Bottom) within the x-slice
        if py > sy:
            new_spaces.append((px, sy, sz, bw, py - sy, sd))
            
        # Sub-region 2b: y > py+bh (Top) within the x-slice
        if py + bh < sy + sh:
            new_spaces.append((px, py + bh, sz, bw, sy + sh - (py + bh), sd))
            
        # Now consider the slice where py <= y < py+bh (the box's y-range)
        # And x is in [px, px+bw].
        # We have z to consider.
        
        # Sub-region 2b-i: z < pz (Front) within x-y slice
        if pz > sz:
            new_spaces.append((px, py, sz, bw, bh, pz - sz))
            
        # Sub-region 2b-ii: z > pz+bd (Back) within x-y slice
        if pz + bd < sz + sd:
            new_spaces.append((px, py, pz + bd, bw, bh, sz + sd - (pz + bd)))
            
        return new_spaces

    # Helper: Merge overlapping free spaces? 
    # Merging is complex in 3D. We will just keep a list of disjoint free spaces.
    # Over time, the list might grow, but for 40 boxes, it's manageable.
    # We can prune spaces that are too small for any remaining box? 
    # Or just keep them.
    
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Fallback: Place remaining boxes in a new bin using simple greedy
            # to ensure validity and finish in time.
            # This is a safety net.
            # Create a new bin for the rest? Or just place them in bin 0 with no overlap check?
            # No, must not overlap. 
            # Simplest valid fallback: put each remaining box in its own new bin.
            # This maximizes B but guarantees validity.
            for rem_idx, rem_w, rem_h, rem_d in indexed_boxes[idx:]:
                bins.append(new_bin())
                b_idx = len(bins) - 1
                placements[rem_idx] = (b_idx, 0, 0, 0, rem_w, rem_h, rem_d)
                # Update the bin's free space
                bins[b_idx] = split_space(bins[b_idx][0], (0, 0, 0, rem_w, rem_h, rem_d))
            break
            
        best_bin_idx = -1
        best_space_idx = -1
        best_score = float('inf')
        
        # For each bin, find the best free space
        for b_idx, free_spaces in enumerate(bins):
            # If the bin is empty (only one full space), opening cost is high if we haven't used it before?
            # Actually, if b_idx > 0, we are opening a new bin.
            # If b_idx == 0, we are using the first bin.
            # Let's define a penalty for opening a new bin.
            # If b_idx == len(bins) - 1 and b_idx > 0, it's a new bin.
            # But we are iterating existing bins.
            # If no bin fits, we open a new one.
            
            for s_idx, space in enumerate(free_spaces):
                # Check if box fits in this space
                if not fits_in_space((0, 0, 0, w, h, d), space):
                    continue
                
                # Calculate placement score
                # We want to minimize the "waste" or "fragmentation".
                # A good heuristic is to place the box in the corner of the free space that
                # minimizes the sum of coordinates, or minimizes the remaining volume of the space?
                # Let's try: Place at the "lowest" corner of the free space.
                sx, sy, sz, sw, sh, sd = space
                px, py, pz = sx, sy, sz
                
                # Score: 
                # 1. Prefer bins that are already open (lower b_idx? or just prefer existing bins)
                # 2. Prefer placements that leave larger free spaces (less fragmentation)
                #    or fill the space more tightly.
                
                # Let's use a simple score: 
                # Score = b_idx * 1000 + (sx + sy + sz) 
                # This strongly prefers earlier bins. 
                # Within a bin, it prefers lower coordinates.
                
                # However, we might want to consider the volume of the free space.
                # Smaller free spaces are better to fill? 
                # Let's try: Score = b_idx * 1000 + (sx + sy + sz) + (sw*sh*sd)/10000
                
                score = b_idx * 1000 + (sx + sy + sz)
                
                if score < best_score:
                    best_score = score
                    best_bin_idx = b_idx
                    best_space_idx = s_idx
        
        if best_bin_idx == -1:
            # No existing bin can fit this box.
            # Open a new bin.
            bins.append(new_bin())
            best_bin_idx = len(bins) - 1
            best_space_idx = 0
            px, py, pz = 0, 0, 0
        else:
            # Place in the selected bin and space
            space = bins[best_bin_idx][best_space_idx]
            px, py, pz = space[0], space[1], space[2]
        
        # Record placement
        placements[idx] = (best_bin_idx, px, py, pz, w, h, d)
        
        # Update bin state: Replace the used free space with the new split spaces
        old_space = bins[best_bin_idx][best_space_idx]
        new_spaces = split_space(old_space, (px, py, pz, w, h, d))
        
        # Remove the old space and add the new ones
        bins[best_bin_idx].pop(best_space_idx)
        bins[best_bin_idx].extend(new_spaces)
        
        # Optional: Merge adjacent free spaces to keep the list small?
        # This is complex. For now, we just keep the list.
        # We can prune spaces that are too small to fit any remaining box.
        # But we don't know the remaining boxes' sizes easily without lookahead.
        # Let's just keep them.
        
    return placements
