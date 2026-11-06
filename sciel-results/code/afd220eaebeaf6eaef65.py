import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State: list of bins. Each bin is a list of placed boxes (x, y, z, w, h, d)
    # AND a list of free spaces (x, y, z, w, h, d)
    bins = []  # list of lists of placed boxes
    free_spaces = [] # list of lists of free spaces (x, y, z, w, h, d)
    bin_volumes = [] # volume of placed boxes in each bin

    def subtract_space(sp, box):
        """
        Subtract box from space sp.
        Returns a list of new disjoint spaces.
        sp: (sx, sy, sz, sw, sh, sd)
        box: (bx, by, bz, bw, bh, bd)
        """
        sx, sy, sz, sw, sh, sd = sp
        bx, by, bz, bw, bh, bd = box
        
        # Check if box is fully contained in space
        if bx < sx or bx + bw > sx + sw or by < sy or by + bh > sy + sh or bz < sz or bz + bd > sz + sd:
            # This case should not happen if we only select valid spaces
            # But if it does, we can't just subtract. 
            # However, in our logic, we only subtract from spaces that fully contain the box.
            # If the box intersects but doesn't fit, it's an error in selection.
            # For safety, return the original space if it doesn't contain the box?
            # No, the algorithm ensures we pick a space that contains the box.
            return [sp]

        new_spaces = []
        
        # Split along X
        if bx > sx:
            new_spaces.append((sx, sy, sz, bx - sx, sh, sd))
        if bx + bw < sx + sw:
            new_spaces.append((bx + bw, sy, sz, sx + sw - (bx + bw), sh, sd))
        
        # Split along Y
        if by > sy:
            new_spaces.append((sx, sy, sz, sw, by - sy, sd))
        if by + bh < sy + sh:
            new_spaces.append((sx, by + bh, sz, sw, sy + sh - (by + bh), sd))
            
        # Split along Z
        if bz > sz:
            new_spaces.append((sx, sy, sz, sw, sh, bz - sz))
        if bz + bd < sz + sd:
            new_spaces.append((sx, sy, bz + bd, sw, sh, sz + sd - (bz + bd)))
            
        # Wait, the simple 6-way split above creates overlaps in the "middle" regions 
        # if we just append all of them. 
        # Example: Box in center.
        # X-split gives Left and Right strips (full Y, Z).
        # Y-split gives Bottom and Top strips (full X, Z).
        # These overlap in the corners (e.g., Left-Bottom).
        
        # Correct Maximal Space Partitioning (Mezzini et al. style):
        # We need to generate the 6 sub-spaces that are disjoint.
        
        # Let's use the standard algorithm:
        # 1. If box doesn't touch left face (bx > sx), create space (sx, sy, sz, bx-sx, sh, sd)
        # 2. If box doesn't touch right face (bx+bw < sx+sw), create space (bx+bw, sy, sz, sx+sw-(bx+bw), sh, sd)
        # 3. If box doesn't touch bottom face (by > sy), create space (sx, sy, sz, sw, by-sy, sd)
        # 4. If box doesn't touch top face (by+bh < sy+sh), create space (sx, by+bh, sz, sw, sy+sh-(by+bh), sd)
        # 5. If box doesn't touch front face (bz > sz), create space (sx, sy, sz, sw, sh, bz-sz)
        # 6. If box doesn't touch back face (bz+bd < sz+sd), create space (sx, sy, bz+bd, sw, sh, sz+sd-(bz+bd))
        
        # BUT, these 6 spaces overlap each other! 
        # e.g., Space 1 (Left) and Space 3 (Bottom) overlap in the Left-Bottom corner.
        
        # The correct way to maintain disjoint free spaces is more complex.
        # A common simple approach is to NOT maintain disjoint spaces, but rather 
        # a list of "potential" spaces and check for overlap with ALL placed boxes.
        # However, the prompt suggests "Maximal Space Partitioning" which implies disjoint.
        
        # Let's implement a simpler, robust "Free Space" list that is NOT necessarily disjoint 
        # but we filter out spaces that overlap with placed boxes when checking.
        # Actually, a better strategy for Python speed:
        # Maintain a list of free spaces. When placing a box, find a free space that contains it.
        # Remove that free space.
        # Add up to 6 new free spaces (the ones listed above).
        # These new spaces MAY overlap with other existing free spaces.
        # To keep the list manageable and valid, we must ensure that when we check if a new box fits 
        # in a free space, we also check it against all PLACED boxes in that bin.
        # This makes it O(N^2) worst case but N is small (40 boxes).
        
        # Let's refine the strategy:
        # 1. Maintain `free_spaces[bin_idx]` as a list of candidate rectangular regions.
        # 2. When placing a box in a bin:
        #    a. Find a free space in `free_spaces[bin_idx]` that contains the box.
        #    b. If found, remove it.
        #    c. Generate up to 6 new sub-spaces from the intersection.
        #    d. Add these new sub-spaces to `free_spaces[bin_idx]`.
        #    e. IMPORTANT: We must also prune `free_spaces[bin_idx]` to remove any other free spaces 
        #       that are now intersecting the newly placed box? 
        #       Or, simpler: Just keep the list of free spaces, but when testing if a box fits in a free space, 
        #       we ALSO check against all placed boxes. 
        #       This is safe. The free space list is just a heuristic to find candidate positions quickly.
        
        # Let's implement this "Candidate Free Space" approach.
        
        return new_spaces # This function is not used in the final logic below, kept for reference

    def place_box(w, h, d):
        # 1. Try to place in existing bins (Best Fit)
        best_bin_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for i in range(len(bins)):
            # Find best position in this bin
            # Iterate through free spaces
            valid_pos = None
            min_residual_vol = float('inf')
            
            # We need to check if the box fits in any free space.
            # If a free space contains the box, it's a candidate.
            # We pick the one with minimum residual volume (Best Fit).
            # Residual volume = free_space_vol - box_vol.
            # Since box_vol is constant for this step, minimize free_space_vol.
            
            candidate_indices = []
            
            for idx, (fx, fy, fz, fw, fh, fd) in enumerate(free_spaces[i]):
                if fw >= w and fh >= h and fd >= d:
                    # Check if the box can actually be placed at (fx, fy, fz) without overlapping
                    # other placed boxes. 
                    # Wait, if the free space was derived from previous splits, it should be empty.
                    # But since we don't strictly prune overlaps between free spaces, 
                    # two free spaces might overlap, and one might contain a placed box?
                    # No, free spaces are generated from empty space. 
                    # However, if we don't prune, a free space might overlap with a placed box 
                    # if the placed box was placed in a DIFFERENT free space that overlapped with this one.
                    # So we MUST check against placed boxes.
                    
                    # Check overlap with placed boxes in bin i
                    overlap = False
                    for (ox, oy, oz, ow, oh, od) in bins[i]:
                        if not (fx + w <= ox or ox + ow <= fx or
                                fy + h <= oy or oy + oh <= fy or
                                fz + d <= oz or oz + od <= fz):
                            overlap = True
                            break
                    if not overlap:
                        vol = fw * fh * fd
                        if vol < min_residual_vol:
                            min_residual_vol = vol
                            valid_pos = (fx, fy, fz)
                            best_candidate_idx = idx
            
            if valid_pos is not None:
                # We found a valid position in bin i
                # Score: lower is better. Use the volume of the free space used.
                if min_residual_vol < best_score:
                    best_score = min_residual_vol
                    best_bin_idx = i
                    best_pos = valid_pos
                    # We also need to remember which free space index to remove
                    # But we might find a better one in another bin.
                    # So we just store the position.
        
        if best_bin_idx != -1:
            b = best_bin_idx
            x, y, z = best_pos
            
            # Place the box
            bins[b].append((x, y, z, w, h, d))
            bin_volumes[b] += w * h * d
            
            # Update free spaces for bin b
            # Find the free space that contains (x,y,z) and has size >= (w,h,d)
            # We need to remove the specific free space we used.
            # Since we didn't store the index, we'll find it again.
            removed = False
            new_spaces = []
            
            for idx, (fx, fy, fz, fw, fh, fd) in enumerate(free_spaces[b]):
                if fx == x and fy == y and fz == z and fw >= w and fh >= h and fd >= d:
                    # This is the space we used.
                    # Generate sub-spaces
                    if x > fx:
                        new_spaces.append((fx, fy, fz, x - fx, fh, fd))
                    if x + w < fx + fw:
                        new_spaces.append((x + w, fy, fz, fx + fw - (x + w), fh, fd))
                    if y > fy:
                        new_spaces.append((fx, fy, fz, fw, y - fy, fd))
                    if y + h < fy + fh:
                        new_spaces.append((fx, y + h, fz, fw, fy + fh - (y + h), fd))
                    if z > fz:
                        new_spaces.append((fx, fy, fz, fw, fh, z - fz))
                    if z + d < fz + fd:
                        new_spaces.append((fx, fy, z + d, fw, fh, fz + fd - (z + d)))
                    
                    removed = True
                    break
            
            if removed:
                # Remove the used space from the list
                # We need to remove the specific one. 
                # Since we broke the loop, we don't have the index. 
                # Let's re-iterate to remove it properly.
                for idx, (fx, fy, fz, fw, fh, fd) in enumerate(free_spaces[b]):
                    if fx == x and fy == y and fz == z and fw >= w and fh >= h and fd >= d:
                        free_spaces[b].pop(idx)
                        break
                
                # Add new spaces
                free_spaces[b].extend(new_spaces)
            
            return b
        
        # 2. If no existing bin works, open a new bin
        new_bin_idx = len(bins)
        bins.append([(0, 0, 0, w, h, d)])
        # The remaining free space in the new bin
        # The box is at (0,0,0). 
        # Free space is the container minus the box.
        # We can represent this as a list of up to 6 spaces.
        new_free = []
        if w < W: new_free.append((w, 0, 0, W - w, H, D))
        if h < H: new_free.append((0, h, 0, W, H - h, D))
        if d < D: new_free.append((0, 0, d, W, H, D - d))
        # Note: These 3 spaces overlap! 
        # e.g., (w,0,0) and (0,h,0) do not overlap in x, but (w,0,0) and (0,0,d) do not overlap in z.
        # Wait, (w, 0, 0, W-w, H, D) is the space to the right.
        # (0, h, 0, W, H-h, D) is the space above.
        # They overlap in the region (w, h, 0, W-w, H-h, D).
        # This is fine for our "Candidate Free Space" approach because we check against placed boxes.
        
        free_spaces.append(new_free)
        bin_volumes.append(w * h * d)
        return new_bin_idx

    # Initialize
    # No bins initially
    
    # Process boxes
    placements = [None] * n
    
    for idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes in new bins
            break
            
        b = place_box(w, h, d)
        pos = bins[b][-1]
        placements[idx] = (b, pos[0], pos[1], pos[2], w, h, d)
        
    # Handle any unplaced boxes if we broke out early
    if time.time() > deadline:
        # Find unplaced
        unplaced_indices = [i for i in range(n) if placements[i] is None]
        # For each unplaced, assign to a new bin at (0,0,0)
        # This guarantees validity but poor score.
        for i in unplaced_indices:
            w, h, d = boxes[i]
            b = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            # Free space for this new bin
            new_free = []
            if w < W: new_free.append((w, 0, 0, W - w, H, D))
            if h < H: new_free.append((0, h, 0, W, H - h, D))
            if d < D: new_free.append((0, 0, d, W, H, D - d))
            free_spaces.append(new_free)
            bin_volumes.append(w * h * d)
            placements[i] = (b, 0, 0, 0, w, h, d)
            
    return placements
