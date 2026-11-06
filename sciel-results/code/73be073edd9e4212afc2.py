import time

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # Each bin is a list of empty spaces: (x, y, z, w, h, d)
    bins = []
    placements = [None] * n
    
    def is_dominated(space, others):
        """Check if space is completely contained in any other space in others."""
        sx, sy, sz, sw, sh, sd = space
        for ox, oy, oz, ow, oh, od in others:
            if sx >= ox and sy >= oy and sz >= oz and \
               sx + sw <= ox + ow and sy + sh <= oy + oh and sz + sd <= oz + od:
                return True
        return False

    def add_space(bin_spaces, space):
        """Add a space to bin_spaces, pruning dominated spaces."""
        if space[3] <= 0 or space[4] <= 0 or space[5] <= 0:
            return
        # Check if this new space is dominated by any existing space
        if is_dominated(space, bin_spaces):
            return
        
        # Remove any existing spaces that are dominated by this new space
        new_spaces = []
        for s in bin_spaces:
            if not is_dominated(s, [space]):
                new_spaces.append(s)
        
        new_spaces.append(space)
        bin_spaces[:] = new_spaces

    # Deadline guard: 140s
    start_time = time.time()
    deadline = start_time + 140
    
    for i in box_indices:
        if time.time() > deadline:
            # Emergency fallback: place remaining boxes in new bins or first fit
            # This ensures validity if the pruning/search explodes (unlikely)
            for j in range(i, n):
                w, h, d = boxes[j]
                # Try first fit in existing bins quickly
                placed = False
                for b_idx, bin_spaces in enumerate(bins):
                    for s_idx, (sx, sy, sz, sw, sh, sd) in enumerate(bin_spaces):
                        if w <= sw and h <= sh and d <= sd:
                            placements[j] = (b_idx, sx, sy, sz, w, h, d)
                            # Update space (simple, no pruning in emergency)
                            del bin_spaces[s_idx]
                            if sx + w < sw:
                                bin_spaces.append((sx + w, sy, sz, sw - (sx + w), sh, sd))
                            if sy + h < sh:
                                bin_spaces.append((sx, sy + h, sz, sw, sh - (sy + h), sd))
                            if sz + d < sd:
                                bin_spaces.append((sx, sy, sz + d, sw, sh, sd - (sz + d)))
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    bins.append([(0, 0, 0, W, H, D)])
                    placements[j] = (len(bins) - 1, 0, 0, 0, w, h, d)
                    # Update space in new bin
                    bin_spaces = bins[-1]
                    sx, sy, sz, sw, sh, sd = bin_spaces[0]
                    del bin_spaces[0]
                    if 0 + w < sw:
                        bin_spaces.append((w, 0, 0, sw - w, sh, sd))
                    if 0 + h < sh:
                        bin_spaces.append((0, h, 0, sw, sh - h, sd))
                    if 0 + d < sd:
                        bin_spaces.append((0, 0, d, sw, sh, sd - d))
            return placements

        w, h, d = boxes[i]
        best_bin_idx = -1
        best_space_idx = -1
        best_x, best_y, best_z = 0, 0, 0
        best_score = float('inf')
        
        # Try to place in existing bins
        for b_idx, bin_spaces in enumerate(bins):
            for s_idx, (sx, sy, sz, sw, sh, sd) in enumerate(bin_spaces):
                if w <= sw and h <= sh and d <= sd:
                    # It fits
                    # Score: minimize x+y+z to keep packing compact at the origin
                    score = sx + sy + sz
                    if score < best_score:
                        best_score = score
                        best_bin_idx = b_idx
                        best_space_idx = s_idx
                        best_x, best_y, best_z = sx, sy, sz
        
        if best_bin_idx == -1:
            # Open a new bin
            bins.append([(0, 0, 0, W, H, D)])
            best_bin_idx = len(bins) - 1
            best_space_idx = 0
            best_x, best_y, best_z = 0, 0, 0
            
        # Record placement
        placements[i] = (best_bin_idx, best_x, best_y, best_z, w, h, d)
        
        # Update bin spaces
        bin_spaces = bins[best_bin_idx]
        sx, sy, sz, sw, sh, sd = bin_spaces[best_space_idx]
        
        # Remove the used space
        del bin_spaces[best_space_idx]
        
        # Add remaining spaces
        # 1. Space to the right (x dimension)
        if best_x + w < sx + sw:
            add_space(bin_spaces, (best_x + w, sy, sz, sx + sw - (best_x + w), sh, sd))
        # 2. Space above (y dimension)
        if best_y + h < sy + sh:
            add_space(bin_spaces, (sx, best_y + h, sz, sw, sy + sh - (best_y + h), sd))
        # 3. Space behind (z dimension)
        if best_z + d < sz + sd:
            add_space(bin_spaces, (sx, sy, best_z + d, sw, sh, sz + sd - (best_z + d)))
            
    return placements
