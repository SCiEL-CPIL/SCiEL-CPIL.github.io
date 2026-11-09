"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import heapq

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    deadline = time.time() + 135.0  # Safety margin before 150s kill
    container_vol = W * H * D
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (x[4], max(x[1], x[2], x[3])), reverse=True)
    
    # MEP (Maximal Empty Prism) representation for each bin
    bins_meps = []
    bins_placed = []
    bin_open_order = []  # Track when each bin was first opened (box index)
    
    def split_mep(mep, box):
        """
        Split an MEP by a box placed inside it.
        mep: (mx, my, mz, mw, mh, md) - the empty prism
        box: (bx, by, bz, bw, bh, bd) - the placed box
        Returns a list of up to 6 new non-overlapping sub-prisms.
        """
        mx, my, mz, mw, mh, md = mep
        bx, by, bz, bw, bh, bd = box
        
        new_meps = []
        
        # Left slice (x < bx)
        if bx > mx:
            new_meps.append((mx, my, mz, bx - mx, mh, md))
        
        # Right slice (x > bx + bw)
        if bx + bw < mx + mw:
            new_meps.append((bx + bw, my, mz, mx + mw - bx - bw, mh, md))
        
        # For the region where x overlaps with box, split in y
        x_start = max(mx, bx)
        x_end = min(mx + mw, bx + bw)
        x_width = x_end - x_start
        if x_width > 0:
            # Below slice (y < by)
            if by > my:
                new_meps.append((x_start, my, mz, x_width, by - my, md))
            
            # Above slice (y > by + bh)
            if by + bh < my + mh:
                new_meps.append((x_start, by + bh, mz, x_width, my + mh - by - bh, md))
            
            # For the region where y overlaps with box, split in z
            y_start = max(my, by)
            y_end = min(my + mh, by + bh)
            y_height = y_end - y_start
            if y_height > 0:
                # Front slice (z < bz)
                if bz > mz:
                    new_meps.append((x_start, y_start, mz, x_width, y_height, bz - mz))
                
                # Back slice (z > bz + bd)
                if bz + bd < mz + md:
                    new_meps.append((x_start, y_start, bz + bd, x_width, y_height, mz + md - bz - bd))
        
        # Remove degenerate prisms (zero volume)
        new_meps = [m for m in new_meps if m[3] > 0 and m[4] > 0 and m[5] > 0]
        
        return new_meps
    
    def get_best_position_for_bin(mep_list, bw, bh, bd):
        """
        Find the best position to place a box with dims (bw, bh, bd) in the MEP list.
        Returns (x, y, z, mep_idx) or None.
        Uses Best-Fit strategy: minimize remaining volume of the MEP.
        """
        best_pos = None
        best_remaining = float('inf')
        best_mep_idx = -1
        
        for i, mep in enumerate(mep_list):
            mx, my, mz, mw, mh, md = mep
            
            # Check if box can fit in this MEP
            if bw <= mw and bh <= mh and bd <= md:
                # Try placing at the "bottom-left-front" corner of the MEP
                bx, by, bz = mx, my, mz
                
                # Calculate remaining volume if placed here
                remaining_vol = mw * mh * md - bw * bh * bd
                
                if remaining_vol < best_remaining:
                    best_remaining = remaining_vol
                    best_pos = (bx, by, bz)
                    best_mep_idx = i
        
        if best_pos is not None:
            return (best_pos[0], best_pos[1], best_pos[2], best_mep_idx)
        return None
    
    def try_place_in_bin(bin_idx, bw, bh, bd):
        """Try to place a box in a bin using MEP representation."""
        mep_list = bins_meps[bin_idx]
        
        # Get the best position
        result = get_best_position_for_bin(mep_list, bw, bh, bd)
        
        if result is None:
            return None
        
        bx, by, bz, mep_idx = result
        
        # Place the box by splitting the MEP
        mep = mep_list[mep_idx]
        box = (bx, by, bz, bw, bh, bd)
        new_meps = split_mep(mep, box)
        
        # Replace the old MEP with the new ones
        mep_list[mep_idx:mep_idx+1] = new_meps
        
        # Record the placed box
        bins_placed[bin_idx].append(box)
        return (bx, by, bz)
    
    # Place each box
    placements = [None] * len(boxes)
    
    for orig_idx, w, h, d, vol in indexed_boxes:
        placed = False
        
        # Check time
        if time.time() > deadline:
            # Fallback: place in new bin at origin
            b = len(bins_meps)
            bins_meps.append([(0, 0, 0, W, H, D)])
            bins_placed.append([])
            bin_open_order.append(orig_idx)
            # Place at origin
            box = (0, 0, 0, w, h, d)
            new_meps = split_mep((0, 0, 0, W, H, D), box)
            bins_meps[b] = new_meps
            bins_placed[b].append(box)
            placements[orig_idx] = (b, 0, 0, 0, w, h, d)
            continue
        
        # Try existing bins first, ordered by "fullness" (least remaining volume first)
        bin_order = sorted(range(len(bins_meps)), 
                          key=lambda b: sum(m[3]*m[4]*m[5] for m in bins_meps[b]))
        
        for b in bin_order:
            if time.time() > deadline:
                break
            pos = try_place_in_bin(b, w, h, d)
            if pos is not None:
                placements[orig_idx] = (b, pos[0], pos[1], pos[2], w, h, d)
                placed = True
                break
        
        if not placed:
            # Open a new bin
            b = len(bins_meps)
            bins_meps.append([(0, 0, 0, W, H, D)])
            bins_placed.append([])
            bin_open_order.append(orig_idx)
            
            if time.time() <= deadline:
                pos = try_place_in_bin(b, w, h, d)
                if pos is not None:
                    placements[orig_idx] = (b, pos[0], pos[1], pos[2], w, h, d)
                else:
                    # Fallback: place at origin
                    box = (0, 0, 0, w, h, d)
                    new_meps = split_mep((0, 0, 0, W, H, D), box)
                    bins_meps[b] = new_meps
                    bins_placed[b].append(box)
                    placements[orig_idx] = (b, 0, 0, 0, w, h, d)
            else:
                # Time up, force place at origin
                box = (0, 0, 0, w, h, d)
                new_meps = split_mep((0, 0, 0, W, H, D), box)
                bins_meps[b] = new_meps
                bins_placed[b].append(box)
                placements[orig_idx] = (b, 0, 0, 0, w, h, d)
    
    # Renumber bins to 0..B-1 in first-use order to make the last bin (max label)
    # the one that received its first box latest, which is typically fuller
    if len(bins_meps) > 0:
        # Create mapping from old bin index to new bin index based on open order
        # bin_open_order[i] = orig_idx of the first box placed in bin i
        # We want to sort bins by when they were first opened
        # The bin with the latest first-open should get the highest label
        
        # Sort bin indices by their open order (index in bin_open_order)
        # Actually, bin i was opened at step bin_open_order[i]
        # We want to renumber so that the bin opened last gets the highest number
        
        # Create a list of (open_time, old_bin_idx) and sort by open_time
        bin_info = [(bin_open_order[i], i) for i in range(len(bins_meps))]
        bin_info.sort(key=lambda x: x[0])  # Sort by when first opened
        
        # Now assign new labels: the bin opened first gets 0, next gets 1, etc.
        old_to_new = {}
        for new_idx, (open_time, old_idx) in enumerate(bin_info):
            old_to_new[old_idx] = new_idx
        
        # Apply the renumbering to all placements
        for i in range(len(placements)):
            if placements[i] is not None:
                old_b, x, y, z, w, h, d = placements[i]
                new_b = old_to_new[old_b]
                placements[i] = (new_b, x, y, z, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
