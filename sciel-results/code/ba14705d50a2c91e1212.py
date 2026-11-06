import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])

    # Free-space representation for each bin
    # Each space is (x, y, z, w, h, d)
    # We maintain a list of free spaces per bin

    def create_initial_space(W, H, D):
        return [(0, 0, 0, W, H, D)]

    def split_space(space, box_x, box_y, box_z, box_w, box_h, box_d):
        """Split a free space after placing a box at (box_x, box_y, box_z) with dims (box_w, box_h, box_d).
        Returns a list of new free spaces."""
        sx, sy, sz, sw, sh, sd = space
        bx, by, bz, bw, bh, bd = box_x, box_y, box_z, box_w, box_h, box_d
        
        # Check if box fits in this space
        if bx < sx or by < sy or bz < sz:
            return []
        if bx + bw > sx + sw or by + bh > sy + sh or bz + bd > sz + sd:
            return []
        
        new_spaces = []
        # Split along x
        if bx > sx:
            new_spaces.append((sx, sy, sz, bx - sx, sh, sd))
        if bx + bw < sx + sw:
            new_spaces.append((bx + bw, sy, sz, sx + sw - (bx + bw), sh, sd))
        
        # Split along y
        if by > sy:
            new_spaces.append((bx, sy, bz, bw, by - sy, bd))
        if by + bh < sy + sh:
            new_spaces.append((bx, by + bh, bz, bw, sy + sh - (by + bh), bd))
        
        # Split along z
        if bz > sz:
            new_spaces.append((bx, by, sz, bw, bh, bz - sz))
        if bz + bd < sz + sd:
            new_spaces.append((bx, by, bz + bd, bw, bh, sz + sd - (bz + bd)))
        
        return new_spaces

    def find_best_position(spaces, box_w, box_h, box_d):
        """Find the best position for a box in the given free spaces.
        Returns (x, y, z, space_index) or None if no valid position."""
        best_pos = None
        best_score = None
        best_space_idx = -1
        
        for si, (sx, sy, sz, sw, sh, sd) in enumerate(spaces):
            # Check if box can fit in this space
            if box_w > sw or box_h > sh or box_d > sd:
                continue
            
            # Try to place at the "bottom-left-back" corner of this space
            # This is the corner closest to origin
            x, y, z = sx, sy, sz
            
            # Score: minimize distance to origin (tight packing)
            score = x + y + z
            
            if best_score is None or score < best_score:
                best_score = score
                best_pos = (x, y, z)
                best_space_idx = si
        
        if best_pos is None:
            return None, -1
        
        return best_pos, best_space_idx

    def place_box_in_bin(spaces, box_w, box_h, box_d):
        """Try to place a box in a bin with given free spaces.
        Returns (success, new_spaces, position) where position is (x,y,z)."""
        pos, space_idx = find_best_position(spaces, box_w, box_h, box_d)
        if pos is None:
            return False, spaces, None
        
        x, y, z = pos
        sx, sy, sz, sw, sh, sd = spaces[space_idx]
        
        # Remove the original space
        new_spaces = spaces[:space_idx] + spaces[space_idx+1:]
        
        # Split the space
        new_free = split_space((sx, sy, sz, sw, sh, sd), x, y, z, box_w, box_h, box_d)
        new_spaces.extend(new_free)
        
        return True, new_spaces, (x, y, z)

    # Initialize bins
    bins = []  # Each bin is a list of free spaces
    placements = []  # (bin_idx, x, y, z, w, h, d)
    
    # Sort order: volume descending
    order = [i for i, (_, _, _, _, vol) in enumerate(indexed_boxes)]
    
    for box_idx in order:
        i, w, h, d, vol = indexed_boxes[box_idx]
        box_dims = (w, h, d)
        
        placed = False
        best_bin = -1
        best_new_spaces = None
        best_pos = None
        best_score = None
        
        # Try each existing bin
        for bi in range(len(bins)):
            if time.time() > deadline:
                # Emergency: place in first bin that fits, or create new
                break
            success, new_spaces, pos = place_box_in_bin(bins[bi], w, h, d)
            if success:
                x, y, z = pos
                score = x + y + z
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin = bi
                    best_new_spaces = new_spaces
                    best_pos = pos
        
        if not placed and best_bin >= 0:
            bins[best_bin] = best_new_spaces
            placements.append((best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placed = True
        
        if not placed:
            # Create a new bin
            new_bin_spaces = create_initial_space(W, H, D)
            success, new_bin_spaces, pos = place_box_in_bin(new_bin_spaces, w, h, d)
            if success:
                bins.append(new_bin_spaces)
                placements.append((len(bins) - 1, pos[0], pos[1], pos[2], w, h, d))
            else:
                # Box doesn't fit in an empty container - shouldn't happen, but handle gracefully
                # Place at origin in a new bin (will overlap, but this is a fallback)
                bins.append(new_bin_spaces)
                placements.append((len(bins) - 1, 0, 0, 0, w, h, d))
    
    # Map back to original indices
    # placements currently has (bin, x, y, z, w, h, d) in sorted order
    # We need to return in original box order
    result = [None] * n
    for pi, (bi, x, y, z, w, h, d) in enumerate(placements):
        orig_idx = indexed_boxes[pi][0]
        result[orig_idx] = (bi, x, y, z, w, h, d)
    
    return result
