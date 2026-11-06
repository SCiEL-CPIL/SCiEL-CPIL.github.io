import time
import sys
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

def pack(boxes, container):
    """
    3D Bin Packing using Maximal Space Partitioning (MSP) with best-fit heuristic.
    Uses a priority queue of free spaces per bin, splitting on placement.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, then by max dimension for tie-breaking
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[1]*x[2]*x[3], -max(x[1], x[2], x[3])))
    
    # Each bin has a list of free spaces: (min_x, min_y, min_z, max_x, max_y, max_z)
    # We use a heap for each bin to get the "best" space (smallest corner)
    bins = []  # list of heaps, each heap contains (x, y, z, space_tuple)
    
    def new_bin():
        # Initial free space is the entire container
        space = (0, 0, 0, W, H, D)
        heap = [(0, 0, 0, space)]
        bins.append(heap)
        return len(bins) - 1
    
    def split_space(space, box):
        """
        Split a free space around a placed box.
        box = (x, y, z, w, h, d) where x,y,z is the position
        space = (min_x, min_y, min_z, max_x, max_y, max_z)
        Returns list of new non-overlapping free spaces.
        """
        sx, sy, sz, ex, ey, ez = space
        bx, by, bz, bw, bh, bd = box
        
        new_spaces = []
        
        # Left split (x < bx)
        if bx > sx:
            new_spaces.append((sx, sy, sz, bx, ey, ez))
        # Right split (x > bx+bw)
        if bx + bw < ex:
            new_spaces.append((bx + bw, sy, sz, ex, ey, ez))
        # Bottom split (y < by, within x range of box)
        if by > sy:
            new_spaces.append((bx, sy, sz, bx + bw, by, ez))
        # Top split (y > by+bh, within x range of box)
        if by + bh < ey:
            new_spaces.append((bx, by + bh, sz, bx + bw, ey, ez))
        # Back split (z < bz, within x,y range of box)
        if bz > sz:
            new_spaces.append((bx, by, sz, bx + bw, by + bh, bz))
        # Front split (z > bz+bd, within x,y range of box)
        if bz + bd < ez:
            new_spaces.append((bx, by, bz + bd, bx + bw, by + bh, ez))
        
        return new_spaces
    
    def find_placement(w, h, d):
        """
        Find the best placement for a box of dimensions (w, h, d).
        Returns (bin_idx, x, y, z) or None if no placement found.
        Best = smallest (x, y, z) in any bin's free space.
        """
        best = None
        best_key = None
        
        for bin_idx, heap in enumerate(bins):
            # We need to check all spaces in the heap since the top might not fit
            # But we can prune: if the top space's min corner already exceeds our best, skip
            # Actually, let's just iterate through all spaces for correctness
            # For efficiency, we'll check spaces in order of their min corner
            
            # Make a copy of the heap to iterate (we'll modify it when placing)
            spaces_to_check = []
            while heap:
                item = heapq.heappop(heap)
                spaces_to_check.append(item)
            
            for x, y, z, space in spaces_to_check:
                sx, sy, sz, ex, ey, ez = space
                # Check if box fits in this space
                if sx + w <= ex and sy + h <= ey and sz + d <= ez:
                    # This is a valid placement at (sx, sy, sz)
                    key = (bin_idx, x, y, z)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (bin_idx, sx, sy, sz, space)
            
            # Put spaces back in the heap
            for item in spaces_to_check:
                heapq.heappush(heap, item)
        
        return best
    
    # Place boxes one by one
    placements = [None] * n  # indexed by original box index
    
    for orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in a new bin at origin (may overlap, but better than crash)
            # Actually, let's try to place them quickly
            for rem_idx, rw, rh, rd in indexed_boxes:
                if placements[rem_idx] is None:
                    # Just assign to last bin or new bin at (0,0,0) - this may be invalid
                    # Better: create new bins
                    bin_idx = new_bin()
                    placements[rem_idx] = (bin_idx, 0, 0, 0, rw, rh, rd)
            return [(b, x, y, z, w, h, d) for b, x, y, z, w, h, d in placements]
        
        result = find_placement(w, h, d)
        
        if result is None:
            # No space in existing bins, create a new bin
            bin_idx = new_bin()
            x, y, z = 0, 0, 0
            space = (0, 0, 0, W, H, D)
        else:
            bin_idx, x, y, z, space = result
        
        # Place the box at (x, y, z) in bin_idx
        placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
        
        # Update the free spaces in bin_idx
        heap = bins[bin_idx]
        new_spaces = []
        spaces_to_check = []
        while heap:
            item = heapq.heappop(heap)
            spaces_to_check.append(item)
        
        for sx_min, sy_min, sz_min, space in spaces_to_check:
            sx, sy, sz, ex, ey, ez = space
            # Check if this space contains the placed box
            if sx <= x and sy <= y and sz <= z and x + w <= ex and y + h <= ey and z + d <= ez:
                # Split this space
                box_tuple = (x, y, z, w, h, d)
                new_spaces.extend(split_space(space, box_tuple))
            else:
                # Keep this space as is
                new_spaces.append(space)
        
        # Rebuild heap
        bins[bin_idx] = []
        for space in new_spaces:
            heapq.heappush(bins[bin_idx], (space[0], space[1], space[2], space))
    
    return [(b, x, y, z, w, h, d) for b, x, y, z, w, h, d in placements]

# EVOLVE-BLOCK-END
