"""Maximal Space Partitioning (MSP) bin packing with best-fit selection."""
import time
import sys
import heapq
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: -x[1][0]*x[1][1]*x[1][2])
    
    # Each bin maintains a list of free spaces (disjoint axis-aligned cuboids)
    # Free space: (x, y, z, w, h, d) representing a box from (x,y,z) to (x+w, y+h, z+d)
    bins = []  # list of lists of free spaces
    placements = [None] * n  # placement per original index
    
    def new_bin():
        """Create a new bin with one free space covering the entire container."""
        return [(0, 0, 0, W, H, D)]
    
    def split_space(space, box_pos, box_dims):
        """
        Split a free space by placing a box at box_pos with box_dims.
        Returns a list of new free spaces that remain.
        space: (sx, sy, sz, sw, sh, sd)
        box_pos: (bx, by, bz)
        box_dims: (bw, bh, bd)
        """
        sx, sy, sz, sw, sh, sd = space
        bx, by, bz = box_pos
        bw, bh, bd = box_dims
        
        # The box occupies [bx, bx+bw) x [by, by+bh) x [bz, bz+bd)
        # We need to split the space into up to 6 non-overlapping sub-spaces
        new_spaces = []
        
        # Split along x-axis: left and right of the box
        if bx > sx:
            # Left portion
            new_spaces.append((sx, sy, sz, bx - sx, sh, sd))
        if bx + bw < sx + sw:
            # Right portion
            new_spaces.append((bx + bw, sy, sz, sx + sw - bx - bw, sh, sd))
        
        # For the y-axis split, we only consider the region in x that overlaps with the box
        # Actually, let's do a proper 3D split:
        # The space is [sx, sx+sw) x [sy, sy+sh) x [sz, sz+sd)
        # The box is [bx, bx+bw) x [by, by+bh) x [bz, bz+bd)
        # We split into regions that don't overlap with the box
        
        # Reset and do it properly
        new_spaces = []
        
        # Region below the box in z (if any)
        if bz > sz:
            new_spaces.append((sx, sy, sz, sw, sh, bz - sz))
        
        # Region above the box in z (if any)
        if bz + bd < sz + sd:
            new_spaces.append((sx, sy, bz + bd, sw, sh, sz + sd - bz - bd))
        
        # Now for the middle z-slab where the box exists in z: [bz, bz+bd)
        # But we need to be careful - the box only occupies [bx, bx+bw) x [by, by+bh) in that slab
        # So in the z-slab [max(bz, sz), min(bz+bd, sz+sd)), we have the box footprint
        
        z_slab_start = max(bz, sz)
        z_slab_end = min(bz + bd, sz + sd)
        if z_slab_start < z_slab_end:
            z_slab_depth = z_slab_end - z_slab_start
            
            # In this z-slab, the box occupies [bx, bx+bw) x [by, by+bh)
            # Split the x-y region around the box footprint
            
            # Below box in y
            if by > sy:
                new_spaces.append((sx, sy, z_slab_start, sw, by - sy, z_slab_depth))
            
            # Above box in y
            if by + bh < sy + sh:
                new_spaces.append((sx, by + bh, z_slab_start, sw, sy + sh - by - bh, z_slab_depth))
            
            # Now for the y-slab where the box exists: [max(by, sy), min(by+bh, sy+sh))
            y_slab_start = max(by, sy)
            y_slab_end = min(by + bh, sy + sh)
            if y_slab_start < y_slab_end:
                y_slab_height = y_slab_end - y_slab_start
                
                # Left of box in x
                if bx > sx:
                    new_spaces.append((sx, y_slab_start, z_slab_start, bx - sx, y_slab_height, z_slab_depth))
                
                # Right of box in x
                if bx + bw < sx + sw:
                    new_spaces.append((bx + bw, y_slab_start, z_slab_start, sx + sw - bx - bw, y_slab_height, z_slab_depth))
        
        return new_spaces
    
    def can_place_in_space(space, w, h, d):
        """Check if a box of dimensions (w,h,d) fits in the given space."""
        sw, sh, sd = space[3], space[4], space[5]
        return w <= sw and h <= sh and d <= sd
    
    def find_best_position(bin_free_spaces, w, h, d):
        """
        Find the best position to place a box in the given bin's free spaces.
        Returns (bin_index, position) or (None, None) if no fit.
        We use best-fit: minimize the wasted space, preferring positions that
        keep the box close to the origin for compactness.
        """
        best_score = float('inf')
        best_pos = None
        best_space_idx = None
        
        for idx, space in enumerate(bin_free_spaces):
            if not can_place_in_space(space, w, h, d):
                continue
            
            sx, sy, sz, sw, sh, sd = space
            
            # Try placing at the corner closest to origin within this space
            # Position (sx, sy, sz) is always valid if the box fits
            bx, by, bz = sx, sy, sz
            
            # Score: prefer positions that minimize x+y+z (compactness)
            # and also consider how well the box fits (less waste is better)
            waste = (sw - w) * sh * sd + (sh - h) * sw * sd + (sd - d) * sw * sh
            pos_score = bx + by + bz
            score = pos_score * 1000 + waste
            
            if score < best_score:
                best_score = score
                best_pos = (bx, by, bz)
                best_space_idx = idx
        
        return best_pos, best_space_idx
    
    # Main packing loop
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a new bin each
            for remaining_idx in range(orig_idx, n):
                ow, oh, od = boxes[remaining_idx]
                if placements[remaining_idx] is None:
                    b = len(bins)
                    bins.append(new_bin())
                    placements[remaining_idx] = (b, 0, 0, 0, ow, oh, od)
            return placements
        
        placed = False
        
        # Try to place in existing bins first
        for b_idx in range(len(bins)):
            free_spaces = bins[b_idx]
            pos, space_idx = find_best_position(free_spaces, w, h, d)
            if pos is not None:
                # Place the box
                placements[orig_idx] = (b_idx, pos[0], pos[1], pos[2], w, h, d)
                
                # Update free spaces
                old_space = free_spaces[space_idx]
                new_spaces = split_space(old_space, pos, (w, h, d))
                free_spaces[space_idx:space_idx+1] = new_spaces
                placed = True
                break
        
        if not placed:
            # Start a new bin
            b_idx = len(bins)
            bins.append(new_bin())
            placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
            # Place at origin, update free space
            new_spaces = split_space((0, 0, 0, W, H, D), (0, 0, 0), (w, h, d))
            bins[b_idx] = new_spaces
    
    # Polish pass: try to improve last-bin occupancy by moving small boxes
    if time.time() < deadline - 5.0:
        # Get the last bin index
        if bins:
            last_bin = max(p[0] for p in placements if p is not None)
            # Try to move boxes from earlier bins to the last bin to improve occupancy
            # This is complex; skip for now to keep within time budget
    
    # Ensure all placements are filled
    for i in range(n):
        if placements[i] is None:
            # Should not happen, but fallback
            w, h, d = boxes[i]
            b = len(bins)
            bins.append(new_bin())
            placements[i] = (b, 0, 0, 0, w, h, d)
    
    return placements
