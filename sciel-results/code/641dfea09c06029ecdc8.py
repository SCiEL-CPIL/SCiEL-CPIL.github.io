"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
import random
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    # And a list of maximal empty spaces (x, y, z, w, h, d)
    bins = []
    
    def split_space(space, box):
        """
        Split a maximal space by a box placed inside it.
        Returns a list of new maximal spaces that do not overlap with the box.
        space: (sx, sy, sz, sw, sh, sd)
        box: (x, y, z, w, h, d)
        """
        sx, sy, sz, sw, sh, sd = space
        x, y, z, w, h, d = box
        
        new_spaces = []
        
        # Split in X dimension
        # Left part
        if sx < x:
            new_w = x - sx
            if new_w > 0:
                new_spaces.append((sx, sy, sz, new_w, sh, sd))
        # Right part
        if x + w < sx + sw:
            new_x = x + w
            new_w = sx + sw - new_x
            if new_w > 0:
                new_spaces.append((new_x, sy, sz, new_w, sh, sd))
        
        # Split in Y dimension (for the middle part in X)
        # The middle part in X is [x, x+w]
        # We need to split the region [x, x+w] x [sy, sy+sh] x [sz, sz+sd] by [x, x+w] x [y, y+h] x [z, z+d]
        # This effectively splits the Y-Z plane for the X-slab [x, x+w]
        
        # Bottom part (Y < y)
        if sy < y:
            new_h = y - sy
            if new_h > 0:
                # This space is in the X-slab [x, x+w]
                # But wait, the left and right parts already cover the full height.
                # The "middle" X-slab is [x, x+w].
                # So we add spaces in this X-slab that are below or above the box in Y,
                # and in front or behind the box in Z.
                
                # Bottom
                new_spaces.append((x, sy, sz, w, new_h, sd))
        
        # Top part (Y > y+h)
        if y + h < sy + sh:
            new_y = y + h
            new_h = sy + sh - new_y
            if new_h > 0:
                new_spaces.append((x, new_y, sz, w, new_h, sd))
        
        # Now split the remaining Y-slab [y, y+h] in Z dimension
        # Front part (Z < z)
        if sz < z:
            new_d = z - sz
            if new_d > 0:
                new_spaces.append((x, y, sz, w, h, new_d))
        
        # Back part (Z > z+d)
        if z + d < sz + sd:
            new_z = z + d
            new_d = sz + sd - new_z
            if new_d > 0:
                new_spaces.append((x, y, new_z, w, h, new_d))
                
        return new_spaces

    def can_place_in_space(space, w, h, d):
        """Check if a box with dims (w,h,d) fits in the given space."""
        sw, sh, sd = space[3], space[4], space[5]
        return w <= sw and h <= sh and d <= sd

    def find_best_space(bin_spaces, w, h, d):
        """
        Find the best space in the bin to place the box.
        Returns (space_index, x, y, z) or None if no space fits.
        Heuristic: Minimize x + y + z (compactness).
        """
        best_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for i, space in enumerate(bin_spaces):
            if can_place_in_space(space, w, h, d):
                x, y, z = space[0], space[1], space[2]
                # Score: prefer positions closer to origin
                score = x + y + z
                if score < best_score:
                    best_score = score
                    best_idx = i
                    best_pos = (x, y, z)
        
        if best_idx == -1:
            return None
        return best_idx, best_pos

    def place_box(w, h, d):
        """
        Place a box in the best existing bin or create a new one.
        Returns (bin_idx, x, y, z)
        """
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        best_space_idx = -1
        
        for b_idx, bin_data in enumerate(bins):
            bin_spaces = bin_data['spaces']
            result = find_best_space(bin_spaces, w, h, d)
            if result is not None:
                space_idx, pos = result
                x, y, z = pos
                # Score: prefer positions that minimize the "corner" distance
                score = x + y + z
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
                    best_space_idx = space_idx
        
        if best_bin == -1:
            # Create a new bin
            new_bin = {
                'boxes': [],
                'spaces': [(0, 0, 0, W, H, D)]
            }
            bins.append(new_bin)
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
            # The box is placed at (0,0,0) in the new bin
            # Update the spaces for the new bin
            space = (0, 0, 0, W, H, D)
            new_spaces = split_space(space, (0, 0, 0, w, h, d))
            bins[best_bin]['spaces'] = new_spaces
            bins[best_bin]['boxes'].append((0, 0, 0, w, h, d))
            return best_bin, 0, 0, 0
        
        # Update the existing bin
        bin_data = bins[best_bin]
        x, y, z = best_pos
        box = (x, y, z, w, h, d)
        
        # Remove the used space and add the split spaces
        old_space = bin_data['spaces'].pop(best_space_idx)
        new_spaces = split_space(old_space, box)
        bin_data['spaces'].extend(new_spaces)
        bin_data['boxes'].append(box)
        
        return best_bin, x, y, z

    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in a new bin each (valid but suboptimal)
            # We must ensure validity. Creating a new bin for each is safe.
            for i in range(idx, n):
                orig_idx, w2, h2, d2, _ = indexed_boxes[i]
                new_bin = {
                    'boxes': [],
                    'spaces': [(0, 0, 0, W, H, D)]
                }
                bins.append(new_bin)
                bin_idx = len(bins) - 1
                box = (0, 0, 0, w2, h2, d2)
                new_spaces = split_space((0, 0, 0, W, H, D), box)
                bins[bin_idx]['spaces'] = new_spaces
                bins[bin_idx]['boxes'].append(box)
                placements[orig_idx] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Fill any None placements (shouldn't happen if deadline guard works)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            new_bin = {
                'boxes': [],
                'spaces': [(0, 0, 0, W, H, D)]
            }
            bins.append(new_bin)
            bin_idx = len(bins) - 1
            box = (0, 0, 0, w, h, d)
            new_spaces = split_space((0, 0, 0, W, H, D), box)
            bins[bin_idx]['spaces'] = new_spaces
            bins[bin_idx]['boxes'].append(box)
            placements[i] = (bin_idx, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
