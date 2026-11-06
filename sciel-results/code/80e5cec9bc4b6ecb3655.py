"""3D Bin Packing using Maximal Space Partitioning (Best-Fit Decreasing)."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending for better packing (Best-Fit Decreasing)
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Each bin is a list of free spaces: (x, y, z, w, h, d)
    bins = []
    # placements[original_index] = (bin_id, x, y, z, w, h, d)
    placements = [None] * len(boxes)
    
    def make_initial_bin():
        return [(0, 0, 0, W, H, D)]
    
    def split_space(space, box):
        """
        Given a free space and a box placed at (bx, by, bz) with dims (bw, bh, bd),
        split the space into up to 6 non-overlapping sub-spaces.
        Returns list of new free spaces.
        """
        sx, sy, sz, sw, sh, sd = space
        bx, by, bz = box[1], box[2], box[3]
        bw, bh, bd = box[4], box[5], box[6]
        
        # Check if box is fully inside space
        if bx < sx or bx + bw > sx + sw or by < sy or by + bh > sy + sh or bz < sz or bz + bd > sz + sd:
            # This shouldn't happen if we only place in valid spaces, but just in case
            return [space]
        
        new_spaces = []
        
        # Split along X
        if bx > sx:
            new_spaces.append((sx, sy, sz, bx - sx, sh, sd))
        if bx + bw < sx + sw:
            new_spaces.append((bx + bw, sy, sz, sx + sw - (bx + bw), sh, sd))
        
        # Split along Y (within the x-range of the box)
        x_start = max(sx, bx)
        x_end = min(sx + sw, bx + bw)
        if by > sy:
            new_spaces.append((x_start, sy, sz, x_end - x_start, by - sy, sd))
        if by + bh < sy + sh:
            new_spaces.append((x_start, by + bh, sz, x_end - x_start, sy + sh - (by + bh), sd))
        
        # Split along Z (within the x-y range of the box)
        y_start = max(sy, by)
        y_end = min(sy + sh, by + bh)
        if bz > sz:
            new_spaces.append((x_start, y_start, sz, x_end - x_start, y_end - y_start, bz - sz))
        if bz + bd < sz + sd:
            new_spaces.append((x_start, y_start, bz + bd, x_end - x_start, y_end - y_start, sz + sd - (bz + bd)))
        
        return new_spaces
    
    def find_best_fit(bin_spaces, w, h, d):
        """
        Find the best fitting space in the bin for a box of dimensions (w, h, d).
        Returns (space_index, x, y, z) or (None, None, None, None) if no fit.
        Best fit: minimize wasted volume. Try placing at multiple candidate positions
        within each free space and pick the one that minimizes fragmentation.
        """
        best_idx = -1
        best_score = float('inf')
        best_pos = (0, 0, 0)
        
        for i, (sx, sy, sz, sw, sh, sd) in enumerate(bin_spaces):
            if w <= sw and h <= sh and d <= sd:
                # Try several candidate positions within this space
                # Candidate 1: origin of space
                candidates = [(sx, sy, sz)]
                # Also try pushing to other corners/edges to reduce fragmentation
                # Right side
                candidates.append((sx + sw - w, sy, sz))
                # Top
                candidates.append((sx, sy + sh - h, sz))
                # Back
                candidates.append((sx, sy, sz + sd - d))
                # Combine: right + top
                candidates.append((sx + sw - w, sy + sh - h, sz))
                # Combine: right + back
                candidates.append((sx + sw - w, sy, sz + sd - d))
                # Combine: top + back
                candidates.append((sx, sy + sh - h, sz + sd - d))
                # All three
                candidates.append((sx + sw - w, sy + sh - h, sz + sd - d))
                
                for (cx, cy, cz) in candidates:
                    # Verify this position is valid (within space bounds)
                    if cx < sx or cx + w > sx + sw:
                        continue
                    if cy < sy or cy + h > sy + sh:
                        continue
                    if cz < sz or cz + d > sz + sd:
                        continue
                    
                    # Score: prefer positions that create larger, more usable remaining spaces
                    # Compute the "waste" as the sum of areas of the resulting fragments
                    # A simpler proxy: minimize the maximum dimension of remaining gaps
                    gap_x = sw - w
                    gap_y = sh - h
                    gap_z = sd - d
                    
                    # Prefer tight fits: smaller gaps are better
                    # Use a weighted sum where we penalize large gaps more
                    score = gap_x * gap_y * gap_z  # volume of wasted space if we could only use one fragment
                    # Also add a term to prefer compact placements
                    score += 0.1 * (gap_x + gap_y + gap_z)
                    
                    if score < best_score:
                        best_score = score
                        best_idx = i
                        best_pos = (cx, cy, cz)
        
        if best_idx == -1:
            return None, None, None, None
        return best_idx, best_pos[0], best_pos[1], best_pos[2]
    
    # Main loop
    for orig_idx, (w, h, d) in indexed_boxes:
        placed = False
        
        # Try existing bins first (Best Fit)
        best_bin = -1
        best_space_idx = -1
        best_x, best_y, best_z = 0, 0, 0
        best_global_slack = float('inf')
        
        for b in range(len(bins)):
            space_idx, px, py, pz = find_best_fit(bins[b], w, h, d)
            if space_idx is not None:
                # Calculate slack for comparison across bins
                space = bins[b][space_idx]
                slack = (space[3] - w) + (space[4] - h) + (space[5] - d)
                if slack < best_global_slack:
                    best_global_slack = slack
                    best_bin = b
                    best_space_idx = space_idx
                    best_x, best_y, best_z = px, py, pz
        
        if best_bin != -1:
            # Place in existing bin
            bin_spaces = bins[best_bin]
            space = bin_spaces[best_space_idx]
            
            # Create the box tuple for splitting: (dummy, x, y, z, w, h, d)
            box_tuple = (0, best_x, best_y, best_z, w, h, d)
            
            # Split the space
            new_spaces = split_space(space, box_tuple)
            
            # Replace the old space with the new ones
            bin_spaces[best_space_idx:best_space_idx+1] = new_spaces
            
            placements[orig_idx] = (best_bin, best_x, best_y, best_z, w, h, d)
            placed = True
        
        if not placed:
            # Open a new bin
            new_bin = make_initial_bin()
            bin_id = len(bins)
            bins.append(new_bin)
            
            # Place at origin
            x, y, z = 0, 0, 0
            box_tuple = (0, x, y, z, w, h, d)
            new_spaces = split_space(new_bin[0], box_tuple)
            new_bin[0:1] = new_spaces
            
            placements[orig_idx] = (bin_id, x, y, z, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
