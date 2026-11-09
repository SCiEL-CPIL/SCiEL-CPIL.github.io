"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import namedtuple

# BSP Node: represents an empty rectangular space
# We use a simple approach: maintain a list of empty boxes.
# When placing a box, we find an empty box that fits, then subdivide it.

def pack(boxes, container):
    W, H, D = container
    # Sort boxes by volume descending to pack larger items first
    # But we must preserve the original order in the output?
    # "preserving its original ordered dimensions" refers to w,h,d not being rotated.
    # The output list should correspond to the input list order?
    # "one placement ... per box" implies the i-th element of the return list corresponds to the i-th box in input.
    # So we need to assign placements to specific indices.
    
    n = len(boxes)
    # indices sorted by volume descending
    indices = list(range(n))
    indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # List of empty spaces: each is (x, y, z, w, h, d)
    empty_spaces = [(0, 0, 0, W, H, D)]
    
    # Result array
    placements = [None] * n
    
    # To improve packing, we might want to try multiple positions in the empty space
    # But simple BSP insertion:
    # For a given box (w,h,d), find the first empty space that contains it.
    # If found, place it at the bottom-left-front corner of that space (x, y, z).
    # Then subdivide the space into up to 3 new empty spaces.
    # If not found, create a new bin.
    
    # To optimize, we can sort empty spaces by some heuristic (e.g., closest to origin)
    # to encourage tighter packing.
    
    for idx in indices:
        w, h, d = boxes[idx]
        placed = False
        # Try to place in existing bins (empty_spaces list is global across bins? No, per bin)
        # Wait, the current empty_spaces list is for the *current* bin only?
        # If we use a single list for all bins, we need to track which bin each space belongs to.
        # Or we can maintain a list of bins, each with its own list of empty spaces.
        
        # Let's use a list of bins. Each bin is a dict or list of empty spaces.
        # bin 0 starts with one empty space.
        
        # Actually, the previous code used a single list `empty_spaces` and incremented `b`.
        # Let's structure it as:
        # bins = [ [ (0,0,0,W,H,D) ] ]  # list of lists of empty spaces
        # We need to modify the loop to iterate through bins.
        
        # Let's restart the structure inside the function.
        
    # Re-initializing logic with per-bin empty spaces
    bins = []
    # Add first bin
    bins.append([(0, 0, 0, W, H, D)])
    
    for idx in indices:
        w, h, d = boxes[idx]
        placed = False
        bin_idx = -1
        space_idx = -1
        
        # Search for a place in existing bins
        # We can iterate through bins and their empty spaces
        # To make it faster, we could keep empty spaces in a global list with bin IDs,
        # but simple iteration is fine for 40 boxes.
        
        for b_idx, spaces in enumerate(bins):
            for s_idx, (sx, sy, sz, sw, sh, sd) in enumerate(spaces):
                if w <= sw and h <= sh and d <= sd:
                    # Fits in this space
                    # Place at (sx, sy, sz)
                    bin_idx = b_idx
                    space_idx = s_idx
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            # Create new bin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, W, H, D)])
            space_idx = 0
            # The space is (0,0,0,W,H,D)
            sx, sy, sz, sw, sh, sd = 0, 0, 0, W, H, D
        
        # Get the space
        sx, sy, sz, sw, sh, sd = bins[bin_idx][space_idx]
        
        # Place box at (sx, sy, sz)
        placements[idx] = (bin_idx, sx, sy, sz, w, h, d)
        
        # Subdivide the space
        # Remove the old space
        bins[bin_idx].pop(space_idx)
        
        # Add new spaces
        # 1. Space to the right (x + w)
        if sx + w < sx + sw:
            bins[bin_idx].append((sx + w, sy, sz, sw - w, sh, sd))
        # 2. Space above (y + h) - only if the box didn't fill the width? 
        #    Actually, standard BSP splits:
        #    After placing at (sx, sy, sz) with size (w, h, d) in space (sx, sy, sz, sw, sh, sd):
        #    - Right: (sx+w, sy, sz, sw-w, sh, sd) if sw-w > 0
        #    - Above: (sx, sy+h, sz, w, sh-h, sd) if sh-h > 0  (only the part directly above the placed box)
        #    - Behind: (sx, sy, sz+d, sw, h, sd-d) if sd-d > 0  (only the part directly behind the placed box)
        #    Wait, this leaves a gap if the box doesn't fill the space completely in x or y.
        #    Standard "Bottom-Left-Front" with simple subdivision:
        #    The remaining space is an L-shape or more complex.
        #    A common simple BSP approach:
        #    Split the space into 3 parts:
        #    1. (sx+w, sy, sz, sw-w, sh, sd)  # Right
        #    2. (sx, sy+h, sz, w, sh-h, sd)  # Top (above the placed box, same width as box)
        #    3. (sx, sy, sz+d, sw, h, sd-d)  # Back (behind the placed box, same width/height as space? No, same as box? )
        #    
        #    Let's refine:
        #    The placed box occupies [sx, sx+w] x [sy, sy+h] x [sz, sz+d].
        #    The original space is [sx, sx+sw] x [sy, sy+sh] x [sz, sz+sd].
        #    
        #    Empty regions:
        #    R1: x in [sx+w, sx+sw], y in [sy, sy+sh], z in [sz, sz+sd]  -> (sx+w, sy, sz, sw-w, sh, sd)
        #    R2: x in [sx, sx+w], y in [sy+h, sy+sh], z in [sz, sz+sd]  -> (sx, sy+h, sz, w, sh-h, sd)
        #    R3: x in [sx, sx+sw], y in [sy, sy+h], z in [sz+d, sz+sd]  -> (sx, sy, sz+d, sw, h, sd-d)
        #    
        #    Wait, R2 and R3 overlap in the region [sx, sx+w] x [sy+h, sy+sh] x [sz+d, sz+sd]?
        #    No, R2 is y > sy+h, R3 is z > sz+d.
        #    The region [sx, sx+w] x [sy+h, sy+sh] x [sz+d, sz+sd] is included in both R2 and R3?
        #    R2: x in [sx, sx+w], y in [sy+h, sy+sh], z in [sz, sz+sd]. This includes z > sz+d.
        #    R3: x in [sx, sx+sw], y in [sy, sy+h], z in [sz+d, sz+sd]. This includes x > sx+w? No, x up to sx+sw.
        #    
        #    This simple 3-split creates overlaps in the empty space representation if not careful.
        #    Actually, the standard simple BSP for packing uses:
        #    1. Right: (sx+w, sy, sz, sw-w, sh, sd)
        #    2. Top: (sx, sy+h, sz, sw, sh-h, sd)  <-- Note: width is sw, not w.
        #    3. Back: (sx, sy, sz+d, sw, h, sd-d) <-- Note: height is h, not sh? Or sh?
        #    
        #    If we use:
        #    1. (sx+w, sy, sz, sw-w, sh, sd)
        #    2. (sx, sy+h, sz, sw, sh-h, sd)
        #    3. (sx, sy, sz+d, w, h, sd-d)  <-- This is getting complicated.
        #    
        #    Let's use a safer, though less efficient, method for subdivision:
        #    Just keep the whole remaining space as one box if it's rectangular?
        #    No, the remaining space is not rectangular.
        #    
        #    Alternative: Use a simpler heuristic.
        #    Instead of complex BSP, we can use a "Shelf" approach but with better sorting.
        #    The current seed is a shelf approach.
        #    Let's try a "Best Fit" with a simple grid or list of empty spaces, but handle subdivision correctly.
        #    
        #    Correct non-overlapping subdivision of a box after placing a smaller box at the corner:
        #    The remaining space can be represented by 3 boxes:
        #    1. (sx+w, sy, sz, sw-w, sh, sd)
        #    2. (sx, sy+h, sz, w, sh-h, sd)
        #    3. (sx, sy, sz+d, sw, h, sd-d)
        #    
        #    Let's check for overlaps between these 3 new boxes:
        #    1 & 2: 1 has x >= sx+w, 2 has x <= sx+w. No overlap in x (touching at boundary).
        #    1 & 3: 1 has x >= sx+w, 3 has x <= sx+sw. 3 has z >= sz+d, 1 has z <= sz+sd.
        #           Overlap in x: [sx+w, sx+sw]. Overlap in z: [sz+d, sz+sd]. Overlap in y: [sy, sy+h] (3) vs [sy, sy+sh] (1).
        #           So 1 and 3 DO overlap in the region [sx+w, sx+sw] x [sy, sy+h] x [sz+d, sz+sd].
        #    This means we cannot just add all 3.
        #    
        #    We need to be more careful.
        #    Let's use a different strategy:
        #    When placing a box, we find the best empty space.
        #    We then *remove* that space and add the *remaining* parts.
        #    To avoid overlaps, we can use a recursive subdivision that ensures disjointness.
        #    
        #    Simpler approach for this task (40 boxes):
        #    Use a list of empty spaces.
        #    When placing a box in space S, remove S.
        #    Add back the parts of S that are not covered by the box.
        #    Since the box is at the corner (sx, sy, sz), the uncovered parts are:
        #    - If box doesn't fill x: (sx+w, sy, sz, sw-w, sh, sd)
        #    - If box doesn't fill y: (sx, sy+h, sz, w, sh-h, sd)  <-- Only the part directly above the box's x-range
        #    - If box doesn't fill z: (sx, sy, sz+d, sw, h, sd-d)  <-- Only the part directly behind the box's x,y range?
        #    
        #    Actually, let's just use a "Skyline" or "Shelf" approach which is simpler and robust.
        #    The seed is already a shelf approach.
        #    Let's improve the shelf approach by sorting boxes and trying to fit them into existing shelves.
        
        # Let's abandon the complex BSP for a moment and stick to a robust Shelf/Row approach
        # but with better box ordering and bin selection.
        
        # The current seed:
        # - Processes boxes in input order.
        # - Uses a single "current" position within a bin.
        # - Moves to next row if x overflow, next shelf if y overflow, next layer if z overflow, next bin if z overflow.
        # - This is a "First Fit" into a linearized shelf structure.
        
        # Improvement:
        # 1. Sort boxes by volume (descending).
        # 2. For each box, try to place it in the best available position in existing bins.
        #    We can maintain for each bin a list of "shelves" or just try to find a position.
        #    Given the small number of boxes (40), we can afford to check all existing placements
        #    to find a valid (x,y,z) for a new box in a bin.
        #    
        #    Algorithm:
        #    1. Sort indices by volume desc.
        #    2. bins = [] (list of lists of placements)
        #    3. For each box in sorted order:
        #       a. Try to place in existing bins (0 to len(bins)-1).
        #          For each bin, check all possible (x,y,z) positions?
        #          Too slow.
        #       b. Instead, use a heuristic:
        #          For each bin, maintain the "frontier" or use a simple grid check.
        #          Or, simpler: Just use the shelf heuristic but allow re-evaluating bins.
        
        # Let's try a "Best Fit Decreasing" with a simple 3D grid check for empty spots.
        # Since W,H,D are likely small or moderate, we can discretize? No, continuous.
        
        # Let's go back to the BSP idea but implement it correctly.
        # The issue with the 3-split was overlap.
        # Correct split:
        # Space: (sx, sy, sz, sw, sh, sd)
        # Box: (w, h, d) at (sx, sy, sz)
        # 
        # New spaces:
        # 1. (sx+w, sy, sz, sw-w, sh, sd)  # Right
        # 2. (sx, sy+h, sz, w, sh-h, sd)  # Top (above box, same width)
        # 3. (sx, sy, sz+d, sw, h, sd-d)  # Back (behind box, full width, same height as box? No, same height as space? )
        
        # Let's check overlap of 1 and 3 again.
        # 1: x in [sx+w, sx+sw], y in [sy, sy+sh], z in [sz, sz+sd]
        # 3: x in [sx, sx+sw], y in [sy, sy+h], z in [sz+d, sz+sd]
        # Overlap region: x in [sx+w, sx+sw], y in [sy, sy+h], z in [sz+d, sz+sd].
        # This region is empty in the original space (not covered by box).
        # But it is included in both 1 and 3. So we have double counting of empty space.
        # This is allowed as long as we don't place a box in the overlap twice.
        # But if we place a box in 1, it might overlap with a box placed in 3 if they share the overlap region.
        # So we must ensure that when we place a box in one of these sub-spaces, we subdivide *that* sub-space,
        # and the other sub-space remains untouched.
        # The problem is that the sub-spaces are not disjoint.
        
        # To make them disjoint, we can use:
        # 1. (sx+w, sy, sz, sw-w, sh, sd)
        # 2. (sx, sy+h, sz, sw, sh-h, sd)  # Top, full width
        # 3. (sx, sy, sz+d, w, h, sd-d)    # Back, only above the box's footprint?
        
        # Let's check 1 and 2:
        # 1: x >= sx+w, 2: x <= sx+sw. Overlap in x: [sx+w, sx+sw].
        # 1: y <= sy+sh, 2: y >= sy+h. Overlap in y: [sy+h, sy+sh].
        # 1: z <= sz+sd, 2: z <= sz+sd. Overlap in z.
        # So 1 and 2 overlap in [sx+w, sx+sw] x [sy+h, sy+sh] x [sz, sz+sd].
        
        # This is getting messy.
        
        # Let's use a very simple, robust method:
        # For each bin, maintain a list of placed boxes.
        # To find a place for a new box, iterate through a grid of candidate positions.
        # Candidate x: 0, and x+w_i for all placed boxes i.
        # Candidate y: 0, and y+h_i for all placed boxes i.
        # Candidate z: 0, and z+d_i for all placed boxes i.
        # Check each (x,y,z) to see if the box fits without overlap.
        # With 40 boxes, the number of candidates is at most 40^3 = 64000 per bin.
        # Checking overlap is O(N) per candidate, so 64000 * 40 = 2.5M ops per box.
        # 40 boxes * 2.5M = 100M ops. This might be too slow in Python.
        
        # Optimization: Only check candidates in the "frontier".
        # Or, use the shelf heuristic which is O(1) per box.
        
        # Let's improve the shelf heuristic:
        # 1. Sort boxes by height (or volume).
        # 2. For each bin, maintain shelves.
        # 3. Try to fit the box into the best shelf (lowest y, then lowest x).
        
        # Here is a better Shelf algorithm:
        # bins = [] # Each bin is a list of shelves. Each shelf is (y, row_h, x_pos, boxes)
        # Actually, simpler:
        # bins = [] # Each bin is a list of placed boxes (x,y,z,w,h,d)
        
        # For each box (sorted):
        #   best_bin = -1
        #   best_pos = None
        #   best_score = inf
        #   for each bin:
        #      for each candidate y (0, and y+h of existing boxes):
        #         for each candidate x (0, and x+w of existing boxes at same y):
        #            for each candidate z (0, and z+d of existing boxes at same x,y):
        #               if fits:
        #                  score = y*W*H + x*H + z (prefer low y, then low x, then low z)
        #                  if score < best_score:
        #                     best_score = score
        #                     best_bin = bin_idx
        #                     best_pos = (x,y,z)
        #   if best_bin == -1:
        #      create new bin
        #   place box
        
        # This is still potentially slow.
        
        # Let's stick to the simple Shelf but with sorting.
        # The seed is:
        # for w,h,d in boxes:
        #   if x+w > W: x=0, y+=row_h, row_h=0
        #   if y+h > H: y=0, z+=layer_d, layer_d=0, x=0, row_h=0
        #   if z+d > D: b+=1, reset all
        #   place
        #   x+=w, row_h=max(row_h,h), layer_d=max(layer_d,d)
        
        # This is a "First Fit" into a single linear shelf per layer.
        
        # Improvement:
        # 1. Sort boxes by volume desc.
        # 2. Keep track of multiple bins.
        # 3. For each box, try to place it in the bin where it fits with the least "waste".
        
        # Let's try a "Best Fit" with a simple 1D shelf per bin.
        # Each bin has a current (x, y, z, row_h, layer_d).
        # When a box doesn't fit in the current position of a bin, we move to the next row/shelf/layer.
        # If it doesn't fit anywhere in a bin, we try the next bin.
        # If it doesn't fit in any bin, we create a new bin.
        
        # This is similar to the seed but allows checking multiple bins.
        
        bins_state = [] # Each entry: (x, y, z, row_h, layer_d)
        placements = [None] * n
        indices = list(range(n))
        indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for idx in indices:
            w, h, d = boxes[idx]
            placed = False
            best_bin = -1
            best_state = None
            
            for b_idx, state in enumerate(bins_state):
                x, y, z, row_h, layer_d = state
                # Try to advance the state to fit the box
                nx, ny, nz, nrow_h, nlayer_d = x, y, z, row_h, layer_d
                
                # Advance x
                if nx + w > W:
                    nx = 0
                    ny = ny + nrow_h
                    nrow_h = 0
                
                # Advance y
                if ny + h > H:
                    ny = 0
                    nz = nz + nlayer_d
                    nlayer_d = 0
                    nx = 0
                    nrow_h = 0
                
                # Advance z
                if nz + d > D:
                    # Doesn't fit in this bin
                    continue
                
                # It fits
                # Score: prefer bins that are more "full" or lower coordinates
                score = nz * W * H + ny * W + nx
                if best_bin == -1 or score < best_state[0]:
                    best_bin = b_idx
                    best_state = (score, nx, ny, nz, nrow_h, nlayer_d)
            
            if best_bin == -1:
                # Create new bin
                best_bin = len(bins_state)
                bins_state.append((0, 0, 0, 0, 0))
                nx, ny, nz, nrow_h, nlayer_d = 0, 0, 0, 0, 0
            else:
                _, nx, ny, nz, nrow_h, nlayer_d = best_state
                # Update the bin state
                bins_state[best_bin] = (nx + w, ny, nz, max(nrow_h, h), max(nlayer_d, d))
                # Wait, the state update in the seed was:
                # x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
                # But y and z were updated in the "advance" steps.
                # So the new state is (nx+w, ny, nz, max(nrow_h, h), max(nlayer_d, d))
            
            placements[idx] = (best_bin, nx, ny, nz, w, h, d)
        
        return placements

# EVOLVE-BLOCK-END
