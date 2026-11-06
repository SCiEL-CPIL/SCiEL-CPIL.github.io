import time
import math
import heapq
from typing import List, Tuple, Optional

# EVOLVE-BLOCK-START

def pack(boxes, container):
    """
    Packs boxes into containers using a Best-Fit Decreasing (BFD) heuristic.
    Returns a list of tuples (bin_index, x, y, z, w, h, d).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin for 150s limit
    
    W, H, D = container
    container_volume = W * H * D
    
    # If no boxes, return empty list
    if not boxes:
        return []
        
    # We will try a few different sorting strategies and pick the best result
    # Strategy 1: Sort by volume (largest first)
    # Strategy 2: Sort by max dimension (largest first)
    # Strategy 3: Sort by sum of dimensions (largest first)
    
    results = []
    
    # Define sorting keys
    sort_keys = [
        lambda b: b[0]*b[1]*b[2],
        lambda b: max(b),
        lambda b: b[0]+b[1]+b[2]
    ]
    
    best_score = float('inf')
    best_placements = []
    
    for key_idx, sort_key in enumerate(sort_keys):
        if time.time() > deadline:
            break
            
        # Sort boxes by the current key, descending
        # We need to keep original indices to return correct placements? 
        # The task says "preserving its original ordered dimensions" but the output is 
        # "one placement ... per box". Usually, the order in the output list corresponds 
        # to the input order of boxes, or we just need to return a valid set of placements.
        # The prompt says: "one placement (bin, x, y, z, w, h, d) per box".
        # It does not explicitly state the order of the output list must match input order, 
        # but typically these problems expect the i-th output tuple to correspond to the i-th input box.
        # Let's assume the output list must correspond to the input box order.
        
        sorted_indices = sorted(range(len(boxes)), key=lambda i: sort_key(boxes[i]), reverse=True)
        
        placements = _pack_bfd(boxes, container, sorted_indices, deadline)
        
        if placements is None:
            continue
            
        # Calculate score
        score = _calculate_score(placements, boxes, container)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            
    # If no valid result was found (should not happen), fallback to simple shelf
    if not best_placements:
        best_placements = _pack_shelf(boxes, container)
        
    return best_placements

def _calculate_score(placements, boxes, container):
    """
    Calculate the score: B + 0.1 * (1 - last_bin_occupied_volume / container_volume)
    """
    if not placements:
        return 0.0
        
    W, H, D = container
    container_volume = W * H * D
    
    bins_used = {}
    last_bin = -1
    
    for p in placements:
        bin_idx = p[0]
        w, h, d = p[4], p[5], p[6]
        if bin_idx not in bins_used:
            bins_used[bin_idx] = 0
        bins_used[bin_idx] += w * h * d
        
    if not bins_used:
        return 0.0
        
    num_bins = len(bins_used)
    last_bin = max(bins_used.keys())
    
    # The "last bin" is the one with the largest label.
    # The score definition says: "largest label defines the last bin".
    # "last-bin occupied volume fraction"
    
    last_bin_vol = bins_used[last_bin]
    fraction = last_bin_vol / container_volume if container_volume > 0 else 1.0
    
    score = num_bins + 0.1 * (1 - fraction)
    
    return score

def _pack_bfd(boxes, container, sorted_indices, deadline):
    """
    Core BFD packing algorithm.
    """
    W, H, D = container
    
    # List of bins. Each bin is a list of free spaces (x, y, z, w, h, d)
    bins = []
    # Map from original box index to placement tuple
    placements_map = {}
    
    # Initialize first bin with the whole container as a free space
    bins.append([(0, 0, 0, W, H, D)])
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        
        if time.time() > deadline:
            return None # Timeout, return best so far or None
            
        best_bin_idx = -1
        best_pos = None
        best_remaining_vol = float('inf')
        
        # Try to fit the box in existing bins
        for b_idx, free_spaces in enumerate(bins):
            # Find the best free space in this bin
            pos = _find_best_position_in_bin(free_spaces, w, h, d)
            if pos is not None:
                # Calculate the "waste" or remaining volume after placing
                # We prefer the space that leaves the least wasted volume or is "tightest"
                # A simple heuristic: minimize the sum of dimensions of the remaining free space
                # Or just pick the first fit? Best Fit usually means smallest free space that fits.
                
                # Let's refine: Find the free space that can contain the box and has the smallest volume
                # Actually, _find_best_position_in_bin should return the position that minimizes
                # some cost function. Let's modify it to return the position with minimum "waste".
                
                # For now, let's assume _find_best_position_in_bin returns the "best" position 
                # according to a heuristic (e.g., corner, then smallest gap).
                # We need to compare across bins.
                
                # Let's use the volume of the free space used as a proxy for tightness.
                # Smaller free space used = better fit.
                
                # We need to re-evaluate _find_best_position_in_bin to return metadata.
                
                pass
        
        # The above loop is incomplete. Let's rewrite the logic more cleanly.
        
        # Reset
        best_bin_idx = -1
        best_pos = None
        best_fit_score = float('inf')
        
        for b_idx, free_spaces in enumerate(bins):
            # Find the best free space in this bin for this box
            pos, fit_score = _find_best_position_in_bin(free_spaces, w, h, d)
            if pos is not None:
                if fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin_idx = b_idx
                    best_pos = pos
        
        if best_bin_idx == -1:
            # Open a new bin
            best_bin_idx = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
            
        # Place the box in the best bin
        x, y, z = best_pos
        free_spaces = bins[best_bin_idx]
        
        # Update free spaces
        new_free_spaces = _update_free_spaces(free_spaces, x, y, z, w, h, d)
        bins[best_bin_idx] = new_free_spaces
        
        placements_map[idx] = (best_bin_idx, x, y, z, w, h, d)
        
    # Reconstruct placements in original order
    placements = [placements_map[i] for i in range(len(boxes))]
    return placements

def _find_best_position_in_bin(free_spaces, w, h, d):
    """
    Finds the best position for a box (w, h, d) in a list of free spaces.
    Returns (position, fit_score) or (None, inf).
    fit_score is lower for better fits (e.g., smaller remaining volume, or closer to corner).
    """
    best_pos = None
    best_score = float('inf')
    
    for fx, fy, fz, fw, fh, fd in free_spaces:
        if w <= fw and h <= fh and d <= fd:
            # It fits. Now, which position within this free space is best?
            # Common heuristics:
            # 1. Place at (fx, fy, fz) - bottom-left-front corner.
            # 2. Minimize the "waste" volume.
            
            # Let's try placing at the corner (fx, fy, fz)
            # The "waste" is the volume of the free space minus the box volume?
            # Or the sum of the dimensions of the resulting gaps?
            
            # Let's use a score based on the "tightness" of the fit.
            # Score = (fw - w) + (fh - h) + (fd - d)
            # Lower score means tighter fit.
            
            score = (fw - w) + (fh - h) + (fd - d)
            
            # Also, we might prefer positions closer to the origin to keep the "front" of the bin packed?
            # For now, let's just use the tightness score.
            # If multiple spaces have the same tightness, we can break ties by position.
            
            if score < best_score:
                best_score = score
                best_pos = (fx, fy, fz)
                
    if best_pos is None:
        return None, float('inf')
        
    return best_pos, best_score

def _update_free_spaces(free_spaces, x, y, z, w, h, d):
    """
    Removes the space occupied by the box from the free spaces list
    and adds the remaining sub-spaces.
    """
    new_free_spaces = []
    
    for fx, fy, fz, fw, fh, fd in free_spaces:
        # Check if the box is inside this free space
        # The box is placed at (x, y, z) with size (w, h, d)
        # The free space is (fx, fy, fz) with size (fw, fh, fd)
        
        # We assume the box is placed at the corner (fx, fy, fz) of the selected free space.
        # So x == fx, y == fy, z == fz.
        
        # If the box doesn't fill the entire free space, we split it.
        
        # 1. Right of the box (x + w to fx + fw)
        if x + w < fx + fw:
            new_free_spaces.append((x + w, y, z, fx + fw - (x + w), h, d))
            
        # 2. Above the box (y + h to fy + fh)
        # But only for the part of the free space that is NOT to the right of the box?
        # Standard splitting:
        # If we place a box at (x,y,z) in a space (fx,fy,fz,fw,fh,fd):
        # The box occupies [x, x+w], [y, y+h], [z, z+d].
        # Since x=fx, y=fy, z=fz:
        # Box occupies [fx, fx+w], [fy, fy+h], [fz, fz+d].
        
        # Remaining regions:
        # A: x in [fx+w, fx+fw], y in [fy, fy+h], z in [fz, fz+d]  (Right)
        # B: x in [fx, fx+w], y in [fy+h, fy+fh], z in [fz, fz+d]  (Above/Back)
        # C: x in [fx, fx+fw], y in [fy, fy+fh], z in [fz+d, fz+fd]  (Top)
        
        # Wait, this simple splitting creates overlaps if not done carefully.
        # The standard "guillotine" or "corner" splitting creates 3 new boxes:
        # 1. (x+w, y, z, fw-w, h, d)  -- Right
        # 2. (x, y+h, z, w, fh-h, d)  -- Above
        # 3. (x, y, z+d, w, h, fd-d)  -- Behind
        
        # However, region (x+w, y+h, z, fw-w, fh-h, d) is covered by both 1 and 2?
        # No, 1 is x > x+w. 2 is y > y+h. They don't overlap in x or y.
        # But what about the space (x+w, y+h, z+d)? It is covered by 1 (if y+h <= y+h... no, 1 has y in [y, y+h])?
        # Let's check coordinates.
        # Space 1: x in [x+w, fx+fw], y in [y, y+h], z in [z, z+d].
        # Space 2: x in [x, x+w], y in [y+h, fy+fh], z in [z, z+d].
        # Space 3: x in [x, x+w], y in [y, y+h], z in [z+d, fz+fd].
        
        # The region x > x+w, y > y+h, z > z+d is NOT covered by any of these 3.
        # This simple 3-way split is insufficient for non-guillotine cuts or complex remaining spaces.
        # However, for a "corner" placement, the remaining space is an L-shape (or more complex).
        # To handle this correctly, we often need to keep the free space as a list of non-overlapping boxes.
        # The 3-way split above actually leaves a gap: the box at (x+w, y+h, z+d) with size (fw-w, fh-h, fd-d) is not accounted for in the simple 3-way split if we just add those 3.
        
        # Let's use a more robust splitting:
        # The remaining space is the original space minus the box.
        # We can decompose the remaining space into up to 6 boxes (one for each face of the box).
        # But they must be non-overlapping.
        
        # A common method:
        # 1. Split along X: Left/Right. Since box is at fx, only Right remains.
        #    Right: (x+w, y, z, fw-w, h, d) -- wait, this only covers the height h and depth d of the box.
        #    The rest of the height and depth in the right section is free.
        
        # Let's do this:
        # 1. Right of box: (x+w, y, z, fw-w, fh, fd)
        # 2. Above box (in the left part): (x, y+h, z, w, fh-h, d)
        # 3. Behind box (in the left, below part): (x, y, z+d, w, h, fd-d)
        
        # This covers:
        # - All x > x+w (covered by 1)
        # - x <= x+w, y > y+h (covered by 2)
        # - x <= x+w, y <= y+h, z > z+d (covered by 3)
        
        # This decomposition is valid and non-overlapping.
        
        if x + w < fx + fw:
            # Right part
            new_free_spaces.append((x + w, y, z, fx + fw - (x + w), fh, fd))
            
        if y + h < fy + fh:
            # Above part (only for x in [x, x+w])
            new_free_spaces.append((x, y + h, z, w, fy + fh - (y + h), d))
            
        if z + d < fz + fd:
            # Behind part (only for x in [x, x+w], y in [y, y+h])
            new_free_spaces.append((x, y, z + d, w, h, fz + fd - (z + d)))
            
        # If the box fills the entire space, no new free spaces are added.
        
    # Remove the original free space that was used?
    # The input `free_spaces` list contains the specific space that was used.
    # We should NOT add the other free spaces from the original list that were not used.
    # Wait, `_update_free_spaces` receives the list of ALL free spaces in the bin.
    # We need to remove the one that was used and add the new sub-spaces.
    
    # How do we know which one was used?
    # The function `_find_best_position_in_bin` returned a position (x,y,z).
    # We can assume the box was placed in the free space that contained (x,y,z) at its corner.
    # Since we placed it at the corner of that free space, we can identify the free space by its coordinates.
    
    # Let's refine `_update_free_spaces` to take the index of the used free space or the specific space.
    # For simplicity, let's assume we pass the specific free space that was used.
    
    # Re-implementing `_pack_bfd` to pass the specific space.
    
    return new_free_spaces

# Refactor `_pack_bfd` to use the improved logic

def _pack_bfd(boxes, container, sorted_indices, deadline):
    W, H, D = container
    bins = []
    placements_map = {}
    
    bins.append([(0, 0, 0, W, H, D)])
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        
        if time.time() > deadline:
            return None
            
        best_bin_idx = -1
        best_pos = None
        best_space_idx = -1
        best_fit_score = float('inf')
        
        for b_idx, free_spaces in enumerate(bins):
            for s_idx, space in enumerate(free_spaces):
                fx, fy, fz, fw, fh, fd = space
                if w <= fw and h <= fh and d <= fd:
                    score = (fw - w) + (fh - h) + (fd - d)
                    if score < best_fit_score:
                        best_fit_score = score
                        best_bin_idx = b_idx
                        best_pos = (fx, fy, fz)
                        best_space_idx = s_idx
        
        if best_bin_idx == -1:
            best_bin_idx = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
            best_space_idx = 0 # Will add initial space
            # Initialize new bin
            bins[best_bin_idx] = [(0, 0, 0, W, H, D)]
            
        x, y, z = best_pos
        free_spaces = bins[best_bin_idx]
        
        # Get the specific space that was used
        used_space = free_spaces[best_space_idx]
        fx, fy, fz, fw, fh, fd = used_space
        
        # Remove the used space
        free_spaces.pop(best_space_idx)
        
        # Add new sub-spaces
        if x + w < fx + fw:
            free_spaces.append((x + w, y, z, fx + fw - (x + w), fh, fd))
        if y + h < fy + fh:
            free_spaces.append((x, y + h, z, w, fy + fh - (y + h), d))
        if z + d < fz + fd:
            free_spaces.append((x, y, z + d, w, h, fz + fd - (z + d)))
            
        placements_map[idx] = (best_bin_idx, x, y, z, w, h, d)
        
    placements = [placements_map[i] for i in range(len(boxes))]
    return placements

def _pack_shelf(boxes, container):
    W, H, D = container
    placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
    for w, h, d in boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
        placements.append((b, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return placements

# EVOLVE-BLOCK-END
