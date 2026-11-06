"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def pack(boxes, container):
    """
    3D Bin Packing using a Best-Fit Decreasing (BFD) heuristic with
    a simplified empty-space management per bin.
    """
    W, H, D = container
    container_volume = W * H * D
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (x[1][0]*x[1][1]*x[1][2], max(x[1])), reverse=True)
    
    # Each bin stores a list of placed boxes: (x, y, z, w, h, d)
    bins = []
    
    def can_place_in_bin(bin_placements, x, y, z, w, h, d):
        # Check bounds
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with all placed boxes in this bin
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Overlap if intervals intersect in all 3 dimensions
            if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
                return False
        return True
    
    def find_position(bin_placements, w, h, d):
        """
        Find a position in the bin that minimizes the "worst" coordinate sum.
        We try to pack towards the origin (0,0,0) to keep space contiguous.
        We use a simple grid-based search or heuristic:
        Try placing at the lowest possible z, then lowest y, then lowest x.
        To do this efficiently, we can iterate through potential z, y, x.
        However, a full grid search is O(W*H*D).
        
        A better heuristic:
        1. For each existing box, consider positions just above it, just in front, just to the side.
        2. Also consider the origin.
        
        Let's collect candidate positions:
        - (0, 0, 0)
        - For each box (px, py, pz, pw, ph, pd):
          - (px, py, pz + pd)  # on top
          - (px, py + ph, pz)  # in front
          - (px + pw, py, pz)  # to the side
        - Also try shifting these to be as low as possible?
        Actually, the "Best Fit" usually implies minimizing the sum of distances to the closest boundaries (x, y, z axes).
        
        Let's generate candidates and filter by validity, then pick the one with min(x+y+z).
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Positions adjacent to existing boxes
            candidates.add((px, py, pz + pd))
            candidates.add((px, py + ph, pz))
            candidates.add((px + pw, py, pz))
            
            # Also consider corners of the "empty space" defined by the union?
            # A simple improvement: try to fill gaps.
            # But the above 3 neighbors + origin is a standard First-Fit/Best-Fit heuristic base.
            
            # Let's also try to align with the max extent of existing boxes in each dimension?
            # No, that's more complex.
            
        # Filter valid candidates
        valid_candidates = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place_in_bin(bin_placements, x, y, z, w, h, d):
                    valid_candidates.append((x, y, z))
        
        if not valid_candidates:
            return None
        
        # Best Fit: minimize x + y + z (closest to origin)
        # Ties can be broken by minimizing max(x, y, z) or similar, but sum is standard.
        valid_candidates.sort(key=lambda pos: pos[0] + pos[1] + pos[2])
        return valid_candidates[0]

    # Place each box
    for orig_idx, (w, h, d) in indexed_boxes:
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        for i, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score: sum of coordinates (lower is better for BFD)
                score = x + y + z
                if score < best_score:
                    best_score = score
                    best_bin_idx = i
                    best_pos = pos
        
        if best_bin_idx == -1:
            # Open new bin
            bins.append([])
            best_bin_idx = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin_idx].append((x, y, z, w, h, d))
        
    # Construct result
    # We need to map back to original indices
    # bins[i] contains (x, y, z, w, h, d) but we lost the original index in the loop above?
    # Wait, the loop `for orig_idx, (w, h, d) in indexed_boxes:`
    # We need to store the placement for each original index.
    
    # Let's redo the placement logic to keep track of placements per original index.
    placements = [None] * len(boxes)
    
    # Re-initialize bins
    bins = []
    
    for orig_idx, (w, h, d) in indexed_boxes:
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        for i, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                score = x + y + z
                if score < best_score:
                    best_score = score
                    best_bin_idx = i
                    best_pos = pos
        
        if best_bin_idx == -1:
            bins.append([])
            best_bin_idx = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin_idx].append((x, y, z, w, h, d))
        placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        
    return placements

# EVOLVE-BLOCK-END
