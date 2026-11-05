"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing: place boxes into numbered bins to minimize mean B + 0.1*(1 - last-bin fill).
    Uses Best-Fit Decreasing with a grid-based collision check for speed.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best Fit Decreasing)
    # Keep original index to map back
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Each bin maintains a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # list of lists
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box at (x,y,z) with dims (w,h,d) fits in the bin without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check overlap: two boxes overlap if they intersect in all 3 dimensions
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """
        Find the best position for a box in a given bin.
        Uses a heuristic: try to place at the lowest z, then lowest y, then lowest x
        that doesn't overlap. Also tries to maximize contact with existing boxes.
        """
        best_pos = None
        best_score = float('inf')
        
        # Candidate positions: try each placed box's "top", "front", "right" faces
        # plus the origin
        candidates = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            candidates.append((px, py, pz + pd))  # on top
            candidates.append((px + pw, py, pz))  # to the right
            candidates.append((px, py + ph, pz))  # in front
        
        # Also try some grid positions for better packing
        # But to keep it fast, we'll stick with face-based candidates
        
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if not can_place(bin_boxes, cx, cy, cz, w, h, d):
                continue
            # Score: prefer lower z, then lower y, then lower x
            score = cz * 10000 + cy * 100 + cx
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos
    
    # Place boxes
    placements = [None] * n
    
    for idx, (orig_i, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            # Emergency: place remaining boxes in a new bin each (or just return what we have)
            # But we must place all boxes. Use a simple fallback.
            for rem_idx in range(idx, n):
                oi, rw, rh, rd, rv = indexed_boxes[rem_idx]
                # Try to fit in existing bins, else new bin
                placed = False
                for b_idx in range(len(bins)):
                    pos = find_position(bins[b_idx], rw, rh, rd)
                    if pos is not None:
                        bins[b_idx].append((pos[0], pos[1], pos[2], rw, rh, rd))
                        placements[oi] = (b_idx, pos[0], pos[1], pos[2], rw, rh, rd)
                        placed = True
                        break
                if not placed:
                    b_idx = len(bins)
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placements[oi] = (b_idx, 0, 0, 0, rw, rh, rd)
            break
        
        placed = False
        
        # Try to place in existing bins, best-fit (minimize wasted space)
        best_bin_idx = -1
        best_pos = None
        best_waste = float('inf')
        
        for b_idx in range(len(bins)):
            pos = find_position(bins[b_idx], w, h, d)
            if pos is not None:
                # Calculate "waste" or "fit" score
                # Prefer bins that are already used, and positions that minimize max coordinate
                cx, cy, cz = pos
                # A simple heuristic: minimize the sum of distances from origin
                waste = cz * 1000 + cy * 100 + cx
                if waste < best_waste:
                    best_waste = waste
                    best_bin_idx = b_idx
                    best_pos = pos
        
        if best_bin_idx >= 0:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placements[orig_i] = (best_bin_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d)
        else:
            # Open a new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[orig_i] = (b_idx, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
