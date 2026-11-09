"""Improved 3D container loading with rotation and skyline-based placement."""
import time
import itertools
import heapq

def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110  # safety margin under 120s
    
    # Generate all 6 rotations for each box
    def rotations(w, h, d):
        return list(set(itertools.permutations((w, h, d))))
    
    # Sort boxes by volume descending to place larger boxes first
    box_indices = list(range(len(boxes)))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in box_indices]
    sorted_indices = sorted(box_indices, key=lambda i: box_volumes[i], reverse=True)
    
    # Skyline representation: list of (x, y, z, w, h, d) for placed boxes
    # We'll use a simple approach: track occupied space using a grid or interval tree
    # For efficiency, use a 3D occupancy check with intervals
    
    # Use a simpler approach: try to place each box in the "best" position
    # Best position defined as minimizing the max coordinate (bottom-left-front bias)
    
    placements = [None] * len(boxes)
    
    # Track occupied regions as a list of (x, y, z, w, h, d)
    occupied = []
    
    def can_place(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with all occupied boxes
        for ox, oy, oz, ow, oh, od in occupied:
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return False
        return True
    
    def find_best_position(w, h, d):
        """Find the best position for a box with given dimensions."""
        # Try to place at bottom-left-front corner, then sweep
        # For efficiency, try a limited set of candidate positions
        
        # Candidate x positions: 0 and right edges of existing boxes
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for ox, oy, oz, ow, oh, od in occupied:
            if ox + ow <= W:
                x_candidates.append(ox + ow)
            if oy + oh <= H:
                y_candidates.append(oy + oh)
            if oz + od <= D:
                z_candidates.append(oz + od)
        
        x_candidates = sorted(set(x for x in x_candidates if x + w <= W))
        y_candidates = sorted(set(y for y in y_candidates if y + h <= H))
        z_candidates = sorted(set(z for z in z_candidates if z + d <= D))
        
        best_pos = None
        best_score = float('inf')
        
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if can_place(x, y, z, w, h, d):
                        # Score: prefer lower z, then lower y, then lower x
                        score = z * W * H + y * W + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
        
        bw, bh, bd = boxes[idx]
        rots = rotations(bw, bh, bd)
        
        placed = False
        for rw, rh, rd in rots:
            pos = find_best_position(rw, rh, rd)
            if pos is not None:
                x, y, z = pos
                placements[idx] = (0, x, y, z, rw, rh, rd)
                occupied.append((x, y, z, rw, rh, rd))
                placed = True
                break
        
        if not placed:
            placements[idx] = (-1, 0, 0, 0, bw, bh, bd)
    
    # Fill any remaining None placements (shouldn't happen, but safety)
    for i in range(len(boxes)):
        if placements[i] is None:
            bw, bh, bd = boxes[i]
            placements[i] = (-1, 0, 0, 0, bw, bh, bd)
    
    return placements
