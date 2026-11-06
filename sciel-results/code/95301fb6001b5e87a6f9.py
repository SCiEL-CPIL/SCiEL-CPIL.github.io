"""3D Container Loading - Best-Fit Decreasing with Candidate Points."""
import time
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 115  # seconds, leave margin for safety
    
    n = len(boxes)
    
    # Sort boxes by volume descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
    
    # Placed boxes: list of (x, y, z, w, h, d, original_index)
    placed_boxes = []
    
    # Candidate positions: set of (x, y, z)
    # Start with the origin
    candidates = {(0, 0, 0)}
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        orients = set(permutations([w, h, d]))
        return list(orients)
    
    placements = [None] * n
    
    def check_overlap(x, y, z, ow, oh, od):
        """Check if a box at (x,y,z) with dims (ow,oh,od) overlaps with any placed box."""
        for px, py, pz, pw, ph, pd, _ in placed_boxes:
            # Check if intervals overlap in all 3 dimensions
            # Overlap if: x < px+pw AND px < x+ow
            #             y < py+ph AND py < y+oh
            #             z < pz+pd AND pz < z+od
            if (x < px + pw and px < x + ow and
                y < py + ph and py < y + oh and
                z < pz + pd and pz < z + od):
                return True
        return False
    
    for idx in box_indices:
        if time.time() - t0 > DEADLINE:
            break
            
        w, h, d = boxes[idx]
        orientations = get_orientations(w, h, d)
        
        best_pos = None
        best_score = float('inf')
        
        # Sort candidates by (z, y, x) to prioritize lower positions
        # This allows early termination if we find a valid placement at the lowest possible position
        sorted_candidates = sorted(candidates, key=lambda c: (c[2], c[1], c[0]))
        
        found = False
        for cx, cy, cz in sorted_candidates:
            if found:
                break
            for ow, oh, od in orientations:
                # Check if this orientation even fits in the container
                if ow > W or oh > H or od > D:
                    continue
                    
                # Check if the box fits within the container bounds
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                    
                # Check overlap with placed boxes
                if check_overlap(cx, cy, cz, ow, oh, od):
                    continue
                    
                # If it fits, this is the best position (lowest z, then y, then x)
                best_pos = (cx, cy, cz, ow, oh, od)
                found = True
                break
        
        if best_pos is not None:
            bx, by, bz, bw, bh, bd = best_pos
            placements[idx] = (0, bx, by, bz, bw, bh, bd)
            placed_boxes.append((bx, by, bz, bw, bh, bd, idx))
            
            # Generate new candidate positions from the placed box
            # The 8 corners of the placed box
            new_candidates = [
                (bx, by, bz),
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx + bw, by + bh, bz),
                (bx, by, bz + bd),
                (bx + bw, by, bz + bd),
                (bx, by + bh, bz + bd),
                (bx + bw, by + bh, bz + bd),
            ]
            
            # Add unique candidates to the set
            for nc in new_candidates:
                candidates.add(nc)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill in any remaining boxes that weren't processed
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
