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
    
    # Candidate positions: list of (x, y, z)
    # Start with the origin
    candidates = [(0, 0, 0)]
    
    # Pre-compute unique orientations for each box
    # We will compute these on the fly to save memory, but caching helps if boxes repeat
    # For 60 boxes, on-the-fly is fine.
    
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
        
        # Sort candidates to prioritize lower z, then lower y, then lower x
        # This helps fill the container from the bottom-up
        # We don't need to sort the whole list every time, just iterate and keep best
        # But sorting candidates by (z, y, x) helps find a good spot faster if we break early?
        # Actually, we want the "best" fit. Let's define score as (z, y, x) to minimize height first.
        
        for ow, oh, od in orientations:
            # Check if this orientation even fits in the container
            if ow > W or oh > H or od > D:
                continue
                
            for cx, cy, cz in candidates:
                # Check if the box fits within the container bounds
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                    
                # Check overlap with placed boxes
                if check_overlap(cx, cy, cz, ow, oh, od):
                    continue
                    
                # If it fits, evaluate this position
                # We prefer positions with lower z, then lower y, then lower x
                score = (cz, cy, cx)
                
                if best_pos is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz, ow, oh, od)
                    
                    # Optional: Early termination if we find a perfect corner (0,0,0) or similar
                    # But since we are iterating, we can't easily break unless we sort candidates
                    # and know that subsequent candidates are worse. 
                    # Since we are checking all candidates for all orientations, 
                    # we can't break early without sorting. 
                    # However, 60 candidates * 6 orientations is small.
        
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
            
            # Add unique candidates
            for nc in new_candidates:
                if nc not in candidates:
                    candidates.append(nc)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill in any remaining boxes that weren't processed
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
