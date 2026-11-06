import time
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)

    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    deadline = start_time + 115.0

    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)

    placements = [None] * n
    placed_boxes = []  # List of (x, y, z, w, h, d)

    def check_overlap(x, y, z, w, h, d):
        # Check bounds
        if x < 0 or y < 0 or z < 0:
            return True
        if x + w > W or y + h > H or z + d > D:
            return True
        
        # Check overlap with placed boxes
        for px, py, pz, pw, ph, pd in placed_boxes:
            # Separating Axis Theorem for AABBs
            if x < px + pw and x + w > px and \
               y < py + ph and y + h > py and \
               z < pz + pd and z + d > pz:
                return True
        return False

    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break

        orients = list(set(itertools.permutations([w, h, d])))
        
        best_pos = None
        best_orient = None
        
        # Generate candidate positions
        # 1. Origin
        candidates = [(0, 0, 0)]
        
        # 2. Adjacent to placed boxes
        for px, py, pz, pw, ph, pd in placed_boxes:
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            # Also try corners relative to placed boxes? 
            # For simplicity, just the 3 face-adjacent spots and origin.
            # This is a common heuristic.

        for ow, oh, od in orients:
            for cx, cy, cz in candidates:
                if not check_overlap(cx, cy, cz, ow, oh, od):
                    # Valid placement found
                    # We can pick the first one, or the one with lowest (z, y, x)
                    # Let's pick the one with lowest (z, y, x) to pack tightly
                    if best_pos is None or (cz, cy, cx) < (best_pos[2], best_pos[1], best_pos[0]):
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placements[idx] = (0, cx, cy, cz, ow, oh, od)
            placed_boxes.append((cx, cy, cz, ow, oh, od))
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any remaining None
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
