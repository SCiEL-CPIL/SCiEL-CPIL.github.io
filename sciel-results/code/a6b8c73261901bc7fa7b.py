import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending
    indexed_boxes = [(w * h * d, i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(reverse=True)
    
    placed = [] # List of (x, y, z, w, h, d)
    candidates = [(0, 0, 0)]
    results = [None] * len(boxes)
    
    def get_orientations(w, h, d):
        # Unique orientations
        return set(itertools.permutations([w, h, d]))
    
    def fits(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with placed boxes
        for px, py, pz, pw, ph, pd in placed:
            # No overlap if separated in any dimension
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    for vol, idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
            
        best_placement = None
        best_cost = float('inf')
        
        orientations = get_orientations(w, h, d)
        
        # Sort candidates by cost to potentially find good ones early?
        # Or just iterate all.
        for cx, cy, cz in candidates:
            for ow, oh, od in orientations:
                if not fits(cx, cy, cz, ow, oh, od):
                    continue
                # Cost function: prioritize low z, then low y, then low x
                # Using a large multiplier for z ensures we fill layers
                cost = cz * 10000 + cy * 100 + cx
                if cost < best_cost:
                    best_cost = cost
                    best_placement = (cx, cy, cz, ow, oh, od)
        
        if best_placement:
            bx, by, bz, bw, bh, bd = best_placement
            placed.append((bx, by, bz, bw, bh, bd))
            results[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Generate new candidates
            # We add positions adjacent to the placed box
            new_cands = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd),
                (bx + bw, by + bh, bz),
                (bx + bw, by, bz + bd),
                (bx, by + bh, bz + bd),
                (bx + bw, by + bh, bz + bd)
            ]
            
            # Add to candidates, avoiding duplicates
            for nc in new_cands:
                if nc not in candidates:
                    candidates.append(nc)
        else:
            results[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill remaining results with -1 if not processed due to timeout
    for i in range(len(boxes)):
        if results[i] is None:
            w, h, d = boxes[i]
            results[i] = (-1, 0, 0, 0, w, h, d)
            
    return results
