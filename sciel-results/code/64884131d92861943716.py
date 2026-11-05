import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Construct Paley tournament on p=67
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    # Build in-neighborhood masks for p=67
    # mask[v] has bit u set if u -> v
    masks = [0] * p
    for v in range(p):
        mask = 0
        for u in range(p):
            if u == v: continue
            if (v - u) % p in squares:
                mask |= (1 << u)
        masks[v] = mask
        
    # Helper to check if a subset of vertices (as a list) is valid
    def check_subset(vertices):
        n = len(vertices)
        if n < 4:
            return True
        # Map old index to new index? No, we just need to check property on original indices
        # restricted to 'vertices'.
        # The dominator must be in 'vertices'.
        # So we create a bitmask for 'vertices'.
        v_mask = 0
        for v in vertices:
            v_mask |= (1 << v)
            
        # Iterate all 4-subsets of 'vertices'
        for subset in itertools.combinations(vertices, 4):
            inter = masks[subset[0]] & masks[subset[1]] & masks[subset[2]] & masks[subset[3]]
            if (inter & v_mask) == 0:
                return False
        return True

    # Helper to extract and re-index orientation
    def build_orientation(vertices):
        n = len(vertices)
        # Map old index to new index
        old_to_new = {old: new for new, old in enumerate(vertices)}
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                u = vertices[i]
                v = vertices[j]
                # u -> v in original tournament?
                # Check mask: does u have bit v set? No, mask[v] has bit u if u->v.
                # So check if masks[v] has bit u.
                if (masks[v] >> u) & 1:
                    orientation.append(0) # u -> v
                else:
                    orientation.append(1) # v -> u
        return orientation

    # Try to reduce n by deleting vertices
    # Start with all 67 vertices
    all_vertices = list(range(p))
    
    # Try n=66 (delete 1)
    # Try each single deletion
    for i in range(p):
        if time.time() > deadline:
            break
        vertices = [v for v in all_vertices if v != i]
        if check_subset(vertices):
            orientation = build_orientation(vertices)
            return {'n': 66, 'orientation': orientation}
            
    # Try n=65 (delete 2)
    # Try random pairs
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        deleted = random.sample(all_vertices, 2)
        deleted_set = set(deleted)
        vertices = [v for v in all_vertices if v not in deleted_set]
        if check_subset(vertices):
            orientation = build_orientation(vertices)
            return {'n': 65, 'orientation': orientation}
            
    # Try n=64 (delete 3)
    for _ in range(200):
        if time.time() > deadline:
            break
        deleted = random.sample(all_vertices, 3)
        deleted_set = set(deleted)
        vertices = [v for v in all_vertices if v not in deleted_set]
        if check_subset(vertices):
            orientation = build_orientation(vertices)
            return {'n': 64, 'orientation': orientation}
            
    # Try n=63 (delete 4)
    for _ in range(200):
        if time.time() > deadline:
            break
        deleted = random.sample(all_vertices, 4)
        deleted_set = set(deleted)
        vertices = [v for v in all_vertices if v not in deleted_set]
        if check_subset(vertices):
            orientation = build_orientation(vertices)
            return {'n': 63, 'orientation': orientation}

    # Fallback to 67
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (masks[j] >> i) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
