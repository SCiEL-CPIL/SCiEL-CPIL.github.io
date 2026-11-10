import random
import time

def construct():
    d = 7
    n = 1 << d
    # Generate all edges
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
    
    # Map edge to index for fast lookup? Or just use set of tuples.
    # Set of tuples is fine.
    
    # Precompute for each edge, the 6 squares it belongs to.
    # For each square, store the other 3 edges.
    edge_info = {}
    for u, v in edges:
        # Find bit i
        diff = u ^ v
        i = diff.bit_length() - 1
        others = []
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # The other 3 edges are (u, u2), (v, v2), (u2, v2)
            # Normalize them
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            others.append((e1, e2, e3))
        edge_info[(u, v)] = others

    best_edges = 0
    best_set = set()
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    while time.time() < deadline:
        selected = set()
        random.shuffle(edges)
        
        for e in edges:
            # Check if adding e creates a C4
            ok = True
            for e1, e2, e3 in edge_info[e]:
                if e1 in selected and e2 in selected and e3 in selected:
                    ok = False
                    break
            if ok:
                selected.add(e)
        
        if len(selected) > best_edges:
            best_edges = len(selected)
            best_set = selected.copy()
            # If we hit the known best, we might want to stop or continue to see if we can exceed?
            # 304 is best known. If we hit 304, we can keep searching for 305-308.
            # But 308 is upper bound.
            # Let's keep searching.

    return [list(e) for e in best_set]
