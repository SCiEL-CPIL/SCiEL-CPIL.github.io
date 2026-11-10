import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    # Initial Graph: Grötzsch (0-10)
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    init_edges = [list(e) for e in cycle]
    for u, w in cycle:
        init_edges += [[u, 5 + w], [w, 5 + u]]
    init_edges += [[5 + u, 10] for u in range(5)]
    
    best_edges = [list(e) for e in init_edges]
    best_count = len(best_edges)
    
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    while time.time() < deadline:
        # Build bitmask adjacency from best_edges
        neigh = [0] * 16
        for u, v in best_edges:
            neigh[u] |= (1 << v)
            neigh[v] |= (1 << u)
            
        candidates = [p for p in all_pairs if not (neigh[p[0]] & (1 << p[1]))]
        random.shuffle(candidates)
        
        current_count = len(best_edges)
        added = False
        for u, v in candidates:
            if time.time() > deadline:
                break
            # Check triangle-free: no common neighbor
            if not (neigh[u] & neigh[v]):
                # Add edge
                neigh[u] |= (1 << v)
                neigh[v] |= (1 << u)
                best_edges.append([u, v])
                current_count += 1
                added = True
        
        if current_count > best_count:
            best_count = current_count
        elif not added:
            # No improvement in this pass.
            # If we have done many passes with no improvement, break?
            # For now, just continue.
            pass

    return best_edges
