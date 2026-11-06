import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    """
    color = [-1] * n
    
    # Convert edges to tuples for hashing if needed? No, just lists.
    # For speed, we can precompute edge indices.
    # But for small n, it's fine.
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                uncolored = -1
                for v in e:
                    if color[v] == 0:
                        c0 += 1
                    elif color[v] == 1:
                        c1 += 1
                    else:
                        uncolored = v
                
                if c0 == 4:
                    if color[uncolored] == -1:
                        color[uncolored] = 1
                        changed = True
                    elif color[uncolored] == 0:
                        return False
                elif c1 == 4:
                    if color[uncolored] == -1:
                        color[uncolored] = 0
                        changed = True
                    elif color[uncolored] == 1:
                        return False
                elif c0 == 5 or c1 == 5:
                    return False
        return True

    def backtrack():
        if not propagate():
            return False
        
        uncolored = [i for i in range(n) if color[i] == -1]
        if not uncolored:
            # All colored
            # Check monochromatic edges (redundant if propagate works correctly, but safe)
            for e in edges:
                c = color[e[0]]
                for v in e[1:]:
                    if color[v] != c:
                        break
                else:
                    return False
            return True
        
        # Heuristic: choose vertex with max degree in uncolored subgraph?
        # Or just first.
        # Let's try to pick a vertex that is in the most "tight" edges.
        # For simplicity, first uncolored.
        v = uncolored[0]
        
        # Try 0
        color[v] = 0
        if backtrack():
            return True
        color[v] = -1
        
        # Try 1
        color[v] = 1
        if backtrack():
            return True
        color[v] = -1
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Search for minimum m
    # m(5) is between 29 and 51
    for m in range(29, 55):
        if time.time() > deadline:
            break
        for n in range(9, 18):
            if time.time() > deadline:
                break
            if m > len(list(combinations(range(n), 5))):
                continue
                
            pair_start = time.time()
            pair_limit = 1.5 # seconds
            
            while time.time() < pair_start + pair_limit and time.time() < deadline:
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:m]]
                if not is_2_colorable(n, edges):
                    return {'n': n, 'edges': edges}
                    
    # Fallback to seed
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
