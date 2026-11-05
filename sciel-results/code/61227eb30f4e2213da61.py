import time
from itertools import combinations

def is_2_colorable_and_get_coloring(n, edges):
    color = [None] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                vals = [color[v] for v in e]
                if None in vals:
                    if vals.count(0) == 4:
                        idx = vals.index(None)
                        color[e[idx]] = 1
                        changed = True
                    elif vals.count(1) == 4:
                        idx = vals.index(None)
                        color[e[idx]] = 0
                        changed = True
                    elif vals.count(0) == 5 or vals.count(1) == 5:
                        return False
                else:
                    if vals.count(0) == 5 or vals.count(1) == 5:
                        return False
        return True

    def backtrack(idx):
        if idx == n:
            return True
        color[idx] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        color[idx] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        color[idx] = None
        return False

    if backtrack(0):
        return True, color
    return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    
    # Try blocking algorithm on n=11
    n = 11
    edges = []
    edge_set = set()
    
    while True:
        if time.time() > deadline:
            break
        
        colorable, coloring = is_2_colorable_and_get_coloring(n, edges)
        
        if not colorable:
            if len(edges) < len(best_edges):
                best_n = n
                best_edges = edges
            break
        
        # Find a monochromatic edge
        # Coloring is 0/1
        # Check all 5-sets
        found = False
        for e in combinations(range(n), 5):
            if e in edge_set:
                continue
            vals = [coloring[v] for v in e]
            if vals.count(0) == 5 or vals.count(1) == 5:
                edges.append(list(e))
                edge_set.add(e)
                found = True
                break
        
        if not found:
            # Should not happen if colorable
            break
    
    return {'n': best_n, 'edges': best_edges}
