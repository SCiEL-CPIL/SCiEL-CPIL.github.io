import itertools
import time
import random

def _is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    colors = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for edge in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in edge:
                    c = colors[v]
                    if c == 0:
                        c0 += 1
                    elif c == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 1
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 0
                    changed = True
        return True

    def backtrack():
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
        
        if v == -1:
            return True
        
        saved_colors = colors[:]
        
        colors[v] = 0
        if propagate():
            if backtrack():
                return True
        
        colors[:] = saved_colors
        
        colors[v] = 1
        if propagate():
            if backtrack():
                return True
        
        colors[:] = saved_colors
        return False

    if not propagate():
        return False
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    current_edges = set(tuple(e) for e in all_edges)
    best_edges = set(current_edges)
    best_count = len(best_edges)
    
    edge_list = list(current_edges)
    random.shuffle(edge_list)
    
    i = 0
    while time.time() < deadline:
        if i >= len(edge_list):
            random.shuffle(edge_list)
            i = 0
            if best_count <= 55:
                break
        
        edge_to_remove = edge_list[i]
        i += 1
        
        current_edges.discard(edge_to_remove)
        
        if not _is_2_colorable(n, list(current_edges)):
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)
        else:
            current_edges.add(edge_to_remove)
        
        if time.time() > deadline:
            break
    
    return {'n': n, 'edges': [list(e) for e in best_edges]}
