"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2col(n, edges, timeout=10.0):
    start = time.time()
    m = len(edges)
    if m == 0:
        return False  # Empty hypergraph is 2-colourable
    
    v_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_edges[v].append(i)
    
    # Vertex ordering: try vertices with most edges first
    order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i
    
    assignment = [-1] * n
    n_assigned = 0
    
    def backtrack():
        nonlocal n_assigned
        if time.time() - start > timeout:
            return 'timeout'
        if n_assigned == n:
            return True  # Found valid 2-colouring
        
        # Pick next unassigned vertex with most edges
        best_v = -1
        best_count = -1
        for v in order:
            if assignment[v] == -1:
                if len(v_edges[v]) > best_count:
                    best_count = len(v_edges[v])
                    best_v = v
        if best_v == -1:
            return True
        
        v = best_v
        for c in [0, 1]:
            assignment[v] = c
            n_assigned += 1
            ok = True
            for ei in v_edges[v]:
                e = edges[ei]
                c0 = c1 = 0
                for x in e:
                    if assignment[x] == 0: c0 += 1
                    elif assignment[x] == 1: c1 += 1
                if c0 == 5 or c1 == 5:
                    ok = False
                    break
            if ok:
                result = backtrack()
                if result == 'timeout':
                    return 'timeout'
                if result:
                    return True
            n_assigned -= 1
        assignment[v] = -1
        return False
    
    result = backtrack()
    if result == 'timeout':
        return None
    return not result

def construct():
    deadline = time.time() + 140
    best_n = None
    best_edges = None
    best_count = 201
    
    random.seed(42)
    
    # Try random search for small non-2-colourable 5-uniform hypergraphs
    for n in [10, 12, 14, 16, 18, 20, 22, 26]:
        if time.time() > deadline - 15:
            break
        
        all_edges = list(combinations(range(n), 5))
        total = len(all_edges)
        
        for m in range(28, min(60, total + 1)):
            if time.time() > deadline - 15:
                break
            if m >= best_count:
                continue
            
            for trial in range(4):
                if time.time() > deadline - 15:
                    break
                
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:m]]
                
                result = _is_not_2col(n, edges, timeout=4.0)
                if result is True:
                    if m < best_count:
                        best_count = m
                        best_n = n
                        best_edges = edges
                    break
    
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: complete 5-uniform on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
