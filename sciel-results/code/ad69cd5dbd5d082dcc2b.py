"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges):
    """
    Check if hypergraph is NOT 2-colorable.
    Returns True if NOT 2-colorable.
    """
    if not edges:
        return False
        
    edge_tuples = [tuple(e) for e in edges]
    m = len(edge_tuples)
    
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_tuples):
        for v in e:
            vert_edges[v].append(i)
            
    assignment = [-1] * n
    c0 = [0] * m
    c1 = [0] * m
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_tuples):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                
                if c0[i] == 4:
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 1
                            c1[i] += 1
                            for ei in vert_edges[v]:
                                if ei != i:
                                    c1[ei] += 1
                                    if c1[ei] == 5:
                                        return False
                            changed = True
                            break
                elif c1[i] == 4:
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 0
                            c0[i] += 1
                            for ei in vert_edges[v]:
                                if ei != i:
                                    c0[ei] += 1
                                    if c0[ei] == 5:
                                        return False
                            changed = True
                            break
        return True

    def backtrack():
        idx = -1
        for i in range(n):
            if assignment[i] == -1:
                idx = i
                break
        
        if idx == -1:
            for i in range(m):
                if c0[i] == 5 or c1[i] == 5:
                    return False
            return True
            
        # Branch 0
        assignment[idx] = 0
        affected = []
        for ei in vert_edges[idx]:
            c0[ei] += 1
            affected.append(ei)
            
        valid = True
        for ei in affected:
            if c0[ei] == 5:
                valid = False
                break
                
        if valid and propagate():
            if backtrack():
                return True
                
        assignment[idx] = -1
        for ei in affected:
            c0[ei] -= 1
            
        # Branch 1
        assignment[idx] = 1
        affected = []
        for ei in vert_edges[idx]:
            c1[ei] += 1
            affected.append(ei)
            
        valid = True
        for ei in affected:
            if c1[ei] == 5:
                valid = False
                break
                
        if valid and propagate():
            if backtrack():
                return True
                
        assignment[idx] = -1
        for ei in affected:
            c1[ei] -= 1
            
        return False

    return not backtrack()

def construct():
    start_time = time.time()
    deadline = 140
    
    best_edges = None
    best_n = None
    
    def try_reduce(edges, n, time_limit):
        current_set = set(edges)
        shuffled = list(current_set)
        random.shuffle(shuffled)
        
        for e in shuffled:
            if time.time() > time_limit:
                break
            if e not in current_set:
                continue
            current_set.remove(e)
            if _is_not_2_colorable(n, list(current_set)):
                pass # Keep removed
            else:
                current_set.add(e)
                
        return list(current_set)
    
    # Try K_9^5
    n = 9
    all_edges = list(combinations(range(n), 5))
    if _is_not_2_colorable(n, all_edges):
        best_edges = all_edges
        best_n = n
        reduced = try_reduce(best_edges, best_n, deadline - 5)
        if len(reduced) < len(best_edges):
            best_edges = reduced
    
    # Try K_8^5
    if time.time() < deadline - 10:
        n = 8
        all_edges = list(combinations(range(n), 5))
        if _is_not_2_colorable(n, all_edges):
            reduced = try_reduce(all_edges, n, deadline - 5)
            if best_edges is None or len(reduced) < len(best_edges):
                best_edges = reduced
                best_n = n
                
    # Try K_7^5
    if time.time() < deadline - 10:
        n = 7
        all_edges = list(combinations(range(n), 5))
        if _is_not_2_colorable(n, all_edges):
            reduced = try_reduce(all_edges, n, deadline - 5)
            if best_edges is None or len(reduced) < len(best_edges):
                best_edges = reduced
                best_n = n

    if best_edges is None:
        # Fallback
        best_edges = list(combinations(range(9), 5))
        best_n = 9
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
