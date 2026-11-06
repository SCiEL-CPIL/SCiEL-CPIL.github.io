import random
import time
import sys
from itertools import combinations

def is_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Returns False if it IS 2-colorable.
    Uses DPLL with Unit Propagation and state copying.
    """
    if not edges:
        return False
        
    edge_tuples = [tuple(e) for e in edges]
    E = len(edge_tuples)
    
    # Precompute vertex to edges mapping
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_tuples):
        for v in e:
            vertex_edges[v].append(i)
            
    # Initial state
    init_color = [-1] * n
    init_edge_counts = [[0, 0, 5] for _ in range(E)]
    
    def propagate(color, edge_counts):
        """
        Returns (success, new_color, new_edge_counts)
        success: True if no contradiction, False otherwise.
        """
        # We copy the state to avoid side effects
        c = color[:]
        ec = [list(x) for x in edge_counts]
        
        changed = True
        while changed:
            changed = False
            for i in range(E):
                c0, c1, c_un = ec[i]
                
                if c_un == 0:
                    if c0 == 5 or c1 == 5:
                        return False, c, ec
                elif c_un == 1:
                    if c0 == 4:
                        # Must be 1
                        for v in edge_tuples[i]:
                            if c[v] == -1:
                                if c[v] != -1 and c[v] != 1:
                                    return False, c, ec
                                if c[v] == -1:
                                    c[v] = 1
                                    ec[i][1] += 1
                                    ec[i][2] -= 1
                                    for ei in vertex_edges[v]:
                                        if ei != i:
                                            ec[ei][1] += 1
                                            ec[ei][2] -= 1
                                    changed = True
                                break
                    elif c1 == 4:
                        # Must be 0
                        for v in edge_tuples[i]:
                            if c[v] == -1:
                                if c[v] != -1 and c[v] != 0:
                                    return False, c, ec
                                if c[v] == -1:
                                    c[v] = 0
                                    ec[i][0] += 1
                                    ec[i][2] -= 1
                                    for ei in vertex_edges[v]:
                                        if ei != i:
                                            ec[ei][0] += 1
                                            ec[ei][2] -= 1
                                    changed = True
                                break
        return True, c, ec

    def backtrack(color, edge_counts):
        # Check if all assigned
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
                
        if v == -1:
            # Check for contradictions in fully assigned edges (should be caught by propagate, but just in case)
            # Actually, if propagate returned True, no contradiction.
            return True
            
        # Branch on v
        # Try 0
        new_color_0 = color[:]
        new_ec_0 = [list(x) for x in edge_counts]
        new_color_0[v] = 0
        for ei in vertex_edges[v]:
            new_ec_0[ei][0] += 1
            new_ec_0[ei][2] -= 1
            
        success, c0, ec0 = propagate(new_color_0, new_ec_0)
        if success and backtrack(c0, ec0):
            return True
            
        # Try 1
        new_color_1 = color[:]
        new_ec_1 = [list(x) for x in edge_counts]
        new_color_1[v] = 1
        for ei in vertex_edges[v]:
            new_ec_1[ei][1] += 1
            new_ec_1[ei][2] -= 1
            
        success, c1, ec1 = propagate(new_color_1, new_ec_1)
        if success and backtrack(c1, ec1):
            return True
            
        return False

    # Initial propagation
    success, c_init, ec_init = propagate(init_color, init_edge_counts)
    if not success:
        return True # Contradiction immediately, so NOT 2-colorable
        
    return not backtrack(c_init, ec_init)

def construct():
    start_time = time.time()
    best_edges = None
    best_count = float('inf')
    best_n = 0
    
    for n in range(9, 15):
        all_edges = list(combinations(range(n), 5))
        
        for restart in range(50):
            if time.time() - start_time > 110:
                if best_edges:
                    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
                else:
                    return {'n': 0, 'edges': []}
                
            current_edges = set(all_edges)
            shuffled_edges = all_edges[:]
            random.shuffle(shuffled_edges)
            
            for e in shuffled_edges:
                if e not in current_edges:
                    continue
                
                current_edges.remove(e)
                
                if is_non_2_colorable(n, list(current_edges)):
                    if len(current_edges) < best_count:
                        best_count = len(current_edges)
                        best_n = n
                        best_edges = list(current_edges)
                else:
                    current_edges.add(e)
                    
    if best_edges:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    else:
        # Fallback
        return {'n': 9, 'edges': []}
