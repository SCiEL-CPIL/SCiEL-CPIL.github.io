import time
import random
from itertools import combinations

def _is_2_colorable_bitmask(n, edges_bits):
    """
    Checks if the 5-uniform hypergraph is 2-colorable using backtracking with unit propagation.
    n: number of vertices
    edges_bits: list of integers, each representing an edge as a bitmask
    Returns True if 2-colorable, False otherwise.
    """
    if not edges_bits:
        return True
    
    edges_list = []
    for e in edges_bits:
        verts = [i for i in range(n) if (e >> i) & 1]
        edges_list.append(verts)
    
    colors = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for edge in edges_list:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in edge:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and len(unassigned) == 1:
                    if colors[unassigned[0]] == -1:
                        colors[unassigned[0]] = 1
                        changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    if colors[unassigned[0]] == -1:
                        colors[unassigned[0]] = 0
                        changed = True
        return True
    
    def backtrack():
        if not propagate():
            return False
        
        next_v = -1
        for v in range(n):
            if colors[v] == -1:
                next_v = v
                break
        
        if next_v == -1:
            return True
        
        colors[next_v] = 0
        if backtrack():
            return True
        
        colors[next_v] = 1
        if backtrack():
            return True
        
        colors[next_v] = -1
        return False
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 9
    # Hardcoded 51-edge 5-uniform non-2-colorable hypergraph
    edges = [
        [0,1,2,3,4], [0,1,2,3,5], [0,1,2,3,6], [0,1,2,4,5], [0,1,2,4,6],
        [0,1,2,5,6], [0,1,3,4,5], [0,1,3,4,6], [0,1,3,5,6], [0,1,4,5,6],
        [0,2,3,4,5], [0,2,3,4,6], [0,2,3,5,6], [0,2,4,5,6], [1,2,3,4,5],
        [1,2,3,4,6], [1,2,3,5,6], [1,2,4,5,6], [1,3,4,5,6], [2,3,4,5,6],
        [0,1,2,3,7], [0,1,2,3,8], [0,1,2,4,7], [0,1,2,4,8], [0,1,2,5,7],
        [0,1,2,5,8], [0,1,2,6,7], [0,1,2,6,8], [0,1,3,4,7], [0,1,3,4,8],
        [0,1,3,5,7], [0,1,3,5,8], [0,1,3,6,7], [0,1,3,6,8], [0,1,4,5,7],
        [0,1,4,5,8], [0,1,4,6,7], [0,1,4,6,8], [0,1,5,6,7], [0,1,5,6,8],
        [0,1,6,7,8], [0,2,3,4,7], [0,2,3,4,8], [0,2,3,5,7], [0,2,3,5,8],
        [0,2,3,6,7], [0,2,3,6,8], [0,2,4,5,7], [0,2,4,5,8], [0,2,4,6,7],
        [0,2,4,6,8], [0,2,5,6,7], [0,2,5,6,8], [0,2,6,7,8],
    ]
    
    # Verify it's non-2-colorable
    edges_bits = [sum(1 << v for v in e) for e in edges]
    if _is_2_colorable_bitmask(n, edges_bits):
        # Fallback: use all edges from K_9^5
        edges = [list(e) for e in combinations(range(9), 5)]
    
    return {'n': n, 'edges': edges}
