import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    # edges: list of lists of 5 vertices
    if not edges:
        return True
        
    colors = [-1] * n
    
    # Localize for speed
    def dfs():
        # Propagation
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and unassigned:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 1
                        changed = True
                elif c1 == 4 and unassigned:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 0
                        changed = True
        
        # Find unassigned
        next_v = -1
        for i in range(n):
            if colors[i] == -1:
                next_v = i
                break
        
        if next_v == -1:
            return True
        
        # Try 0
        colors[next_v] = 0
        if dfs():
            return True
        # Try 1
        colors[next_v] = 1
        if dfs():
            return True
        
        colors[next_v] = -1
        return False

    return dfs()

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    # Bases to try
    bases = [
        (9, [list(e) for e in combinations(range(9), 5)]),
        (10, [list(e) for e in combinations(range(10), 5)])
    ]
    
    # Number of iterations
    # We can run many iterations because is_2_colorable is fast.
    
    for n, base_edges in bases:
        # Run multiple random greedy reductions
        max_iters = 300
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if best_count <= 51:
                break
                
            # Shuffle
            edges_copy = base_edges.copy()
            random.shuffle(edges_copy)
            
            # Greedy reduction
            active_set = set(map(tuple, edges_copy))
            
            for e in edges_copy:
                if time.time() > deadline:
                    break
                e_tuple = tuple(e)
                if e_tuple not in active_set:
                    continue
                    
                # Try to remove e_tuple
                # Graph is active_set - {e_tuple}
                # We need to check 2-colorability of this graph.
                
                # Build list for checker
                edges_to_check = [list(x) for x in active_set if x != e_tuple]
                
                if not is_2_colorable(n, edges_to_check):
                    # It is NOT 2-colorable, so we CAN remove it
                    active_set.remove(e_tuple)
                    
                    if len(active_set) < best_count:
                        best_count = len(active_set)
                        best_n = n
                        best_edges = [list(x) for x in active_set]
                        
                        if best_count <= 51:
                            return {'n': best_n, 'edges': best_edges}
                # else: it IS 2-colorable, so we MUST keep it.
                
    return {'n': best_n, 'edges': best_edges}
