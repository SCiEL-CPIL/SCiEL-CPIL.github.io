import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    colors = [-1] * n
    edge_indices = [tuple(e) for e in edges]
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edge_indices:
                c0 = 0
                c1 = 0
                unassigned = -1
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned = v
                
                if c0 == 4 and unassigned != -1 and colors[unassigned] == -1:
                    colors[unassigned] = 1
                    changed = True
                elif c1 == 4 and unassigned != -1 and colors[unassigned] == -1:
                    colors[unassigned] = 0
                    changed = True
                elif c0 == 5 or c1 == 5:
                    return False
        return True

    def backtrack(v):
        if v == n:
            return True
        
        if colors[v] == -1:
            colors[v] = 0
            if propagate():
                if backtrack(v + 1):
                    return True
            colors[v] = -1
            
            colors[v] = 1
            if propagate():
                if backtrack(v + 1):
                    return True
            colors[v] = -1
            
        return False

    if not propagate():
        return False
        
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try n from 15 to 18
    for n in range(15, 19):
        if time.time() > deadline - 10.0:
            break
        
        all_edges = list(combinations(range(n), 5))
        
        # Try multiple random deletion orders
        attempts = 15
        for _ in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            edges = all_edges[:]
            random.shuffle(edges)
            
            current_edges = edges[:]
            
            # Greedy deletion
            i = 0
            while i < len(current_edges):
                # Check time
                if time.time() > deadline - 2.0:
                    break
                    
                # Try removing current_edges[i]
                test_edges = current_edges[:i] + current_edges[i+1:]
                
                # Check if still non-2-colorable
                if not is_2_colorable(n, test_edges):
                    # Can remove
                    current_edges = test_edges
                    # Do not increment i
                else:
                    i += 1
                    
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = [list(e) for e in current_edges]
                best_n = n
                
                # If we found something very small, we can stop
                if best_count <= 30:
                    return {'n': best_n, 'edges': best_edges}
    
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
