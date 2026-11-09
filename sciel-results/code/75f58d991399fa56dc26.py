import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    # Backtracking with unit propagation
    # State: color of each vertex (0, 1, or -1 for unassigned)
    # Constraints: each edge must have both colors.
    # Unit propagation: if an edge has 4 vertices of color 0, the 5th MUST be 1.
    # If an edge has 5 vertices of color 0, conflict.
    
    colors = [-1] * n
    # Pre-compute edge indices
    edge_indices = [tuple(e) for e in edges]
    
    def propagate():
        # Simple propagation loop
        changed = True
        while changed:
            changed = False
            for e in edge_indices:
                # Count colors in edge
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                unassigned = [v for v in e if colors[v] == -1]
                
                if c0 == 5:
                    return False # Conflict
                if c1 == 5:
                    return False # Conflict
                
                if c0 == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 1
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 0
                    changed = True
        return True

    def backtrack(idx):
        # Find first unassigned vertex
        # Simple: just try 0 and 1 for each vertex in order
        # Optimization: pick vertex with highest degree or most constraints
        # For speed, we just iterate 0..n-1
        for i in range(n):
            if colors[i] == -1:
                # Try 0
                old_c = colors[i]
                colors[i] = 0
                if propagate():
                    if backtrack(idx + 1):
                        return True
                colors[i] = old_c
                
                # Try 1
                colors[i] = 1
                if propagate():
                    if backtrack(idx + 1):
                        return True
                colors[i] = old_c
                return False
        # All assigned
        return True

    if not propagate():
        return False
    return backtrack(0)

def construct():
    start_time = time.time()
    time_limit = 10.0
    
    best_edges = None
    best_n = 9
    best_edge_list = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edge_list)
    
    # Try to find a smaller one
    # Search space: n=10, 11, 12
    for n in [10, 11, 12]:
        if time.time() - start_time > time_limit:
            break
        
        # Random restarts
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            edges = set()
            # Start with a random set of edges
            # We want few edges, so we start small and add?
            # Or start with all and remove?
            # Removing is safer for validity, but slow.
            # Adding is faster to find non-2-colorable, but might be dense.
            
            # Let's try adding random edges until not 2-colorable
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            current_edges = []
            for e in all_edges:
                current_edges.append(e)
                if is_2_colorable(n, current_edges):
                    continue
                else:
                    # Found a non-2-colorable hypergraph
                    if len(current_edges) < best_count:
                        best_count = len(current_edges)
                        best_n = n
                        best_edges = [list(x) for x in current_edges]
                    break # Restart with new shuffle
            
            # If we didn't break, it's 2-colorable with all edges (impossible for n>=5? 
            # Complete hypergraph is not 2-colorable for n>=5?
            # K_5^(5) has 1 edge. 2-colorable? Yes.
            # K_6^(5) has 6 edges.
            # K_9^(5) is not 2-colorable.
            # So for n=10, it will eventually become not 2-colorable.
            
    return {'n': best_n, 'edges': best_edges}
