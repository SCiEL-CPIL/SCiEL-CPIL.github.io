import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110  # Leave 10s margin for 120s limit
    
    # Fast 2-coloring check
    def is_non_2_colorable(n, edges):
        # edges: list of tuples
        # Returns True if NOT 2-colorable
        
        # Precompute vertex to edge index mapping
        v_to_e = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_e[v].append(i)
        
        # State:
        # colors: list of -1, 0, 1
        # c0: count of 0s in each edge
        # c1: count of 1s in each edge
        
        def dfs(colors, c0, c1):
            # Check for immediate conflicts (monochromatic edges)
            # This is handled by propagation, but we check here for safety
            for i in range(len(edges)):
                if c0[i] == 5 or c1[i] == 5:
                    return False
            
            # Find unassigned vertex with most constraints (heuristic)
            # Or just first unassigned for speed
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            
            if v == -1:
                return True # Found a valid coloring
            
            # Try color 0
            new_colors = colors[:]
            new_c0 = c0[:]
            new_c1 = c1[:]
            
            if propagate_and_check(v, 0, new_colors, new_c0, new_c1, v_to_e, edges):
                if dfs(new_colors, new_c0, new_c1):
                    return True
            
            # Try color 1
            new_colors = colors[:]
            new_c0 = c0[:]
            new_c1 = c1[:]
            
            if propagate_and_check(v, 1, new_colors, new_c0, new_c1, v_to_e, edges):
                if dfs(new_colors, new_c0, new_c1):
                    return True
            
            return False
        
        def propagate_and_check(v, color, colors, c0, c1, v_to_e, edges):
            if colors[v] != -1:
                return colors[v] == color
            
            colors[v] = color
            if color == 0:
                for e_idx in v_to_e[v]:
                    c0[e_idx] += 1
                    if c0[e_idx] == 5:
                        return False
                    if c0[e_idx] == 4:
                        edge = edges[e_idx]
                        unassigned = -1
                        for u in edge:
                            if colors[u] == -1:
                                unassigned = u
                                break
                        if unassigned != -1:
                            if not propagate_and_check(unassigned, 1, colors, c0, c1, v_to_e, edges):
                                return False
            else:
                for e_idx in v_to_e[v]:
                    c1[e_idx] += 1
                    if c1[e_idx] == 5:
                        return False
                    if c1[e_idx] == 4:
                        edge = edges[e_idx]
                        unassigned = -1
                        for u in edge:
                            if colors[u] == -1:
                                unassigned = u
                                break
                        if unassigned != -1:
                            if not propagate_and_check(unassigned, 0, colors, c0, c1, v_to_e, edges):
                                return False
            
            return True
        
        init_colors = [-1] * n
        init_c0 = [0] * len(edges)
        init_c1 = [0] * len(edges)
        
        if dfs(init_colors, init_c0, init_c1):
            return False # It IS 2-colorable
        else:
            return True # It is NOT 2-colorable

    # Greedy Minimization from a dense hypergraph
    def minimize_greedy(n, edges):
        current_edges = [tuple(e) for e in edges]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to avoid bias
            indices = list(range(len(current_edges)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                if i >= len(current_edges):
                    continue
                
                # Try removing edge i
                test_edges = [e for j, e in enumerate(current_edges) if j != i]
                
                if is_non_2_colorable(n, test_edges):
                    current_edges = test_edges
                    changed = True
                    # Restart loop to re-shuffle and re-index
                    break
        
        return current_edges

    # Main Logic
    best_edges = None
    best_n = None
    min_edges_found = float('inf')
    
    # Start with K_9^5 (126 edges) which is known to be non-2-colorable
    n = 9
    all_edges = list(combinations(range(n), 5))
    # Verify K_9^5 is non-2-colorable (it should be)
    if not is_non_2_colorable(n, all_edges):
        # Fallback if something went wrong (should not happen)
        return {'n': n, 'edges': [list(e) for e in all_edges]}
        
    # Minimize K_9^5
    minimized_edges = minimize_greedy(n, all_edges)
    
    if len(minimized_edges) < min_edges_found:
        min_edges_found = len(minimized_edges)
        best_edges = [list(e) for e in minimized_edges]
        best_n = n
        
    # Try n=10? K_10^5 has 252 edges.
    # It might minimize to fewer edges?
    # But K_9^5 is already small.
    # Let's try to run a few random restarts on n=10 if time permits?
    # No, greedy on K_9^5 is a good baseline.
    
    # If time remains, try to improve by local search on the minimized graph?
    # Or try n=10.
    
    if time.time() < deadline - 20:
        n2 = 10
        all_edges2 = list(combinations(range(n2), 5))
        if is_non_2_colorable(n2, all_edges2):
            minimized_edges2 = minimize_greedy(n2, all_edges2)
            if len(minimized_edges2) < min_edges_found:
                min_edges_found = len(minimized_edges2)
                best_edges = [list(e) for e in minimized_edges2]
                best_n = n2

    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
