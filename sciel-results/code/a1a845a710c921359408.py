import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin for 150s limit
    
    n = 9
    # Generate all 5-uniform edges on 9 vertices
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    
    # Initial best is the complete hypergraph
    best_set = set(all_edges)
    
    # Solver: Returns True if the hypergraph is NOT 2-colorable
    def is_non_2_colorable(n, edges):
        if n < 9:
            return False
            
        m = len(edges)
        if m == 0:
            return False
            
        # Map vertex to list of edge indices
        v_to_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_edges[v].append(i)
        
        # Colors: -1 unassigned, 0, 1
        colors = [-1] * n
        # Symmetry breaking: Vertex 0 is always 0
        colors[0] = 0
        
        # Propagation function
        # Returns False if conflict, True if consistent (so far)
        def propagate(v, c):
            if colors[v] != -1:
                return colors[v] == c
            
            colors[v] = c
            
            # Check all edges containing v
            for e_idx in v_to_edges[v]:
                edge = edges[e_idx]
                cnt0 = 0
                cnt1 = 0
                unassigned = -1
                
                for u in edge:
                    if colors[u] == 0:
                        cnt0 += 1
                    elif colors[u] == 1:
                        cnt1 += 1
                    else:
                        unassigned = u
                
                # Check for monochromatic edge
                if cnt0 == 5 or cnt1 == 5:
                    return False
                
                # Unit Propagation
                if cnt0 == 4:
                    # The 5th vertex MUST be 1
                    if unassigned != -1:
                        if not propagate(unassigned, 1):
                            return False
                    else:
                        # All vertices assigned.
                        # If cnt0=4 and all assigned, then cnt1=1. Consistent.
                        pass
                elif cnt1 == 4:
                    # The 5th vertex MUST be 0
                    if unassigned != -1:
                        if not propagate(unassigned, 0):
                            return False
                    else:
                        pass
                        
            return True

        # DFS
        def dfs():
            # Find next unassigned vertex
            # Heuristic: Pick vertex with most assigned neighbors in edges?
            # For n=9, first unassigned is fast enough.
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            
            if v == -1:
                return True # All assigned, no conflict
            
            # Try color 0
            old_colors = colors[:]
            if propagate(v, 0):
                if dfs():
                    return True
            colors[:] = old_colors
            
            # Try color 1
            old_colors = colors[:]
            if propagate(v, 1):
                if dfs():
                    return True
            colors[:] = old_colors
            
            return False
            
        if dfs():
            return False # Colorable
        else:
            return True # Not Colorable

    # Minimization Loop
    # We iteratively try to remove edges.
    # We use a set for fast removal.
    
    # We run multiple passes to ensure we find a minimal core.
    # Each pass shuffles the edges and tries to remove them greedily.
    
    while time.time() < deadline:
        current_set = set(best_set)
        
        # Get list of edges to try removing
        # We shuffle to avoid bias
        edges_list = list(current_set)
        random.shuffle(edges_list)
        
        improved = False
        
        for edge in edges_list:
            if time.time() > deadline:
                break
                
            # If this edge was already removed in a previous step of this pass, skip
            if edge not in current_set:
                continue
                
            # Try removing it
            test_set = current_set - {edge}
            test_list = list(test_set)
            
            # Check if still non-2-colorable
            if is_non_2_colorable(n, test_list):
                # We can remove it
                current_set = test_set
                improved = True
                
                # If we found a smaller set, update best
                if len(current_set) < len(best_set):
                    best_set = current_set
        
        # If no improvement in this pass, we might be stuck.
        # But since we shuffle, the next pass might find a different removal path.
        # However, if we are at the absolute minimum, no pass will improve.
        # To save time, we could break if len(current_set) is very low (e.g., <= 52).
        # But we don't know the exact min. 51 is known.
        # Let's just let it run until deadline. The solver is fast.
        
        # Optional: If we reach 51 edges, we can stop early as 51 is best known.
        # But maybe 50 is possible? The prompt says "best known construction has 51".
        # It implies 51 is the target.
        if len(best_set) <= 51:
            break

    return {'n': n, 'edges': [list(e) for e in best_set]}
