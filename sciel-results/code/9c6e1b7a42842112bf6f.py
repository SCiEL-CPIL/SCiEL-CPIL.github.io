import time
import random
from itertools import combinations

def is_not_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph is NOT 2-colorable.
    Returns True if NOT 2-colorable (i.e., no proper 2-colouring exists).
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False  # Empty hypergraph is 2-colorable
    
    num_edges = len(edges)
    
    # Convert edges to tuples of vertices (sorted for consistency)
    edge_tuples = [tuple(sorted(e)) for e in edges]
    
    # Precompute vertex -> list of edge indices
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_tuples):
        for v in e:
            vertex_edges[v].append(i)
    
    # State
    c0 = [0] * num_edges  # count of color 0 in each edge
    c1 = [0] * num_edges  # count of color 1 in each edge
    color = [-1] * n
    
    def propagate():
        """Unit propagation. Returns False on conflict, True otherwise."""
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                
                if c0[i] == 4 and c1[i] == 0:
                    # Must have at least one vertex of color 1
                    # If exactly one unassigned, it must be color 1
                    unassigned = []
                    for v in edge_tuples[i]:
                        if color[v] == -1:
                            unassigned.append(v)
                    if len(unassigned) == 1:
                        v = unassigned[0]
                        color[v] = 1
                        c1[i] += 1
                        for other_e in vertex_edges[v]:
                            c1[other_e] += 1
                        changed = True
                        if c1[i] == 5:
                            return False
                        for other_e in vertex_edges[v]:
                            if c1[other_e] == 5:
                                return False
                        
                elif c1[i] == 4 and c0[i] == 0:
                    # Must have at least one vertex of color 0
                    unassigned = []
                    for v in edge_tuples[i]:
                        if color[v] == -1:
                            unassigned.append(v)
                    if len(unassigned) == 1:
                        v = unassigned[0]
                        color[v] = 0
                        c0[i] += 1
                        for other_e in vertex_edges[v]:
                            c0[other_e] += 1
                        changed = True
                        if c0[i] == 5:
                            return False
                        for other_e in vertex_edges[v]:
                            if c0[other_e] == 5:
                                return False
        return True

    def backtrack(depth=0):
        """Returns True if a valid 2-coloring exists."""
        # Check for conflicts
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
        
        # Find most constrained edge
        best_edge = -1
        best_unassigned = 6
        best_max_color = 0
        
        for i in range(num_edges):
            unassigned = 5 - c0[i] - c1[i]
            max_color = max(c0[i], c1[i])
            if unassigned < best_unassigned or (unassigned == best_unassigned and max_color > best_max_color):
                best_unassigned = unassigned
                best_max_color = max_color
                best_edge = i
        
        if best_edge == -1:
            return True  # All edges satisfied
        
        if best_unassigned == 0:
            return False  # No unassigned vertices in most constrained edge
        
        # Get candidates
        candidates = [v for v in edge_tuples[best_edge] if color[v] == -1]
        if not candidates:
            return True
        
        # Try each candidate with each color
        for v in candidates:
            # Try color 0
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            color[v] = 0
            ok = True
            for e_idx in vertex_edges[v]:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack(depth + 1):
                    return True
            
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            # Try color 1
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            color[v] = 1
            ok = True
            for e_idx in vertex_edges[v]:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack(depth + 1):
                    return True
            
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
        
        return False

    # Symmetry breaking: fix vertex 0 to color 0 if it exists
    if n > 0 and vertex_edges[0]:
        color[0] = 0
        for e_idx in vertex_edges[0]:
            c0[e_idx] += 1
            if c0[e_idx] == 5:
                return True  # Conflict means not 2-colorable
    
    if not propagate():
        return True  # Conflict means not 2-colorable
    
    # If backtrack finds a valid coloring, it IS 2-colorable
    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_edges = None
    best_n = 0
    best_m = 10**9
    
    # Known good starting point: K9^5 has C(9,5)=126 edges
    n = 9
    all_possible = list(combinations(range(n), 5))
    current_edges = [list(e) for e in all_possible]
    current_m = len(current_edges)
    
    # Verify starting point
    if is_not_2_colorable(n, current_edges):
        best_edges = current_edges[:]
        best_n = n
        best_m = current_m
    else:
        # Fallback: try n=10
        n = 10
        all_possible = list(combinations(range(n), 5))
        current_edges = [list(e) for e in all_possible]
        current_m = len(current_edges)
        if is_not_2_colorable(n, current_edges):
            best_edges = current_edges[:]
            best_n = n
            best_m = current_m
    
    # Local search with edge removal
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random restart: start from current best or random subset
        if attempt == 0:
            edges_to_try = current_edges[:]
        else:
            # Start from best found so far
            edges_to_try = best_edges[:]
            # Randomly shuffle order for removal
            random.shuffle(edges_to_try)
        
        improved = True
        while improved:
            improved = False
            for idx in range(len(edges_to_try)):
                if time.time() > deadline:
                    break
                
                new_edges = edges_to_try[:idx] + edges_to_try[idx+1:]
                
                if is_not_2_colorable(n, new_edges):
                    edges_to_try = new_edges
                    improved = True
                    if len(edges_to_try) < best_m:
                        best_m = len(edges_to_try)
                        best_edges = edges_to_try[:]
                        best_n = n
                    # Re-shuffle for next pass
                    random.shuffle(edges_to_try)
                    break
        
        # Periodically try different n values
        if attempt % 10 == 5 and n == 9:
            # Try n=10
            n_alt = 10
            all_alt = list(combinations(range(n_alt), 5))
            random.shuffle(all_alt)
            # Take a random subset and try to reduce
            subset_size = min(60, len(all_alt))
            current_alt = [list(e) for e in all_alt[:subset_size]]
            
            # Try to reduce
            improved_alt = True
            while improved_alt:
                improved_alt = False
                for idx in range(len(current_alt)):
                    if time.time() > deadline:
                        break
                    new_alt = current_alt[:idx] + current_alt[idx+1:]
                    if is_not_2_colorable(n_alt, new_alt):
                        current_alt = new_alt
                        improved_alt = True
                        random.shuffle(current_alt)
                        break
            
            if len(current_alt) < best_m:
                best_m = len(current_alt)
                best_edges = current_alt[:]
                best_n = n_alt
    
    if best_edges is None:
        # Absolute fallback: return K9^5
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
