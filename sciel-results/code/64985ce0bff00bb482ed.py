import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph (n, edges) has a proper 2-coloring.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    Optimized for small n (<= 8).
    """
    if not edges:
        return True
        
    # Convert edges to frozensets for faster lookups if needed, 
    # but for backtracking, lists of indices are fine.
    # We use a recursive backtracking solver.
    
    # Pre-compute for each vertex, which edges contain it?
    # Or just iterate edges.
    
    assignment = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        # Try 0
        assignment[idx] = 0
        if is_consistent(idx):
            if backtrack(idx + 1):
                return True
        # Try 1
        assignment[idx] = 1
        if is_consistent(idx):
            if backtrack(idx + 1):
                return True
        
        assignment[idx] = -1
        return False

    def is_consistent(idx):
        # Check all edges
        for e in edges:
            # If all vertices in e are assigned
            # Optimization: if the max vertex in e is <= idx, it's fully assigned
            # But we assign sequentially 0..n-1, so if max(e) <= idx, it's assigned.
            # Actually, since we assign in order, if any v in e is > idx, it's unassigned.
            # We can check: if all(v <= idx for v in e) is False, skip?
            # No, if v <= idx, it is assigned. If v > idx, it is not.
            # So if min(v for v in e) > idx, skip.
            # If max(v for v in e) <= idx, check colors.
            
            # Simpler: just check if all assigned.
            # For n=8, this is very fast.
            all_assigned = True
            for v in e:
                if assignment[v] == -1:
                    all_assigned = False
                    break
            if all_assigned:
                c0 = False
                c1 = False
                for v in e:
                    if assignment[v] == 0:
                        c0 = True
                    else:
                        c1 = True
                    if c0 and c1:
                        break
                if not (c0 and c1):
                    return False
        return True

    return backtrack(0)

def construct():
    # Start with K_8^(5)
    n = 8
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    best_edges = all_edges[:]
    best_count = len(all_edges)
    
    # Try multiple random restarts of greedy removal
    for seed in range(100):
        random.seed(seed)
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        current_edges = [tuple(e) for e in all_edges]
        
        for edge in edges_to_try:
            if edge not in current_edges:
                continue
            
            # Temporarily remove
            current_edges.remove(edge)
            
            # Check if still non-2-colorable
            if not is_2_colorable(n, current_edges):
                # Still non-2-colorable, keep removed
                pass
            else:
                # Became 2-colorable, add back
                current_edges.append(edge)
        
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            
            # If we reach the theoretical lower bound or a known good bound, we can stop early?
            # The task says best known is 51. If we get 51, that's optimal for this search.
            # We can continue to see if we get lower, but 51 is likely the min.
            if best_count <= 51:
                break

    return {'n': n, 'edges': best_edges}
