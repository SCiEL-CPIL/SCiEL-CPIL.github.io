import itertools
import random
import time

def _is_non_2_colorable(n, edges):
    """
    Check if the hypergraph is NOT 2-colorable.
    Returns True if no 2-coloring exists.
    Returns False if a 2-coloring exists.
    """
    # Map vertices to edges for fast lookup?
    # n is small (9).
    
    # Backtracking with unit propagation
    # colors: -1 (unassigned), 0, 1
    colors = [-1] * n
    
    # Precompute edge lists for each vertex?
    # Or just iterate edges.
    
    # To make it fast, we can use a simple recursive search.
    # Since n=9, the search tree is small if we are lucky, or we prune heavily.
    
    def backtrack(idx):
        if idx == len(edges):
            return False # All edges satisfied (no monochromatic), so 2-colorable exists
        
        edge = edges[idx]
        # Check if edge is already monochromatic
        # If all assigned and same color -> conflict (return True? No, this path is invalid)
        # We want to find A coloring.
        # So if we find a conflict, we return False (this branch failed).
        # If we finish all edges, we return True (coloring found).
        
        # Wait, the logic is:
        # We are trying to find a valid 2-coloring.
        # If we find one, the hypergraph IS 2-colorable.
        # We want to know if it is NOT 2-colorable.
        # So if backtrack finds a coloring, we return False (it IS 2-colorable).
        # If backtrack exhausts all possibilities, we return True (it is NOT 2-colorable).
        
        # Let's rephrase:
        # Function returns True if a valid coloring is FOUND.
        
        # Unit Propagation
        # If 4 vertices in an edge are assigned the same color, the 5th must be the other.
        
        # Simple approach:
        # Assign colors to vertices one by one?
        # Or assign edges?
        # Standard CSP: assign vertices.
        
        pass

    # Better: Assign vertices.
    # But constraints are on edges.
    
    # Let's use a simple recursive vertex assignment.
    # Order vertices by degree?
    
    def solve(assigned_count, colors):
        if assigned_count == n:
            # Check all edges
            for e in edges:
                c0 = colors[e[0]]
                if all(colors[v] == c0 for v in e):
                    return False # Monochromatic edge found, invalid coloring
            return True # Valid coloring found
        
        # Pick next unassigned vertex
        v = assigned_count
        # Try color 0
        colors[v] = 0
        # Pruning: Check if any edge involving v is now forced/conflicting?
        # Simple check: if any edge is now monochromatic, fail.
        ok = True
        for e in edges:
            if v in e:
                # Check if e is monochromatic
                c = colors[e[0]]
                for u in e:
                    if colors[u] != -1 and colors[u] != c:
                        break
                else:
                    # All assigned in e are c
                    # If all are assigned, it's monochromatic.
                    # If some unassigned, it might become non-mono.
                    # But if all assigned are c, and we just set v to c, 
                    # we need to ensure we can break the mono-ness.
                    # If all 5 are assigned and all c, fail.
                    if all(colors[u] != -1 for u in e):
                        ok = False
                        break
                    # Unit prop: if 4 are c, the 5th MUST be 1-c.
                    # We can implement this, but simple check is safer first.
        
        if ok:
            if solve(assigned_count + 1, colors):
                return True
        
        # Try color 1
        colors[v] = 1
        ok = True
        for e in edges:
            if v in e:
                c = colors[e[0]]
                for u in e:
                    if colors[u] != -1 and colors[u] != c:
                        break
                else:
                    if all(colors[u] != -1 for u in e):
                        ok = False
                        break
        
        if ok:
            if solve(assigned_count + 1, colors):
                return True
        
        colors[v] = -1
        return False

    return not solve(0, [-1]*n)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Start with K9^(5)
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Optimize: Precompute vertex-edge incidence?
    # For n=9, iterating edges is fine.
    
    # Pruning loop
    # We want to remove edges while keeping non-2-colorability.
    # We try random edges.
    
    # To make it faster, we can cache the result of is_non_2_colorable?
    # The set of edges changes, so no direct cache.
    
    # Heuristic: Remove edges that are "redundant".
    # An edge is redundant if removing it doesn't make it 2-colorable.
    
    # We iterate until no more edges can be removed or time runs out.
    
    # To speed up, we can use a faster checker.
    # The simple backtracking above might be slow if it explores a large tree.
    # But for n=9, it's at most 2^9 = 512 leaves if no pruning.
    # With pruning, it's much faster.
    
    # Let's try to improve the checker with bitmasks or better pruning.
    # Actually, 512 leaves is VERY fast.
    # The bottleneck is the number of checks.
    # We have 126 edges. We might do 126 * k checks.
    # If each check is ~1ms, 126 checks is 0.1s.
    # We can do many iterations.
    
    # Let's run the pruning.
    
    # Shuffle edges to remove in random order
    random.shuffle(current_edges)
    
    # We will try to remove each edge once.
    # If successful, we might be able to remove more.
    # So we loop until no edge can be removed.
    
    removed_any = True
    while removed_any and (time.time() - start_time) < deadline:
        removed_any = False
        # Shuffle again to avoid bias?
        # Or just iterate.
        for i in range(len(current_edges) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
            
            e = current_edges[i]
            # Try removing e
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if new_edges is non-2-colorable
            if _is_non_2_colorable(n, new_edges):
                # It is still non-2-colorable, so we can remove e
                current_edges = new_edges
                removed_any = True
            # Else, keep e
    
    return {'n': n, 'edges': current_edges}
