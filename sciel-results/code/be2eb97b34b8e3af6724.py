import time
import random
from itertools import combinations

def _is_2_colorable_fast(edges, n, deadline):
    """
    Fast 2-colorability check using backtracking with unit propagation.
    edges: list of tuples of 5 vertices.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
        
    # Build incidence
    inc = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            inc[v].append(i)
            
    # Sort vertices by degree for better branching
    vertices_by_degree = sorted(range(n), key=lambda x: len(inc[x]), reverse=True)
    
    # State
    assign = [-1] * n
    
    # Propagation logic
    # If an edge has 4 vertices of color c, the 5th must be 1-c.
    # If an edge has 5 vertices of color c, conflict.
    
    def propagate(changed):
        stack = list(changed)
        while stack:
            v = stack.pop()
            cv = assign[v]
            if cv == -1:
                continue
            for ei in inc[v]:
                e = edges[ei]
                # Count colors in this edge
                c0 = 0
                c1 = 0
                unassigned = -1
                for u in e:
                    cu = assign[u]
                    if cu == 0:
                        c0 += 1
                    elif cu == 1:
                        c1 += 1
                    else:
                        unassigned = u
                
                if c0 + c1 + (1 if unassigned != -1 else 0) != 5:
                    # More than 1 unassigned, no propagation
                    continue
                
                if unassigned != -1:
                    if c0 == 4:
                        # Must be 1
                        if assign[unassigned] == -1:
                            assign[unassigned] = 1
                            stack.append(unassigned)
                        elif assign[unassigned] == 0:
                            return False
                    elif c1 == 4:
                        # Must be 0
                        if assign[unassigned] == -1:
                            assign[unassigned] = 0
                            stack.append(unassigned)
                        elif assign[unassigned] == 1:
                            return False
                else:
                    # All assigned
                    if c0 == 5 or c1 == 5:
                        return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return None # Timeout, treat as unknown/failed
        
        # Find next unassigned vertex
        while idx < n and assign[vertices_by_degree[idx]] != -1:
            idx += 1
            
        if idx == n:
            # All assigned, check if valid (propagation should have caught conflicts, but double check)
            for e in edges:
                c0 = sum(1 for u in e if assign[u] == 0)
                if c0 == 0 or c0 == 5:
                    return None
            return assign[:]
            
        v = vertices_by_degree[idx]
        
        # Try 0
        assign[v] = 0
        if propagate([v]):
            res = backtrack(idx + 1)
            if res is not None:
                return res
        assign[v] = -1
        
        # Try 1
        assign[v] = 1
        if propagate([v]):
            res = backtrack(idx + 1)
            if res is not None:
                return res
        assign[v] = -1
        
        return None

    res = backtrack(0)
    return res is not None

def _minimize_edges(edges, n, deadline):
    """
    Try to remove edges one by one.
    """
    edges_set = set(edges)
    # Shuffle for randomness
    edges_list = list(edges_set)
    random.shuffle(edges_list)
    
    for e in edges_list:
        if time.time() > deadline:
            break
        if e not in edges_set:
            continue
        edges_set.remove(e)
        # Check if still non-2-colorable
        if not _is_2_colorable_fast(list(edges_set), n, deadline):
            # Still non-2-colorable, keep removed
            pass
        else:
            # Became 2-colorable, put back
            edges_set.add(e)
            
    return list(edges_set)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # We know m(5) <= 51. We aim for < 51.
    # Start with n=9, 10, 11
    
    # Hardcoded fallback: K9^(5) (126 edges)
    fallback_edges = [tuple(e) for e in combinations(range(9), 5)]
    fallback_n = 9
    
    # Strategy: Randomized construction
    # Try to build a non-2-colorable hypergraph with few edges
    
    for n in [9, 10, 11, 12]:
        if time.time() > deadline:
            break
            
        # Try multiple random instances
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Generate a random set of edges
            # Start with a dense set, then minimize
            # Or start sparse and add until non-2-colorable?
            # Adding is hard to check efficiently.
            # Better: Start with K_n^(5) and minimize.
            
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            # Take a subset?
            # For n=9, 126 edges.
            # For n=10, 252 edges.
            
            # We want to find a minimal non-2-colorable subgraph.
            # Start with the full graph, which is definitely non-2-colorable for n>=9.
            # Then minimize.
            
            current_edges = all_edges[:]
            
            # Minimize
            minimized = _minimize_edges(current_edges, n, deadline)
            
            if len(minimized) < best_count:
                # Verify it is indeed non-2-colorable
                if not _is_2_colorable_fast(minimized, n, deadline):
                    best_edges = minimized
                    best_n = n
                    best_count = len(minimized)
                    
                    # If we found something very small, we can stop early
                    if best_count <= 30:
                        break
            
        if best_count <= 30:
            break
            
    # If no better than fallback found, use fallback
    if best_edges is None or best_count > len(fallback_edges):
        best_edges = fallback_edges
        best_n = fallback_n
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
