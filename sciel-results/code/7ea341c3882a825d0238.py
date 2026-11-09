import time
import random
import itertools
import sys
from collections import defaultdict

def _is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable.
    Returns True if 2-colorable, False if not.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Map vertices to 0..n-1 (assumed already)
    # Preprocess: remove duplicates
    unique_edges = set()
    for e in edges:
        unique_edges.add(tuple(sorted(e)))
    if not unique_edges:
        return True
        
    # Order vertices by degree descending
    deg = [0] * n
    for e in unique_edges:
        for v in e:
            deg[v] += 1
    order = sorted(range(n), key=lambda v: -deg[v])
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        
    remapped_edges = []
    for e in unique_edges:
        remapped_edges.append(tuple(pos[v] for v in e))
        
    colors = [-1] * n
    # For speed, we can precompute for each vertex, which edges it belongs to
    # But for n<=26 and e<=200, simple loop is okay if we prune well.
    
    # Unit propagation logic:
    # If an edge has 4 vertices of color c, the 5th must be 1-c.
    
    def propagate(k):
        changed = True
        while changed:
            changed = False
            for e in remapped_edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if v < k:
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                    else:
                        unassigned.append(v)
                        
                if not unassigned:
                    if c0 == 5 or c1 == 5:
                        return False
                else:
                    if c0 == 4 and len(unassigned) == 1:
                        colors[unassigned[0]] = 1
                        changed = True
                    elif c1 == 4 and len(unassigned) == 1:
                        colors[unassigned[0]] = 0
                        changed = True
                    elif c0 == 5 or c1 == 5:
                        return False
        return True

    def backtrack(k):
        if k == n:
            for e in remapped_edges:
                c0 = 0
                c1 = 0
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    else:
                        c1 += 1
                if c0 == 5 or c1 == 5:
                    return False
            return True
            
        # Heuristic: Try color that appears more often in neighbors? Or just 0 then 1.
        colors[k] = 0
        if propagate(k+1):
            if backtrack(k+1):
                return True
        colors[k] = 1
        if propagate(k+1):
            if backtrack(k+1):
                return True
        colors[k] = -1
        return False

    return backtrack(0)

def _minimize_edges(n, initial_edges, time_limit):
    """
    Given a non-2-colorable hypergraph, try to remove edges one by one
    to minimize the count while keeping it non-2-colorable.
    """
    start_time = time.time()
    current_edges = list(initial_edges)
    
    # Sort edges by some heuristic? Maybe random order to avoid bias.
    random.shuffle(current_edges)
    
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        # Try to remove each edge
        for i in range(len(current_edges)):
            if (time.time() - start_time) > time_limit:
                break
            # Temporarily remove edge i
            candidate_edges = current_edges[:i] + current_edges[i+1:]
            # Check if it is still non-2-colorable
            if not _is_2_colorable(n, candidate_edges):
                # It is still non-2-colorable, so we can keep this edge removed
                current_edges = candidate_edges
                changed = True
                break # Restart loop with new list
            # If it became 2-colorable, we must keep edge i.
            
    return current_edges

def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s margin
    
    best_result = None
    best_edge_count = float('inf')
    
    # Strategy: 
    # 1. Try to construct a non-2-colorable hypergraph with a specific structure.
    #    The complete 5-uniform hypergraph on 9 vertices has 126 edges.
    #    We can start with smaller n and random edges, or use a known construction.
    #    However, finding a small one from scratch is hard.
    #    Let's try n=10, 11, 12 with random dense graphs, then minimize.
    
    # To ensure we have a valid non-2-colorable graph to start with,
    # we can use the fact that random graphs become non-2-colorable quickly.
    # For n=10, C(10,5)=252.
    # For n=11, C(11,5)=462.
    
    # Let's try a few random instances for n=10, 11, 12.
    for n in [10, 11, 12]:
        if time.time() - start_time > DEADLINE:
            break
            
        all_possible_edges = list(itertools.combinations(range(n), 5))
        
        # Try a few random subsets that are likely non-2-colorable
        # Random graphs with p=0.5 are usually non-2-colorable for n>=10.
        for attempt in range(20):
            if time.time() - start_time > DEADLINE:
                break
                
            # Select a random subset of edges, biased towards higher density
            # to ensure non-2-colorability initially.
            # If p is too low, it might be 2-colorable.
            # Let's try p=0.8, 0.6, 0.4
            p = 0.5 + 0.2 * (attempt % 3)
            
            initial_edges = []
            for e in all_possible_edges:
                if random.random() < p:
                    initial_edges.append(e)
            
            # Check if non-2-colorable
            if not _is_2_colorable(n, initial_edges):
                # It is non-2-colorable. Now minimize.
                # Time budget for minimization: 10 seconds per attempt
                time_left = DEADLINE - (time.time() - start_time)
                if time_left < 5:
                    continue
                minimized_edges = _minimize_edges(n, initial_edges, time_left - 5)
                
                if len(minimized_edges) < best_edge_count:
                    best_edge_count = len(minimized_edges)
                    best_result = {'n': n, 'edges': [list(e) for e in minimized_edges]}
                    
                # If we find a very small one, we can stop early?
                # Theoretical lower bound is >28. Best known is 51.
                # If we find < 60, it's a good result.
                if best_edge_count <= 55:
                    return best_result

    # Fallback: Complete graph on 9 vertices (126 edges)
    if best_result is None or best_edge_count > 126:
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
        
    return best_result
