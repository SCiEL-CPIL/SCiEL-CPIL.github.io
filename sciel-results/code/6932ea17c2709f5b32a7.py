import time
import itertools
import random

DEADLINE = 145.0

def _now():
    return time.time()

def _is_2_colorable(n, edges):
    """
    Check if the 5-uniform hypergraph on vertices 0..n-1 with given edges
    is 2-colorable.
    Returns True if it is 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # For n <= 20, brute force over 2^(n-1) colorings is feasible if n is small.
    # For n=9, 2^8 = 256 colorings.
    if n > 20:
        # Fallback to a simple backtracking if n is large, but we won't use large n here.
        pass

    for mask in range(1 << (n - 1)):
        colors = [0] * n
        colors[0] = 0
        for i in range(1, n):
            if mask & (1 << (i - 1)):
                colors[i] = 1
        
        is_mono = False
        for edge in edges:
            c0 = colors[edge[0]]
            # Check if all vertices in edge have color c0
            # Unroll for speed? No, n=5 is small.
            if colors[edge[1]] == c0 and colors[edge[2]] == c0 and colors[edge[3]] == c0 and colors[edge[4]] == c0:
                is_mono = True
                break
        
        if not is_mono:
            return True
    
    return False

def construct():
    start_time = _now()
    
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    best_edges = None
    best_count = len(all_edges)
    
    # Try multiple random orderings to find a smaller minimal non-2-colorable subgraph
    num_trials = 50
    
    for trial in range(num_trials):
        if _now() > DEADLINE - 10: # Leave 10s margin
            break
        
        current_edges = all_edges[:]
        random.seed(trial)
        random.shuffle(current_edges)
        
        # Iterative edge removal
        changed = True
        while changed and _now() < DEADLINE - 5:
            changed = False
            # We iterate through the current list of edges
            # To avoid index shifting issues, we build a new list
            new_edges = []
            for edge in current_edges:
                # Try removing this edge
                # Check if current_edges without this edge is 2-colorable
                # This is O(N) check inside a loop, so O(N^2 * CheckCost)
                # CheckCost is ~256 * 5 * E.
                # E is ~60.
                # 256 * 300 = 76k ops.
                # N is ~60.
                # 60 * 76k = 4.5M ops per pass.
                # This is fast.
                
                # Optimization: Instead of creating a new list every time, 
                # we can just check if the rest are 2-colorable.
                # But _is_2_colorable takes a list.
                # Creating a new list of size N-1 is O(N).
                
                # Let's just do it.
                remaining = [e for e in current_edges if e != edge]
                # This is O(N^2) due to list comprehension + comparison.
                # Better to use indices or sets?
                # For N=60, it's fine.
                
                if _is_2_colorable(n, remaining):
                    # Must keep edge
                    new_edges.append(edge)
                else:
                    # Can remove edge
                    changed = True
                    # Do not append edge
            
            current_edges = new_edges
        
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
        # If we found 51 or less, we can stop early?
        # The task says best known is 51.
        if best_count <= 51:
            break

    if best_edges is None:
        best_edges = all_edges
        
    return {'n': n, 'edges': best_edges}
