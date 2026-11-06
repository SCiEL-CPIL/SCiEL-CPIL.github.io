import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph (n, edges) is 2-colorable.
    """
    if not edges:
        return True
    
    # Precompute edge sets for faster access
    edge_sets = [frozenset(e) for e in edges]
    
    # Brute force over all 2^n colorings
    for mask in range(1 << n):
        # Check if any edge is monochromatic
        is_valid = True
        for es in edge_sets:
            # Count vertices in edge with color 1
            count_1 = 0
            for v in es:
                if (mask >> v) & 1:
                    count_1 += 1
            # If count_1 is 0 (all 0) or 5 (all 1), it's monochromatic
            if count_1 == 0 or count_1 == 5:
                is_valid = False
                break
        if is_valid:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try n=9
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Verify initial non-2-colorability
    if is_2_colorable(n, all_edges):
        # If complete graph is 2-colorable (unlikely for n=9), we need a different start
        # But for n=9, it is not.
        pass
        
    current_edges = list(all_edges)
    
    # Greedy removal with random restarts/shuffles
    # We perform multiple passes. In each pass, we shuffle the remaining edges
    # and try to remove them one by one.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Start from the current best set or all edges?
        # Starting from all edges is safer but slower.
        # Starting from the previous result is faster but might miss better solutions.
        # Let's start from all edges for the first few restarts, then from the best found.
        
        if restart < 5:
            working_edges = list(all_edges)
        else:
            if best_result is None:
                working_edges = list(all_edges)
            else:
                working_edges = list(best_result['edges'])
                # Ensure n is correct
                if best_result['n'] != n:
                    continue # Skip if n changed, though we fix n=9 here
            
        random.shuffle(working_edges)
        
        for i in range(len(working_edges)):
            if time.time() > deadline:
                break
                
            edge = working_edges[i]
            
            # Try removing edge
            # Create a list without this edge
            # To avoid O(N^2) copy, we can swap with last and pop, but we need to keep order for shuffling?
            # No, we just need the set of edges.
            
            # Efficient removal check:
            # We can maintain a list of active edges.
            
            # Let's use a simpler approach:
            # Remove edge, check, if bad, put back.
            
            # To make it faster, we can skip the check if we know it's critical?
            # No, we must check.
            
            # Optimization: If the number of edges is small, check is fast.
            
            # We'll just create a new list. For N=126, this is cheap.
            temp_edges = [e for j, e in enumerate(working_edges) if j != i]
            
            if not is_2_colorable(n, temp_edges):
                # Still not 2-colorable, so we can remove it
                working_edges[i] = None # Mark as removed
            # else: keep it
            
        # Clean up
        working_edges = [e for e in working_edges if e is not None]
        
        if len(working_edges) < best_edge_count:
            best_edge_count = len(working_edges)
            best_result = {'n': n, 'edges': working_edges}
            
            # Optional: Try to reduce n? 
            # If we have a solution with n=9, can we have one with n=8?
            # Max edges for n=8 is C(8,5)=56. If we can find a non-2-colorable one with <51 edges on n=8, that would be great.
            # But C(8,5)=56, so min edges could be lower.
            # Let's try n=8 if time permits.

    # Try n=8 if time permits
    if time.time() < deadline - 10:
        n = 8
        all_edges_n8 = [list(e) for e in combinations(range(n), 5)]
        if not is_2_colorable(n, all_edges_n8):
            # It is not 2-colorable, so we can try to reduce edges
            working_edges = list(all_edges_n8)
            for restart in range(10):
                if time.time() > deadline:
                    break
                random.shuffle(working_edges)
                for i in range(len(working_edges)):
                    if time.time() > deadline:
                        break
                    edge = working_edges[i]
                    temp_edges = [e for j, e in enumerate(working_edges) if j != i]
                    if not is_2_colorable(n, temp_edges):
                        working_edges[i] = None
                working_edges = [e for e in working_edges if e is not None]
                if len(working_edges) < best_edge_count:
                    best_edge_count = len(working_edges)
                    best_result = {'n': n, 'edges': working_edges}

    if best_result is None:
        # Fallback to complete graph on 9
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return best_result
