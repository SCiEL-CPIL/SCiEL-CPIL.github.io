import random
import time
from itertools import combinations

def is_2_colorable(edges, n):
    """
    Check if the hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Map vertices to lists of edges they are in
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    
    def propagate(v, c):
        """
        Assign color c to vertex v.
        Returns False if contradiction, True otherwise.
        Modifies colors.
        """
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        # Check edges containing v
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            # Count colors in this edge
            count_c = 0
            count_1_minus_c = 0
            unassigned = -1
            
            for u in e:
                if colors[u] == c:
                    count_c += 1
                elif colors[u] == 1 - c:
                    count_1_minus_c += 1
                else:
                    unassigned = u
            
            # If 4 vertices are 1-c, the last must be c
            if count_1_minus_c == 4:
                if unassigned != -1:
                    if not propagate(unassigned, c):
                        return False
            # If 4 vertices are c, the last must be 1-c
            elif count_c == 4:
                if unassigned != -1:
                    if not propagate(unassigned, 1 - c):
                        return False
            
            # If all 5 are c, contradiction
            if count_c == 5:
                return False
            # If all 5 are 1-c, contradiction
            if count_1_minus_c == 5:
                return False
                
        return True

    def backtrack():
        # Find unassigned vertex with highest degree (most edges)
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return True # All assigned, no contradictions found
        
        # Try color 0
        # Save state
        old_colors = colors[:]
        if propagate(best_v, 0):
            if backtrack():
                return True
        # Restore
        colors[:] = old_colors
        
        # Try color 1
        old_colors = colors[:]
        if propagate(best_v, 1):
            if backtrack():
                return True
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_n = 9
    # Initialize with complete 5-uniform hypergraph on 9 vertices
    all_edges_n9 = list(combinations(range(9), 5))
    best_edges = [list(e) for e in all_edges_n9]
    best_count = len(best_edges)
    
    # We will try to minimize the number of edges.
    # We use a random greedy removal strategy.
    # We start with the complete hypergraph and try to remove edges.
    
    # To improve results, we try multiple random orders of removal.
    # We also try n=10 and n=11 if we have time, but n=9 is dense.
    
    num_trials = 500 # We will break early if time runs out
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Choose n. Start with 9.
        # Maybe try 10 occasionally?
        # For now, stick to 9 to ensure we find a good one quickly.
        n = 9
        all_edges = list(combinations(range(n), 5))
        
        # Shuffle edges
        shuffled_edges = [list(e) for e in all_edges]
        random.shuffle(shuffled_edges)
        
        # Active edges
        active_edges = shuffled_edges[:]
        active_set = set(tuple(e) for e in active_edges)
        
        # Try to remove each edge
        # We iterate through the shuffled list.
        # If the edge is still active, we try to remove it.
        for e in shuffled_edges:
            e_tuple = tuple(e)
            if e_tuple not in active_set:
                continue
                
            # Try to remove e
            # Create a temporary list without e
            # Since active_edges is a list, we can filter it.
            # To be faster, we can just pass a generator or filtered list.
            # is_2_colorable takes a list.
            
            # Optimization: 
            # We can maintain a list of edges.
            # Filtering a list of ~100 elements is fast.
            
            temp_edges = [x for x in active_edges if x != e]
            
            if not is_2_colorable(temp_edges, n):
                # Still non-2-colorable, so e is redundant
                active_set.remove(e_tuple)
                # Remove from active_edges
                # Since we are iterating over shuffled_edges, and we just need the final set,
                # we can just update active_edges.
                # active_edges = temp_edges is O(M).
                # Doing this M times is O(M^2).
                # M=126, M^2=16k. 500 trials * 16k = 8M operations.
                # Plus is_2_colorable cost.
                # Should be fine.
                active_edges = temp_edges
        
        # Update best
        if len(active_edges) < best_count:
            best_count = len(active_edges)
            best_edges = active_edges[:]
            best_n = n

    return {'n': best_n, 'edges': best_edges}
