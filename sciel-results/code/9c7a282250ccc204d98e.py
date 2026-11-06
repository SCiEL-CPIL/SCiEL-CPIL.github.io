import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_map = {}
    
    # Generate all edges
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = len(edges) - 1
    
    n_edges = len(edges)
    
    # Precompute conflicts for each edge
    # conflict[e] is a list of 6 tuples of 3 edges
    # If we add edge e, we check if any of these triples are fully in selected
    conflicts = [[] for _ in range(n_edges)]
    
    for idx, (u, v) in enumerate(edges):
        # Find the bit flipped
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            # The other two vertices of the C4
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            # The 3 other edges
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_map[e1]
            idx2 = edge_map[e2]
            idx3 = edge_map[e3]
            
            conflicts[idx].append((idx1, idx2, idx3))
    
    # Build conflict graph adjacency
    # Two edges conflict if they are in the same C4 and adding both would create a C4
    # Actually, let's think about this differently.
    # We want to find the maximum set of edges with no C4.
    # This is equivalent to finding a maximum independent set in a conflict graph
    # where two edges are adjacent if they "conflict" (i.e., there exists a C4 that contains both).
    
    # Let's build the conflict graph
    conflict_adj = [set() for _ in range(n_edges)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_map[e1]
            idx2 = edge_map[e2]
            idx3 = edge_map[e3]
            
            # idx conflicts with idx1, idx2, idx3 in this C4
            # But actually, the conflict is more subtle.
            # Let me reconsider.
    
    # Actually, the original approach is better. Let's stick with the direct edge addition/removal
    # but improve the local search.
    
    def is_valid_add(edge_idx, selected):
        for t in conflicts[edge_idx]:
            if all(x in selected for x in t):
                return False
        return True
    
    def greedy_fill(selected, available_indices):
        for idx in available_indices:
            if idx not in selected:
                if is_valid_add(idx, selected):
                    selected.add(idx)
        return selected
    
    def local_search_improve(selected, max_iterations=1000):
        """
        Try to improve the solution by:
        1. Removing an edge and trying to re-add multiple others
        2. Swapping one edge for another
        """
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try removing each edge and seeing if we can add more
            for remove_idx in list(selected):
                if len(selected) <= 1:
                    break
                    
                new_set = set(selected)
                new_set.remove(remove_idx)
                
                # Try to add back edges that were removed or weren't added
                # First, try to add the removed edge back in a different context
                # Actually, let's try a different approach:
                # Remove remove_idx, then try to greedily add edges in a smart order
                
                # Get all edges not in new_set
                remaining = [i for i in range(n_edges) if i not in new_set]
                
                # Sort by some heuristic - edges that conflict with fewer selected edges first
                def conflict_count(idx):
                    count = 0
                    for t in conflicts[idx]:
                        if all(x in new_set for x in t):
                            count += 1
                    return count
                
                # Actually, let's just try random order but multiple times
                for attempt in range(3):
                    test_set = set(new_set)
                    rem = list(remaining)
                    random.shuffle(rem)
                    greedy_fill(test_set, rem)
                    
                    if len(test_set) > len(selected):
                        selected = test_set
                        improved = True
                        break
        
        return selected
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    random.seed(42)
    all_indices = list(range(n_edges))
    
    best_set = set()
    best_size = 0
    
    # Multiple restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initial greedy
        current_set = set()
        shuffled = list(all_indices)
        random.shuffle(shuffled)
        greedy_fill(current_set, shuffled)
        
        # Local search improvement
        current_set = local_search_improve(current_set, max_iterations=200)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Final polish on the best solution
    if time.time() - start_time < time_limit - 10:
        best_set = local_search_improve(best_set, max_iterations=500)
        best_size = len(best_set)
    
    # Convert best_set to list of [u, v]
    result = []
    for idx in best_set:
        result.append(list(edges[idx]))
        
    return result
