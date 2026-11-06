import time
import random
from itertools import combinations

def is_2_colorable_brute(n, edges):
    if not edges:
        return True
        
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    for c in range(1 << n):
        is_valid = True
        for em in edge_masks:
            if (em & c) == 0 or (em & c) == em:
                is_valid = False
                break
        if is_valid:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Seed solution
    seed_edges = [list(e) for e in combinations(range(9), 5)]
    best_edges = seed_edges
    best_n = 9
    best_count = len(best_edges)
    
    # Search for better solutions
    # We focus on n up to 15 for speed
    for n in range(9, 16):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time
        # Estimate time per restart: 
        # 1. Initialization: fast
        # 2. Filling to non-2-colorable: few checks
        # 3. Greedy deletion: ~E checks
        
        restarts = 5
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Random initial hypergraph
            # Start with a sparse graph and add edges?
            # Or start dense and remove?
            # Start dense is safer to ensure non-2-colorability
            
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Start with 80% of edges
            k = int(0.8 * len(all_possible))
            edges = [list(e) for e in all_possible[:k]]
            
            # Ensure it is non-2-colorable
            # If it is, add edges
            while is_2_colorable_brute(n, edges):
                if time.time() > deadline:
                    break
                # Add random edge
                # Find an edge not in edges
                # For small n, we can just pick from all_possible
                # and check if in edges
                # To speed up, maintain a set
                edge_set = set(tuple(sorted(e)) for e in edges)
                while True:
                    cand = list(random.choice(all_possible))
                    cand_tup = tuple(sorted(cand))
                    if cand_tup not in edge_set:
                        edges.append(cand)
                        edge_set.add(cand_tup)
                        break
                else:
                    break # No more edges to add
            
            if time.time() > deadline:
                break
                
            if is_2_colorable_brute(n, edges):
                continue
                
            # Greedy deletion
            # We will try to remove edges one by one
            # To avoid index issues, we work with a list and swap-pop
            
            current_edges = [list(e) for e in edges]
            
            # Shuffle indices
            indices = list(range(len(current_edges)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                # If the edge at idx has already been removed, skip
                # Since we swap-pop, the list shrinks
                # We need to track if this specific edge is still present
                # This is hard with simple indices.
                
                # Alternative:
                # Iterate through a copy of the edges list
                # For each edge, try to remove it from the main list
                # If successful, continue
                # If not, leave it
                
                # Let's use a set for fast removal and lookup
                # But we need to return a list of lists
                
                # Let's just do a simple loop
                # We try to remove each edge in the current list
                # We iterate backwards to avoid index shifting issues?
                # No, we want to try all edges.
                
                # Let's use a different strategy:
                # While there are edges:
                # Pick a random edge.
                # Try to remove it.
                # If removed, continue.
                # If not removed, add it to a "blocked" set.
                # If all edges are blocked, stop.
                
                # This is a local search.
                
                # For simplicity and robustness, let's just do a single pass
                # of trying to remove edges in a random order.
                
                pass # This block is replaced by the set-based approach below

            # Set-based deletion
            edge_set = set(tuple(sorted(e)) for e in edges)
            initial_set = set(edge_set)
            
            for edge_tuple in initial_set:
                if time.time() > deadline:
                    break
                
                if edge_tuple not in edge_set:
                    continue
                
                # Try to remove
                new_set = edge_set - {edge_tuple}
                new_list = [list(e) for e in new_set]
                
                if not is_2_colorable_brute(n, new_list):
                    edge_set = new_set
            
            final_edges = [list(e) for e in edge_set]
            if not is_2_colorable_brute(n, final_edges):
                if len(final_edges) < best_count:
                    best_count = len(final_edges)
                    best_edges = final_edges
                    best_n = n

    return {'n': best_n, 'edges': best_edges}
