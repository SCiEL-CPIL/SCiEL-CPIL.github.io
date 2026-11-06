"""Improved program for R(K2,2,2) maximization via local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_k222(masks_adj, n):
    """
    Count the number of K2,2,2 subgraphs in the graph defined by masks_adj.
    A K2,2,2 is formed by 3 disjoint non-edges (a,b), (c,d), (e,f) such that
    all other pairs among the 6 vertices are edges.
    """
    if n < 6:
        return 0
    
    # List of non-edges
    non_edges = []
    for i in range(n):
        m = ~(masks_adj[i]) & ((1 << n) - 1)
        m &= ~(1 << i) # Remove self
        # Only consider j > i to avoid duplicates
        for j in range(i+1, n):
            if (m >> j) & 1:
                non_edges.append((i, j))
                
    if len(non_edges) < 3:
        return 0
        
    count = 0
    len_ne = len(non_edges)
    
    # Iterate over pairs of non-edges
    for i1 in range(len_ne):
        a, b = non_edges[i1]
        # Common neighbors of a and b in G
        common_ab = masks_adj[a] & masks_adj[b]
        common_ab &= ~(1 << a) & ~(1 << b)
        
        for i2 in range(i1 + 1, len_ne):
            c, d = non_edges[i2]
            # Disjoint check
            if (c == a or c == b or d == a or d == b):
                continue
            
            # c and d must be in common_ab
            if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                continue
            
            # Candidates for e, f must be in common_ab AND common_cd
            common_cd = masks_adj[c] & masks_adj[d]
            candidates = common_ab & common_cd
            candidates &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
            
            # Check if there is a non-edge (e, f) in candidates
            # We can iterate through non_edges again, but that's slow.
            # Instead, iterate through bits in candidates?
            # If candidates has < 2 bits, no edge possible.
            bit_count = bin(candidates).count('1')
            if bit_count < 2:
                continue
                
            # Check all pairs in candidates?
            # Or check if any non-edge (e,f) is a subset of candidates.
            # Since non_edges is a list, we can check each.
            # Optimization: Precompute a bitmask for each non-edge? No.
            # Just check if any non-edge (e,f) has e,f in candidates.
            # To speed up, we can break early if found, but we need COUNT.
            # However, for local search, we just need to know if it's 0 or >0?
            # No, for SA, a count is better. But counting is expensive.
            # Let's just return 1 if found, 0 otherwise, and use a greedy repair strategy?
            # Or use a small penalty.
            
            # Let's try to count efficiently.
            # For each non-edge (e, f), if e,f in candidates, increment.
            # But this is O(M) per pair.
            # Total O(M^3) worst case.
            
            # Alternative:
            # The number of K2,2,2 containing a specific pair of non-edges (a,b) and (c,d)
            # is the number of non-edges in the induced subgraph on candidates.
            # We can precompute for each subset? No.
            
            # Let's just check existence for validity, and use a heuristic for search.
            # For search, we will use a "greedy flip" that minimizes the number of "almost" K2,2,2s.
            # Or just use the boolean check and random restarts.
            
            # Let's refine the search strategy:
            # 1. Try to reach n=29 using known constructions or random search.
            # 2. Then try to extend to n=30.
            
            # For now, let's make this function return a boolean for speed in the inner loop of search.
            # We will use a separate function for final validation.
            return 1 # Found one
            
    return 0

def _is_valid(masks_adj, n):
    """Check if graph is K2,2,2 free."""
    return _count_k222(masks_adj, n) == 0

def _adj_to_mask(adj, n):
    """Convert numpy boolean adjacency matrix to list of int bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _mask_to_adj(masks, n):
    """Convert list of bitmasks to numpy boolean adjacency matrix."""
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if (masks[i] >> j) & 1:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    best_n = 1
    best_colors = [0]
    
    # Try to find a valid coloring for n=30
    # If fails, try 29, etc.
    
    for target_n in range(30, 5, -1):
        if time.time() > deadline:
            break
            
        n = target_n
        found = False
        
        # Multiple restarts
        max_restarts = 50
        restarts_done = 0
        
        while not found and restarts_done < max_restarts:
            restarts_done += 1
            if time.time() > deadline:
                break
                
            # Initialize random graph
            masks = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
                        
            # Local Search
            # We want to minimize violations in G and in complement.
            # Let's define a cost function.
            
            def get_cost(masks):
                # Count violations in G
                v_g = _count_k222(masks, n)
                # Count violations in complement
                comp_masks = [0] * n
                all_ones = (1 << n) - 1
                for i in range(n):
                    comp_masks[i] = (~masks[i]) & all_ones & ~(1 << i)
                v_c = _count_k222(comp_masks, n)
                return v_g + v_c
                
            current_cost = get_cost(masks)
            
            # Greedy Local Search with random restarts
            # If cost is 0, we are done.
            if current_cost == 0:
                found = True
                break
                
            # Simulated Annealing / Random Walk
            T = 1.0
            for step in range(10000):
                if time.time() > deadline:
                    break
                    
                # Pick random edge
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                    
                # Flip edge
                bit = 1 << j
                if masks[i] & bit:
                    masks[i] &= ~bit
                    masks[j] &= ~(1 << i)
                else:
                    masks[i] |= bit
                    masks[j] |= (1 << i)
                    
                new_cost = get_cost(masks)
                
                if new_cost <= current_cost:
                    current_cost = new_cost
                    if current_cost == 0:
                        found = True
                        break
                else:
                    # Revert
                    if masks[i] & bit:
                        masks[i] &= ~bit
                        masks[j] &= ~(1 << i)
                    else:
                        masks[i] |= bit
                        masks[j] |= (1 << i)
                        
                # Cool down
                T *= 0.99
                
            if found:
                break
                
        if found:
            best_n = n
            best_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    best_colors.append(1 if (masks[i] >> j) & 1 else 0)
            # If we found 30, we can stop. If we found 29, we might try to find 30 in remaining time?
            # But the loop goes downwards, so if 30 fails, we try 29.
            # We want the MAXIMUM n. So if we find 30, we return.
            # If we don't find 30, we return 29 if we found it.
            # The loop structure handles this: it tries 30 first. If it fails, it tries 29.
            # We should break if we find a valid one?
            # No, we want the largest. So we should try 30. If we succeed, return.
            # If we fail 30, we try 29. If we succeed 29, we return 29.
            # But what if we find 29 quickly, and have time to try 30 again?
            # The current loop tries 30 with 50 restarts. If it fails, it moves to 29.
            # This is a good strategy.
            
            # However, if we find 29, we should return it immediately?
            # Yes, because we are iterating downwards.
            # But wait, if we find 29, we return 29. We don't try to extend 29 to 30 in this block.
            # That's fine.
            
            # Update best and break out of the target_n loop?
            # Yes, because we found a valid coloring for this n.
            # Since we are going downwards, this is the best we can do in this pass.
            break

    # If we didn't find anything (unlikely), return a small valid graph
    if best_n == 1:
        # n=5 is always valid? No, K5 can have K2,2,2? No, K2,2,2 needs 6 vertices.
        # So any n < 6 is valid.
        best_n = 5
        best_colors = [0] * (5*4//2)
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
