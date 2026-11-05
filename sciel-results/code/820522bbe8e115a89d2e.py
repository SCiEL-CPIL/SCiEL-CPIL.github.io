import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Identify valid vertices (2*a is not a square)
    valid_indices = []
    valid_values = []
    for i in range(N):
        a = i + 1
        if 2 * a not in squares:
            valid_indices.append(i)
            valid_values.append(a)
            
    M = len(valid_indices)
    
    # Build adjacency masks
    # adj_mask[i] is a bitmask where bit j is 1 if valid_values[i] + valid_values[j] is a square
    adj_mask = [0] * M
    for i in range(M):
        a = valid_values[i]
        mask = 0
        for j in range(M):
            if i != j:
                b = valid_values[j]
                if a + b in squares:
                    mask |= (1 << j)
        adj_mask[i] = mask
        
    # Sort vertices by degree (descending) to improve coloring bound effectiveness
    # We need to remap indices
    degrees = [bin(adj_mask[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda x: -degrees[x])
    
    # Remap adjacency masks to the new order
    # new_index[old_index] = new_position
    new_index = [0] * M
    for new_pos, old_idx in enumerate(order):
        new_index[old_idx] = new_pos
        
    new_adj_mask = [0] * M
    for old_idx in range(M):
        old_mask = adj_mask[old_idx]
        new_mask = 0
        # Iterate through bits in old_mask
        m = old_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            old_j = lsb.bit_length() - 1
            new_j = new_index[old_j]
            new_mask |= (1 << new_j)
            m ^= lsb
        new_adj_mask[new_index[old_idx]] = new_mask
        
    # Update M if necessary (it's the same)
    # Now adj_mask[i] corresponds to the i-th vertex in the sorted order
    
    best_size = 0
    best_set_mask = 0 # Bitmask of the best set in the sorted order
    
    # Helper to get size of a mask
    def popcount(x):
        return x.bit_count()
    
    # Greedy Coloring for Upper Bound (McConnell's algorithm simplified for bitmask)
    # Returns the number of colors used, which is an upper bound on MIS size in the induced subgraph
    def color_bound(P):
        if P == 0:
            return 0
        
        colors = 0
        U = P
        # We assign colors greedily. 
        # In standard MIS B&B, we often color the candidates P.
        # The number of colors is an upper bound on the size of the independent set in P.
        
        # To make the bound tighter, we should color in an order that minimizes colors?
        # Actually, for the bound, we want a *valid* coloring. 
        # The standard approach in Tomita's algorithm:
        # 1. Sort vertices in P by degree (or some heuristic).
        # 2. Assign colors such that adjacent vertices have different colors.
        # 3. The max color index is the bound.
        
        # Since M is small (<=256), we can do a simple greedy coloring on the induced subgraph P.
        # However, doing this from scratch at every node is expensive.
        # A common optimization is to use the "coloring bound" where we maintain the colors.
        # But for simplicity and robustness with bitmasks, let's implement a fast greedy coloring.
        
        # Heuristic: Process vertices in P in decreasing order of degree within P?
        # Or just iterate?
        # Let's iterate through vertices in P.
        
        # We need to know the neighbors of each vertex within P.
        # color[v] = c
        # For a new vertex v, it cannot have the same color as any neighbor u in P that already has a color.
        
        # This is O(M^2) per call. M~250. 250^2 = 62500 ops. 
        # If we have many nodes, this might be slow.
        # However, bit operations are fast.
        
        # Alternative: Use the simple bound len(P) first. If size + len(P) <= best, prune.
        # Only use coloring if that doesn't prune.
        
        # Let's implement a very fast greedy coloring.
        # We'll create a list of vertices in P.
        verts = []
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        # Sort by degree in P (descending) might help coloring quality (fewer colors)
        # But calculating degree in P for each is expensive.
        # Let's just use the static order (which is already sorted by global degree desc).
        
        color_map = {}
        max_color = 0
        
        for v in verts:
            # Find colors used by neighbors of v in P
            neighbor_colors = set()
            # Get neighbors of v in P
            nbrs = adj_mask[v] & P
            nm = nbrs
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                if u in color_map:
                    neighbor_colors.add(color_map[u])
                nm ^= lsb
                
            c = 1
            while c in neighbor_colors:
                c += 1
            color_map[v] = c
            if c > max_color:
                max_color = c
                
        return max_color

    def expand(P, size):
        nonlocal best_size, best_set_mask
        nonlocal start_time
        
        if time.time() - start_time > DEADLINE:
            return
            
        if P == 0:
            if size > best_size:
                best_size = size
                best_set_mask = 0 # We need to track the actual set, not just size.
                # Wait, my expand function doesn't track the set, only the size.
                # I need to track the set to return it.
                # Passing a list/set is slow. Passing a bitmask is fast.
                # Let's pass the current set as a bitmask too.
            return

        # Simple bound
        p_size = popcount(P)
        if size + p_size <= best_size:
            return
            
        # Coloring bound
        # To save time, only compute coloring bound if p_size is significant
        # or if simple bound is close.
        bound = color_bound(P)
        if size + bound <= best_size:
            return
            
        # Select vertex to branch on
        # Heuristic: Pick vertex with highest degree in P
        # Or lowest? 
        # In MIS, including a high-degree vertex removes many options, often leading to smaller sets.
        # So we usually prefer to *exclude* high-degree vertices or branch on them carefully.
        # Standard Tomita: Pick vertex with highest degree in P? 
        # Actually, for the "Include" branch, we remove neighbors.
        # For the "Exclude" branch, we keep them.
        # If we pick a high degree vertex:
        #   Include: P_new = P \ ({v} U N(v)). Large reduction.
        #   Exclude: P_new = P \ {v}. Small reduction.
        # This creates an imbalance.
        
        # Let's pick a vertex with max degree in P.
        v = -1
        max_deg = -1
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = popcount(adj_mask[idx] & P)
            if deg > max_deg:
                max_deg = deg
                v = idx
            m ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        # New P: Remove v and its neighbors from P
        P_include = P & ~(adj_mask[v] | (1 << v))
        expand(P_include, size + 1)
        
        # Branch 2: Exclude v
        P_exclude = P & ~(1 << v)
        expand(P_exclude, size)

    # The expand function above doesn't track the actual set for the best solution.
    # Let's modify it to track the set mask.
    
    best_set_mask_global = 0
    
    def expand_with_set(P, current_set_mask, size):
        nonlocal best_size, best_set_mask_global
        nonlocal start_time
        
        if time.time() - start_time > DEADLINE:
            return
            
        if P == 0:
            if size > best_size:
                best_size = size
                best_set_mask_global = current_set_mask
            return

        p_size = popcount(P)
        if size + p_size <= best_size:
            return
            
        bound = color_bound(P)
        if size + bound <= best_size:
            return
            
        # Select vertex
        v = -1
        max_deg = -1
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = popcount(adj_mask[idx] & P)
            if deg > max_deg:
                max_deg = deg
                v = idx
            m ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        P_include = P & ~(adj_mask[v] | (1 << v))
        expand_with_set(P_include, current_set_mask | (1 << v), size + 1)
        
        # Branch 2: Exclude v
        P_exclude = P & ~(1 << v)
        expand_with_set(P_exclude, current_set_mask, size)

    # Initial Greedy to get a good lower bound
    # Greedy MIS: Pick vertex with min degree in remaining? 
    # Or max degree? 
    # For MIS, picking min degree is often better for size.
    def greedy_mis():
        P = (1 << M) - 1
        S = 0
        while P:
            # Find vertex with min degree in P
            min_deg = M
            v = -1
            m = P
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                deg = popcount(adj_mask[idx] & P)
                if deg < min_deg:
                    min_deg = deg
                    v = idx
                m ^= lsb
            S |= (1 << v)
            P &= ~(adj_mask[v] | (1 << v))
        return S, popcount(S)
        
    init_set, init_size = greedy_mis()
    best_size = init_size
    best_set_mask_global = init_set
    
    # Randomized Restarts
    # Shuffle the order of vertices? 
    # The current order is sorted by degree. 
    # If we shuffle, the "highest degree" heuristic changes.
    # However, adj_mask is fixed. 
    # We can simulate different orders by permuting the bits? 
    # That's complex. 
    # Instead, we can just run the B&B once with the sorted order.
    # If time permits, we can try a few random perturbations.
    
    # Given the speed of bitmasks, the single B&B run should be very fast.
    # Let's run it.
    
    initial_P = (1 << M) - 1
    expand_with_set(initial_P, 0, 0)
    
    # If time remains, try a few random restarts with different vertex selection heuristics?
    # Or just check if we can improve the greedy start?
    
    # Map back to original values
    # best_set_mask_global contains bits corresponding to the sorted order.
    # We need to map these back to original indices.
    
    result_indices = []
    m = best_set_mask_global
    while m:
        lsb = m & -m
        new_idx = lsb.bit_length() - 1
        # new_idx is the index in the sorted order.
        # order[new_idx] is the old index.
        old_idx = order[new_idx]
        # old_idx is the index in valid_indices.
        # valid_indices[old_idx] is the 0-based index in [0, 255].
        # We want the value, which is valid_indices[old_idx] + 1.
        val = valid_indices[old_idx] + 1
        result_indices.append(val)
        m ^= lsb
        
    return result_indices

# EVOLVE-BLOCK-END
