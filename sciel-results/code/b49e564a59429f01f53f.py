import time
import math

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # seconds - leave 5s safety margin for 60s limit
    
    N = 256
    # 1. Identify valid vertices (2*i must not be a perfect square)
    valid = []
    for i in range(1, N + 1):
        s = 2 * i
        r = int(math.isqrt(s))
        if r * r != s:
            valid.append(i)
            
    M = len(valid)
    
    # 2. Build complement graph adjacency (bitset)
    # adj[i] is bitset of neighbors in complement graph (edges where sum is NOT a square)
    adj = [0] * M
    
    # Precompute squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    for i in range(M):
        ui = valid[i]
        for j in range(i + 1, M):
            uj = valid[j]
            if ui + uj not in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Compute degree for each vertex
    deg = [0] * M
    for i in range(M):
        deg[i] = adj[i].bit_count()
    
    # Order vertices by degree (descending) for better branching
    # Reindex so high-degree vertices come first
    order = sorted(range(M), key=lambda x: -deg[x])
    rev_order = [0] * M
    for new_idx, old_idx in enumerate(order):
        rev_order[old_idx] = new_idx
    
    # Rebuild adjacency with new ordering
    adj_new = [0] * M
    for i in range(M):
        old_i = order[i]
        for j in range(i + 1, M):
            old_j = order[j]
            if adj[old_i] & (1 << old_j):
                adj_new[i] |= (1 << j)
                adj_new[j] |= (1 << i)
    
    adj = adj_new
    deg = [0] * M
    for i in range(M):
        deg[i] = adj[i].bit_count()
    
    # Map from new index to original value
    val_map = [valid[order[i]] for i in range(M)]
    
    # 3. Max Clique on Complement Graph using Tomita-style branch and bound
    best_clique_size = 0
    best_clique_set = set()
    
    def greedy_color(P):
        """Greedy coloring to get upper bound on clique size in P."""
        colors = 0
        U = P
        while U:
            colors += 1
            C = 0
            U_next = U
            while U_next:
                # Pick lowest index in U_next
                v_idx = (U_next & -U_next).bit_length() - 1
                v_bit = 1 << v_idx
                # v can take this color if not adjacent to any vertex in C
                if (adj[v_idx] & C) == 0:
                    C |= v_bit
                    U_next &= ~v_bit
                    U_next &= ~adj[v_idx]
            U &= ~C
        return colors
    
    def branch(P, Q, Q_size):
        nonlocal best_clique_size, best_clique_set
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if not P:
            if Q_size > best_clique_size:
                best_clique_size = Q_size
                best_clique_set = set(Q)
            return
            
        # Get upper bound via greedy coloring
        k = greedy_color(P)
        if Q_size + k <= best_clique_size:
            return
            
        # Select vertex with highest degree in P
        max_deg = -1
        v_sel = -1
        temp_P = P
        while temp_P:
            v_idx = (temp_P & -temp_P).bit_length() - 1
            v_bit = 1 << v_idx
            d = (adj[v_idx] & P).bit_count()
            if d > max_deg:
                max_deg = d
                v_sel = v_idx
                v_sel_bit = v_bit
            temp_P &= ~v_bit
            
        if v_sel == -1:
            return
            
        # Branch 1: Include v_sel
        new_P = P & adj[v_sel]
        new_Q = Q | {v_sel}
        branch(new_P, new_Q, Q_size + 1)
        
        # Branch 2: Exclude v_sel
        new_P2 = P & ~(1 << v_sel)
        branch(new_P2, Q, Q_size)
    
    # Initial call
    initial_P = (1 << M) - 1
    branch(initial_P, set(), 0)
    
    # Map indices back to values
    result = [val_map[i] for i in best_clique_set]
    
    if not result:
        return [1]
        
    return result
