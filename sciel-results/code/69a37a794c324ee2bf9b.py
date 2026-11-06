import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 50.0 # seconds
    
    N = 256
    # 1. Identify valid vertices
    valid = []
    for i in range(1, N + 1):
        s = 2 * i
        r = int(s**0.5)
        if r*r != s:
            valid.append(i)
            
    # Map valid vertices to indices 0..M-1
    M = len(valid)
    idx_map = {v: i for i, v in enumerate(valid)}
    
    # 2. Build complement graph adjacency (bitset)
    # adj[i] is bitset of neighbors in complement graph
    adj = [0] * M
    
    # Precompute squares
    squares = set()
    k = 1
    while k*k <= 2 * N:
        squares.add(k*k)
        k += 1
        
    for i in range(M):
        ui = valid[i]
        for j in range(i + 1, M):
            uj = valid[j]
            if ui + uj not in squares:
                # Edge in complement
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Max Clique on Complement Graph
    best_clique_size = 0
    best_clique_set = set()
    
    # Helper for greedy coloring
    def greedy_color(P):
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
                # Check if v is non-adjacent to all in C
                # In complement graph, "non-adjacent" means edge exists in original?
                # No, we are running Max Clique on Complement.
                # In Max Clique, we pick vertices that are mutually adjacent in the graph we are processing.
                # So in greedy_color for Max Clique, we assign same color to non-adjacent vertices in the processing graph.
                # Wait, Tomita's algorithm for Max Clique uses greedy coloring to bound the clique size.
                # The number of colors is an upper bound on the clique size.
                # Standard greedy coloring:
                # Iterate vertices. If vertex is not adjacent to any in current color class, add it.
                # So we need to check adjacency in the processing graph (Complement).
                
                # Check if v_idx is adjacent to any node in C in the Complement graph.
                # adj[v_idx] contains neighbors in Complement.
                # If (adj[v_idx] & C) == 0, then v is NOT adjacent to any in C.
                # This means v can take this color.
                if (adj[v_idx] & C) == 0:
                    C |= v_bit
                    U_next &= ~v_bit
                    # Remove neighbors of v from U_next (they cannot take this color)
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
                # Reconstruct set if needed, but we just need size for score?
                # Task says "Return nonempty subset... Maximize cardinality".
                # The evaluator measures the raw value (cardinality).
                # So we just need to return a set of that size.
                # We can store the set.
                best_clique_set = set(Q)
            return
            
        k = greedy_color(P)
        if Q_size + k <= best_clique_size:
            return
            
        # Select vertex with highest color? Or just max degree?
        # Tomita selects vertex with highest color in the greedy coloring.
        # For simplicity and speed, selecting max degree in P is often good.
        # Let's try selecting the vertex with the highest index in the color assignment?
        # Actually, let's just pick the vertex with the max degree in P.
        
        max_deg = -1
        v_sel = -1
        temp_P = P
        while temp_P:
            v_idx = (temp_P & -temp_P).bit_length() - 1
            v_bit = 1 << v_idx
            deg = (adj[v_idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_sel = v_idx
                v_sel_bit = v_bit
            temp_P &= ~v_bit
            
        if v_sel == -1:
            return
            
        v_sel_bit = 1 << v_sel
        
        # Branch 1: Include v_sel
        # New P is intersection of current P and neighbors of v_sel in Complement
        new_P = P & adj[v_sel]
        new_Q = Q | {v_sel}
        branch(new_P, new_Q, Q_size + 1)
        
        # Branch 2: Exclude v_sel
        new_P2 = P & ~v_sel_bit
        branch(new_P2, Q, Q_size)
        
    # Initial call
    # P is all vertices
    initial_P = (1 << M) - 1
    branch(initial_P, set(), 0)
    
    # Map indices back to values
    result = [valid[i] for i in best_clique_set]
    
    if not result:
        # Fallback, though unlikely
        return [1]
        
    return result
