import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # seconds, with 5s safety margin before 90s kill

    N = 256

    # 1. Identify valid vertices: i such that 2*i is NOT a perfect square
    valid = []
    for i in range(1, N + 1):
        s = 2 * i
        r = int(s**0.5)
        if r * r != s:
            valid.append(i)

    M = len(valid)

    # Precompute squares up to 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # 2. Build complement graph adjacency (bitset)
    # In the complement graph, an edge exists between i and j if valid[i] + valid[j] is NOT a perfect square
    # We want the maximum clique in this complement graph
    adj = [0] * M

    for i in range(M):
        ui = valid[i]
        for j in range(i + 1, M):
            uj = valid[j]
            if ui + uj not in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # 3. Tomita's maximum clique algorithm with greedy coloring
    best_clique_size = 0
    best_clique_set = set()

    def greedy_color(P):
        """
        Greedy coloring of candidate set P in the complement graph.
        Returns list of color classes and the number of colors used.
        Each color class is a bitset of vertices that can share the same color
        (i.e., are mutually non-adjacent in the complement graph, meaning they
        form an independent set in the complement = clique in original... wait).
        
        Actually, for max clique: we color such that vertices in the same color class
        are NOT adjacent in the graph we're finding clique in (complement graph).
        So same color = independent set in complement graph.
        """
        color_classes = []
        U = P
        while U:
            C = 0
            U_next = U
            while U_next:
                # Pick lowest index in U_next
                v_idx = (U_next & -U_next).bit_length() - 1
                v_bit = 1 << v_idx
                # Check if v is non-adjacent to all in C in the complement graph
                if (adj[v_idx] & C) == 0:
                    C |= v_bit
                    U_next &= ~v_bit
                    U_next &= ~adj[v_idx]
            color_classes.append(C)
            U &= ~C
        return color_classes

    def branch(P, Q, Q_size):
        nonlocal best_clique_size, best_clique_set

        if time.time() - start_time > TIME_LIMIT:
            return

        if not P:
            if Q_size > best_clique_size:
                best_clique_size = Q_size
                best_clique_set = set(Q)
            return

        # Greedy coloring to get upper bound
        color_classes = greedy_color(P)
        num_colors = len(color_classes)
        
        if Q_size + num_colors <= best_clique_size:
            return

        # Process vertices in reverse color order (highest color first)
        # This gives better pruning
        for ci in range(num_colors - 1, -1, -1):
            if time.time() - start_time > TIME_LIMIT:
                return
            if Q_size + ci + 1 <= best_clique_size:
                return
            
            C = color_classes[ci]
            # Process vertices in this color class
            U = C
            while U:
                if time.time() - start_time > TIME_LIMIT:
                    return
                v_idx = (U & -U).bit_length() - 1
                v_bit = 1 << v_idx
                
                # Include v_idx
                new_P = P & adj[v_idx]
                new_Q = Q | {v_idx}
                branch(new_P, new_Q, Q_size + 1)
                
                # Exclude v_idx
                P &= ~v_bit
                U &= ~v_bit

    # Initial call
    initial_P = (1 << M) - 1
    branch(initial_P, set(), 0)

    # Map indices back to values
    result = [valid[i] for i in best_clique_set]

    if not result:
        return [1]

    return result
