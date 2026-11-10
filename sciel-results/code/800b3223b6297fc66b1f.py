import time
import random
import math

def construct():
    deadline = time.time() + 115
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Fast Q3 check using bitmasks
    def creates_q3_fast(u, v, adj):
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def try_add(adj, u, v):
        if adj[u] & (1 << v):
            return False
        if not creates_q3_fast(u, v, adj):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def simulated_annealing(adj, time_limit):
        start_time = time.time()
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        # Initial temperature
        T = 1.0
        cooling = 0.995
        
        # Precompute all edges for quick access
        all_edges_set = set()
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    all_edges_set.add((u, v))
                mask -= bit
        
        missing_edges = [e for e in all_edges if e not in all_edges_set]
        
        while time.time() < start_time + time_limit:
            if not missing_edges and current_count == 120: # Max possible
                break
            
            # Choose a move
            move_type = random.random()
            
            new_adj = adj[:]
            delta = 0
            valid_move = False
            
            if move_type < 0.6 and missing_edges:
                # Try to add an edge
                idx = random.randint(0, len(missing_edges) - 1)
                u, v = missing_edges[idx]
                if try_add(new_adj, u, v):
                    delta = 1
                    valid_move = True
                    # Update missing_edges list? 
                    # It's expensive to update the list every time.
                    # We'll just keep the list and filter invalid adds later or accept that we might retry same edge.
                    # Better: remove from list if added.
                    missing_edges.pop(idx)
                    all_edges_set.add((u, v))
                    
            elif move_type < 0.8 and current_count > 5:
                # Remove a random edge
                # Pick a random existing edge
                # We can pick a random u and random neighbor
                u = random.randint(0, n - 1)
                mask = adj[u]
                if mask:
                    # Pick random bit
                    bit_count = mask.bit_count()
                    # Simple way: iterate
                    bits = []
                    m = mask
                    while m:
                        bit = m & (-m)
                        bits.append(bit)
                        m -= bit
                    if bits:
                        chosen_bit = random.choice(bits)
                        v = chosen_bit.bit_length() - 1
                        if u != v:
                            remove_edge(new_adj, u, v)
                            delta = -1
                            valid_move = True
                            # Update lists
                            edge = (min(u, v), max(u, v))
                            if edge in all_edges_set:
                                all_edges_set.remove(edge)
                                missing_edges.append(edge)
                            
            else:
                # Swap: Remove one, try to add another
                # Pick random edge to remove
                u = random.randint(0, n - 1)
                mask = adj[u]
                if mask:
                    bits = []
                    m = mask
                    while m:
                        bit = m & (-m)
                        bits.append(bit)
                        m -= bit
                    if bits:
                        chosen_bit = random.choice(bits)
                        v = chosen_bit.bit_length() - 1
                        if u != v:
                            remove_edge(new_adj, u, v)
                            edge_out = (min(u, v), max(u, v))
                            all_edges_set.remove(edge_out)
                            missing_edges.append(edge_out)
                            
                            # Try to add a random missing edge
                            if missing_edges:
                                idx = random.randint(0, len(missing_edges) - 1)
                                # Note: the edge we just removed is now in missing_edges
                                # So we might re-add it. That's fine (delta 0).
                                u2, v2 = missing_edges[idx]
                                if try_add(new_adj, u2, v2):
                                    delta = 0 # Removed 1, Added 1
                                    valid_move = True
                                    missing_edges.pop(idx)
                                    all_edges_set.add((u2, v2))
                                else:
                                    # Failed to add, revert removal? 
                                    # Or just accept the removal (delta -1)
                                    # Let's accept the removal as a "remove" move
                                    delta = -1
                                    valid_move = True
                                    # The edge remains in missing_edges
                                    # We don't pop it
                                    # But we already added it to missing_edges above.
                                    # So it's consistent.
                                    
            if valid_move:
                new_count = current_count + delta
                if delta > 0:
                    # Always accept improvement
                    adj = new_adj
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # Accept worse with probability
                    if random.random() < math.exp(delta / T):
                        adj = new_adj
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                
                T *= cooling
                if T < 0.01:
                    T = 0.01 # Floor
            else:
                # Move was not valid (e.g. tried to add but created Q3, or no edges to remove)
                # Do nothing, just cool slightly? Or retry?
                T *= 0.999

        return best_adj, best_count

    best_adj = [0] * n
    best_count = 0
    
    # Phase 1: Greedy Initialization
    # Try multiple random orderings of edges to build a good initial solution
    initializers = []
    
    # 1. Natural order
    adj_nat = [0] * n
    for u, v in all_edges:
        try_add(adj_nat, u, v)
    initializers.append(adj_nat)
    
    # 2. Random orders
    for _ in range(10):
        adj_rand = [0] * n
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        for u, v in shuffled_edges:
            try_add(adj_rand, u, v)
        initializers.append(adj_rand)
        
    # 3. Turan T(16,4) based? 
    # Turan graphs are dense. We can try to prune them.
    # But greedy add from empty is usually better for Q3-free.
    
    # Refine each initializer with SA
    for init in initializers:
        if time.time() > deadline - 20:
            break
        adj, cnt = simulated_annealing(init, 10)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Final polish with longer SA on best
    if time.time() < deadline - 5:
        adj, cnt = simulated_annealing(best_adj, time.time() and (deadline - 5 - time.time()) or 10)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj

    return edges_from_adj(best_adj)
