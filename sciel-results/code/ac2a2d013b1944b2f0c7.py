import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    # Helper: Check if bitmask represents an independent set
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        # Common neighbors must form an independent set
        return is_independent(common, adj)

    # Exact B&B for largest triangle-free induced subgraph
    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound - try multiple orderings
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        # Ordering 1: degree descending
        order1 = sorted(range(n), key=lambda x: -degrees[x])
        greedy_set = 0
        for v in order1:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        # Ordering 2: degree ascending
        order2 = sorted(range(n), key=lambda x: degrees[x])
        greedy_set2 = 0
        for v in order2:
            if is_independent(adj[v] & greedy_set2, adj):
                greedy_set2 |= (1 << v)
        if bin(greedy_set2).count('1') > best[0]:
            best[0] = bin(greedy_set2).count('1')
        
        # Ordering 3: random
        random_order = list(range(n))
        random.shuffle(random_order)
        greedy_set3 = 0
        for v in random_order:
            if is_independent(adj[v] & greedy_set3, adj):
                greedy_set3 |= (1 << v)
        if bin(greedy_set3).count('1') > best[0]:
            best[0] = bin(greedy_set3).count('1')
        
        # B&B
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            # Upper bound
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            # Branch 1: Include v (try first if likely to extend)
            # Check if S ∪ {v} is triangle-free
            # This requires N(v) ∩ S to be independent
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
            
            # Branch 2: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def adj_from_parts(parts):
        """Build adjacency list from a 3-partition of vertices."""
        adj = [0] * n
        # No edges within parts, all edges between different parts
        for i in range(n):
            for j in range(i + 1, n):
                if parts[i] != parts[j]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    random.seed(42)
    best_adj = [0] * n
    best_score = n  # Max possible is n

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # --- Phase 1: Turan-based local search ---
    # Start with balanced 3-partition
    parts = [i % 3 for i in range(n)]
    
    # Try a few initial partitions
    initial_partitions = [
        [i % 3 for i in range(n)],  # balanced 6-6-6
        [0]*6 + [1]*6 + [2]*6,      # block
        [0,1,2]*6,                  # interleaved
    ]
    
    for init_parts in initial_partitions:
        if time.time() > deadline - 20.0:
            break
        parts = init_parts[:]
        adj = adj_from_parts(parts)
        
        # Evaluate initial
        score = largest_tf(adj, time_limit=deadline - 15.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
        
        # Local search: swap vertices between parts
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            # Try all pairs of vertices from different parts
            for i in range(n):
                if time.time() > deadline - 15.0:
                    break
                for j in range(i + 1, n):
                    if parts[i] == parts[j]:
                        continue
                    # Swap parts[i] and parts[j]
                    old_i, old_j = parts[i], parts[j]
                    parts[i], parts[j] = parts[j], parts[i]
                    
                    new_adj = adj_from_parts(parts)
                    new_score = largest_tf(new_adj, time_limit=deadline - 15.0)
                    
                    if new_score < score:
                        score = new_score
                        adj = new_adj
                        if score < best_score:
                            best_score = score
                            best_adj = adj[:]
                        improved = True
                    else:
                        # Revert
                        parts[i], parts[j] = old_i, old_j
                        adj = adj_from_parts(parts)
        
        # Also try random perturbations of the partition
        for _ in range(10):
            if time.time() > deadline - 15.0:
                break
            # Move a random vertex to a random part
            v = random.randint(0, n - 1)
            old_part = parts[v]
            new_part = random.choice([p for p in range(3) if p != old_part])
            parts[v] = new_part
            adj = adj_from_parts(parts)
            score = largest_tf(adj, time_limit=deadline - 15.0)
            if score < best_score:
                best_score = score
                best_adj = adj[:]

    # --- Phase 2: General random restart with K4-free greedy fill ---
    # Use the best found so far as a starting point for further optimization
    while time.time() < deadline - 3.0:
        iteration_adj = best_adj[:]
        
        # Random edge ordering
        random.shuffle(all_edges)
        
        # Greedy fill with K4 constraint (start from current best)
        adj = iteration_adj[:]
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Evaluate
        t_start = time.time()
        score = largest_tf(adj, time_limit=deadline - 3.0)
        t_end = time.time()
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"New best: {best_score}, time taken: {t_end-t_start:.2f}s", flush=True)
            
        # Local Search: Try to add more edges while maintaining K4-free
        if time.time() < deadline - 8.0 and score <= best_score:
            improved = True
            while improved and time.time() < deadline - 8.0:
                improved = False
                random.shuffle(all_edges)
                for u, v in all_edges:
                    if not ((adj[u] >> v) & 1):
                        if can_add_edge(adj, u, v):
                            # Tentatively add
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            
                            # Check score periodically
                            if random.random() < 0.2:
                                new_score = largest_tf(adj, time_limit=deadline - 3.0)
                                if new_score > score:
                                    # Revert
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                else:
                                    if new_score < best_score:
                                        best_score = new_score
                                        best_adj = adj[:]
                                        improved = True
                                    score = min(score, new_score)
                            else:
                                # Keep edge, assume denser tends to be better
                                pass
                            
                            if time.time() > deadline - 8.0:
                                break

    edges = get_edge_list(best_adj)
    print(f"Final: score={best_score}, edges={len(edges)}", flush=True)
    return edges
