import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit

    n = 40

    # Precompute bit masks for fast exclusion
    all_mask = (1 << n) - 1

    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) to the graph represented by adj creates a C6.
        Returns True if it is safe to add (no C6 created), False otherwise.
        """
        # Neighbors of u excluding v
        nu_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        # We need to check if there is a path of length 3 between u and v
        # i.e., u-a-b-c-d-v where a in N(u)\{v}, d in N(v)\{u}
        # and a-b-c-d is a path in G \ {u,v}
        
        # For each pair (a, d) with a in N(u)\{v}, d in N(v)\{u}, a != d:
        #   Check if there exists b in N(a)\{u,v,d,a} and c in N(d)\{u,v,a,d} with b~c
        
        exclude_uv = ~(1 << u) & ~(1 << v)
        
        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a
            
            # b candidates: neighbors of a, excluding u, v, a, and all d in N(v)\{u}
            # We'll handle d-specific exclusion in the inner loop
            b_base = adj[ai] & exclude_uv & ~lb_a  # exclude u, v, a
            
            d = nv_mask & ~lb_a  # d != a
            while d:
                ld_d = d & (-d)
                di = ld_d.bit_length() - 1
                d ^= ld_d
                
                # b candidates: neighbors of a, excluding u, v, a, d
                b_candidates = b_base & ~ld_d
                
                # c candidates: neighbors of d, excluding u, v, a, d
                c_candidates = adj[di] & exclude_uv & ~lb_a & ~ld_d
                
                # Check if any neighbor of any b in b_candidates is in c_candidates
                # Compute union of neighbors of all b in b_candidates
                union_neighbors = 0
                bm = b_candidates
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    union_neighbors |= adj[bi]
                    
                    # Early exit if we already found a connection
                    if union_neighbors & c_candidates:
                        return False
        
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Initialize base structure: 8 disjoint K5s
    base_adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
    
    # Pairs already in base_adj
    existing_edges_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (base_adj[u] >> v) & 1:
                existing_edges_set.add((u, v))
    
    # Pairs to consider for addition
    candidate_pairs = [(u, v) for u, v in all_pairs if (u, v) not in existing_edges_set]
    
    best_edges_list = []
    best_count = 0
    
    # Phase 1: Greedy Random Restarts
    num_greedy_restarts = 30
    
    for restart in range(num_greedy_restarts):
        if time.time() > deadline - 40:  # Reserve at least 40s for local search
            break
            
        current_adj = base_adj[:]
        
        # Shuffle candidate pairs with different seeds
        random.seed(42 + restart * 7)
        random.shuffle(candidate_pairs)
        
        for u, v in candidate_pairs:
            if time.time() > deadline - 40:
                break
            if not ((current_adj[u] >> v) & 1):
                if can_add_edge(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        
        current_count = count_edges(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (current_adj[u] >> v) & 1:
                        best_edges_list.append([u, v])
            
            if best_count >= 105:
                break

    # Phase 2: Intensive Local Search from best solution
    if best_edges_list:
        current_adj = [0] * n
        for u, v in best_edges_list:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
        current_count = best_count
    else:
        current_adj = base_adj[:]
        current_count = 80

    # Build edge lists for local search
    existing_edges_list = []
    non_existing_edges_list = []
    for u, v in all_pairs:
        if (current_adj[u] >> v) & 1:
            existing_edges_list.append((u, v))
        else:
            non_existing_edges_list.append((u, v))
    
    # Local search: repeatedly try to add edges, and use swaps to escape local optima
    # We use a multi-start approach: try many different swap sequences
    
    local_search_deadline = deadline - 2
    
    # First, try pure greedy addition until no more edges can be added
    changed = True
    while changed and time.time() < local_search_deadline:
        changed = False
        for i in range(len(non_existing_edges_list) - 1, -1, -1):
            u, v = non_existing_edges_list[i]
            if can_add_edge(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_count += 1
                non_existing_edges_list.pop(i)
                existing_edges_list.append((u, v))
                changed = True
                if current_count > best_count:
                    best_count = current_count
                    best_edges_list = []
                    for u2 in range(n):
                        for v2 in range(u2 + 1, n):
                            if (current_adj[u2] >> v2) & 1:
                                best_edges_list.append([u2, v2])
                    if best_count >= 105:
                        return best_edges_list
    
    # Now use swap-based local search to try to escape local optima
    # Strategy: remove one edge, then greedily add as many as possible,
    # keep the result if it's better
    random.seed(12345)
    
    swap_attempts = 0
    max_swap_attempts = 200  # limit to avoid infinite loop
    
    while time.time() < local_search_deadline and swap_attempts < max_swap_attempts:
        swap_attempts += 1
        
        # Pick a random existing edge to remove
        if not existing_edges_list:
            break
        
        idx_rem = random.randint(0, len(existing_edges_list) - 1)
        u1, v1 = existing_edges_list[idx_rem]
        
        # Save state
        saved_adj = current_adj[:]
        saved_count = current_count
        saved_existing = existing_edges_list[:]
        saved_non_existing = non_existing_edges_list[:]
        
        # Remove edge
        current_adj[u1] &= ~(1 << v1)
        current_adj[v1] &= ~(1 << u1)
        current_count -= 1
        
        # Move to non-existing
        existing_edges_list.pop(idx_rem)
        non_existing_edges_list.append((u1, v1))
        
        # Now try to greedily add edges
        added = 0
        for i in range(len(non_existing_edges_list) - 1, -1, -1):
            u, v = non_existing_edges_list[i]
            if can_add_edge(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_count += 1
                non_existing_edges_list.pop(i)
                existing_edges_list.append((u, v))
                added += 1
        
        # If we gained an edge, keep it
        if current_count > saved_count:
            best_count = max(best_count, current_count)
            if best_count > saved_count:
                best_edges_list = []
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if (current_adj[u2] >> v2) & 1:
                            best_edges_list.append([u2, v2])
                if best_count >= 105:
                    return best_edges_list
        else:
            # Restore state
            current_adj = saved_adj
            current_count = saved_count
            existing_edges_list = saved_existing
            non_existing_edges_list = saved_non_existing
    
    # Final: try one more pass of greedy addition
    for i in range(len(non_existing_edges_list) - 1, -1, -1):
        u, v = non_existing_edges_list[i]
        if can_add_edge(current_adj, u, v):
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_count += 1
            non_existing_edges_list.pop(i)
            existing_edges_list.append((u, v))
    
    if current_count > best_count:
        best_count = current_count
        best_edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if (current_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])

    if not best_edges_list:
        for u in range(n):
            for v in range(u + 1, n):
                if (base_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])

    return best_edges_list
