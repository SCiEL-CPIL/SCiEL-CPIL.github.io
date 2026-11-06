import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit

    n = 40

    # adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists

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
        # i.e., u-a-b-c-d-v
        # a in N(u), d in N(v), a != d
        # path a-b-c-d exists in G \ {u, v}
        
        # Iterate through all pairs (a, d)
        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a
            
            d = nv_mask & ~lb_a  # d != a
            while d:
                ld_d = d & (-d)
                di = ld_d.bit_length() - 1
                d ^= ld_d
                
                # We need a path a-b-c-d
                # b must be in N(a) \ {u, v, d, a}
                # c must be in N(d) \ {u, v, a, d}
                # and b ~ c
                
                # Mask for b candidates
                b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
                
                # Mask for c candidates
                c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
                
                # Check if any neighbor of any b in b_candidates is in c_candidates
                # Let U = union of neighbors of all b in b_candidates
                # If U & c_candidates is non-empty, then a C6 exists.
                
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
                
            # end d loop
        # end a loop
        
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    # Strategy:
    # 1. Initialize with 8 disjoint K5s (80 edges)
    # 2. Use a hybrid approach: 
    #    a. Greedy random restarts to get a good baseline
    #    b. Simulated Annealing (SA) with Add/Swap moves to refine and potentially exceed greedy limits

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Initialize base structure: 8 disjoint K5s
    # Vertices 0-4, 5-9, ..., 35-39
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
    existing_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (base_adj[u] >> v) & 1:
                existing_edges.add((u, v))
    
    # Pairs to consider for addition
    candidate_pairs = [(u, v) for u, v in all_pairs if (u, v) not in existing_edges]
    
    best_edges_list = []
    best_count = 0
    
    # Phase 1: Greedy Random Restarts
    # We run a limited number of greedy restarts to find a good initial state
    # or to establish a strong baseline.
    
    # We will use a fixed number of restarts first, then switch to SA if time permits.
    # Or we can interleave them.
    # Let's do a few greedy restarts first to get a high-quality starting point for SA.
    
    num_greedy_restarts = 20
    
    for restart in range(num_greedy_restarts):
        if time.time() > deadline - 60:  # Reserve at least 60s for SA
            break
            
        current_adj = base_adj[:]
        
        # Shuffle candidate pairs
        random.seed(100 + restart)
        random.shuffle(candidate_pairs)
        
        for u, v in candidate_pairs:
            if time.time() > deadline - 60:
                break
            if not ((current_adj[u] >> v) & 1):
                if can_add_edge(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        
        current_count = count_edges(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            # Build edge list
            best_edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (current_adj[u] >> v) & 1:
                        best_edges_list.append([u, v])
            
            # Early exit if we hit the known upper bound or close to it
            if best_count >= 105:
                break

    # Phase 2: Simulated Annealing Refinement
    # Start from the best greedy solution found
    if best_edges_list:
        # Reconstruct adjacency from best_edges_list
        current_adj = [0] * n
        for u, v in best_edges_list:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
        current_count = best_count
    else:
        current_adj = base_adj[:]
        current_count = 80

    # SA Parameters
    T = 10.0
    cooling_rate = 0.99
    min_T = 0.1
    
    # Maintain lists of existing and non-existing edges for SA moves
    def get_edge_lists(adj):
        existing = []
        non_existing = []
        for u, v in all_pairs:
            if (adj[u] >> v) & 1:
                existing.append((u, v))
            else:
                non_existing.append((u, v))
        return existing, non_existing

    existing_edges_list, non_existing_edges_list = get_edge_lists(current_adj)
    
    # SA Loop
    while time.time() < deadline - 2:
        # Pick a move type
        # Type A: Add edge (prob 0.6)
        # Type B: Swap edge (prob 0.4)
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add a random non-edge
            if not non_existing_edges_list:
                continue
                
            idx = random.randint(0, len(non_existing_edges_list) - 1)
            u, v = non_existing_edges_list[idx]
            
            if can_add_edge(current_adj, u, v):
                # Accept move
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_count += 1
                
                # Update lists
                non_existing_edges_list.pop(idx)
                existing_edges_list.append((u, v))
                
                if current_count > best_count:
                    best_count = current_count
                    best_edges_list = []
                    for u2 in range(n):
                        for v2 in range(u2 + 1, n):
                            if (current_adj[u2] >> v2) & 1:
                                best_edges_list.append([u2, v2])
                    
                    if best_count >= 105:
                        break
        else:
            # Try a swap: remove a random existing edge, add a random non-edge
            if not existing_edges_list or not non_existing_edges_list:
                continue
                
            idx_rem = random.randint(0, len(existing_edges_list) - 1)
            u1, v1 = existing_edges_list[idx_rem]
            
            idx_add = random.randint(0, len(non_existing_edges_list) - 1)
            u2, v2 = non_existing_edges_list[idx_add]
            
            # Temporarily remove (u1, v1)
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            # Check if (u2, v2) can be added
            if can_add_edge(current_adj, u2, v2):
                # Accept swap
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                # Update lists
                # Remove (u1, v1) from existing, add to non-existing
                existing_edges_list.pop(idx_rem)
                non_existing_edges_list.append((u1, v1))
                
                # Remove (u2, v2) from non-existing, add to existing
                non_existing_edges_list.pop(idx_add)
                existing_edges_list.append((u2, v2))
                
                # Edge count unchanged, so no best update needed here
            else:
                # Reject swap, restore (u1, v1)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Final check
    if not best_edges_list:
        for u in range(n):
            for v in range(u + 1, n):
                if (base_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])

    return best_edges_list
