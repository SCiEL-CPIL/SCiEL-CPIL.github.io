import time
import random

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, task limit 120s.
    # We aim to finish well before 120s to ensure return.
    deadline = start_time + 110.0

    n = 40
    full_mask = (1 << n) - 1

    def now_dead():
        return time.time() >= deadline

    # Helper to check if adding edge (u, v) creates a C6
    # Returns True if safe (no C6 created), False if creates C6.
    # adj is a list of ints, adj[i] has bit j set if edge (i,j) exists.
    def can_add_edge(adj, u, v):
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
    
    # Precompute all pairs not in base_adj
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    base_edges_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (base_adj[u] >> v) & 1:
                base_edges_set.add((u, v))
    
    candidate_pairs = [(u, v) for u, v in all_pairs if (u, v) not in base_edges_set]
    
    best_edges_list = []
    best_count = count_edges(base_adj) # Start with base count (80)
    
    # Store best adjacency matrix for reconstruction
    best_adj = base_adj[:]

    restart = 0
    # We will run restarts until deadline
    while not now_dead():
        restart += 1
        current_adj = base_adj[:]
        
        # Shuffle candidate pairs
        random.seed(42 + restart)
        random.shuffle(candidate_pairs)
        
        for u, v in candidate_pairs:
            if time.time() > deadline - 2.0:
                break
            
            # Check if edge already exists (should not, as we only add, but safe check)
            if not ((current_adj[u] >> v) & 1):
                if can_add_edge(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        
        current_count = count_edges(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]
            
            # Early exit if we hit a very high score, though 105 is the target
            if best_count >= 110:
                break

    # Construct final edge list from best_adj
    if not best_edges_list: # Always populate
        best_edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])

    return best_edges_list
