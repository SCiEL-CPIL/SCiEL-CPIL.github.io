import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit

    n = 40
    
    # adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists
    adj = [0] * n
    
    def can_add_edge(u, v):
        """
        Check if adding edge (u,v) to the current graph creates a C6.
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
                
                # If we finished checking all b for this (a,d) pair and found no connection, continue
                # Note: We can optimize by checking if b_candidates is empty immediately
                # But the loop handles it.
                
            # end d loop
        # end a loop
        
        return True

    def count_edges():
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    best_edges_list = []
    best_count = 0
    
    # Strategy:
    # 1. Initialize with 8 disjoint K5s (80 edges)
    # 2. Greedily add remaining edges in random order
    # 3. Repeat with different seeds to find better orderings
    
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
    
    # We will run several restarts
    # Each restart starts from the same base_adj but shuffles candidate_pairs differently
    # Actually, we can just shuffle the candidate pairs and greedily add them to a copy of base_adj
    
    max_restarts = 100 # We will be limited by time
    restart = 0
    
    while time.time() < deadline:
        restart += 1
        current_adj = base_adj[:]
        
        # Shuffle candidate pairs
        random.seed(42 + restart)
        random.shuffle(candidate_pairs)
        
        for u, v in candidate_pairs:
            if time.time() > deadline - 2:
                break
            if not ((current_adj[u] >> v) & 1):
                if can_add_edge_on(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        
        current_count = count_edges_on(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            # Build edge list
            best_edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (current_adj[u] >> v) & 1:
                        best_edges_list.append([u, v])
            
            # Early exit if we hit the known upper bound or close to it
            # ex(40, C6) is known to be at least 105. If we hit 105, we can stop.
            if best_count >= 105:
                break

    # If no edges found (should not happen with base), return base edges
    if not best_edges_list:
        for u in range(n):
            for v in range(u + 1, n):
                if (base_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])

    return best_edges_list

# Helper functions to operate on a specific adjacency list without global state interference
# This is cleaner for the restart loop

def can_add_edge_on(adj, u, v):
    """
    Check if adding edge (u,v) to the graph represented by adj creates a C6.
    """
    n = 40
    nu_mask = adj[u] & ~(1 << v)
    nv_mask = adj[v] & ~(1 << u)
    
    a = nu_mask
    while a:
        lb_a = a & (-a)
        ai = lb_a.bit_length() - 1
        a ^= lb_a
        
        d = nv_mask & ~lb_a
        while d:
            ld_d = d & (-d)
            di = ld_d.bit_length() - 1
            d ^= ld_d
            
            b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
            c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
            
            union_neighbors = 0
            bm = b_candidates
            while bm:
                lb_b = bm & (-bm)
                bi = lb_b.bit_length() - 1
                bm ^= lb_b
                union_neighbors |= adj[bi]
                
                if union_neighbors & c_candidates:
                    return False
                    
    return True

def count_edges_on(adj):
    count = 0
    for i in range(40):
        count += adj[i].bit_count()
    return count // 2

# Re-define construct to use the helper functions properly to avoid name conflicts or scope issues
def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 40
    
    # Base structure: 8 disjoint K5s
    base_adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
                
    existing_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (base_adj[u] >> v) & 1:
                existing_edges.add((u, v))
                
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    candidate_pairs = [(u, v) for u, v in all_pairs if (u, v) not in existing_edges]
    
    best_edges_list = []
    best_count = 0
    
    restart = 0
    while time.time() < deadline:
        restart += 1
        current_adj = base_adj[:]
        
        random.seed(42 + restart)
        random.shuffle(candidate_pairs)
        
        for u, v in candidate_pairs:
            if time.time() > deadline - 2:
                break
            if not ((current_adj[u] >> v) & 1):
                if can_add_edge_on(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
        current_count = count_edges_on(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (current_adj[u] >> v) & 1:
                        best_edges_list.append([u, v])
                        
            if best_count >= 105:
                break

    if not best_edges_list:
        for u in range(n):
            for v in range(u + 1, n):
                if (base_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])

    return best_edges_list
