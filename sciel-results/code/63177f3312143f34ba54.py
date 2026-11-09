import time
import random

def construct():
    start_time = time.time()
    timeout = 280  # Safety margin
    
    def paley_graph(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges

    def build_bitsets(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_cliques_4(adj, n):
        # Find all 4-cliques
        cliques = []
        # Precompute neighbors
        neighbors = [adj[i] for i in range(n)]
        for i in range(n):
            ni = neighbors[i]
            # Iterate j > i
            j = (ni & ~((1 << (i+1)) - 1))
            while j:
                lb_j = j & -j
                j_idx = lb_j.bit_length() - 1
                j = j - lb_j
                # ni & nj
                common_ij = ni & neighbors[j_idx]
                # Iterate k > j
                k = common_ij & ~((1 << (j_idx+1)) - 1)
                while k:
                    lb_k = k & -k
                    k_idx = lb_k.bit_length() - 1
                    k = k - lb_k
                    # common_ij & nk
                    common_ijk = common_ij & neighbors[k_idx]
                    # Iterate l > k
                    l = common_ijk & ~((1 << (k_idx+1)) - 1)
                    if l:
                        # Found a 4-clique {i, j_idx, k_idx, l_idx}
                        lb_l = l & -l
                        l_idx = lb_l.bit_length() - 1
                        cliques.append((i, j_idx, k_idx, l_idx))
                        # Optimization: We can break early if we just need to check existence, 
                        # but for adding vertex, we need to know if a specific set contains a clique.
                        # Storing all is memory heavy for n=40? 
                        # C(40,4) ~ 91k. Storing 91k tuples is fine.
        return cliques

    def check_k5(adj, n, cliques_4):
        # A K5 exists if any 4-clique has a common neighbor
        for c in cliques_4:
            i, j, k, l = c
            common = adj[i] & adj[j] & adj[k] & adj[l]
            # Exclude the clique members themselves
            mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
            if common & ~mask:
                return True
        return False

    def check_ind5(adj, n, cliques_4_comp):
        # Independent set of size 5 in G is K5 in G_comp
        # We can just call check_k5 on the complement
        return check_k5(adj_comp, n, cliques_4_comp)

    def get_cliques_4_from_adj(adj, n):
        # Reuse logic
        cliques = []
        neighbors = [adj[i] for i in range(n)]
        for i in range(n):
            ni = neighbors[i]
            j = ni & ~((1 << (i+1)) - 1)
            while j:
                lb_j = j & -j
                j_idx = lb_j.bit_length() - 1
                j = j - lb_j
                common_ij = ni & neighbors[j_idx]
                k = common_ij & ~((1 << (j_idx+1)) - 1)
                while k:
                    lb_k = k & -k
                    k_idx = lb_k.bit_length() - 1
                    k = k - lb_k
                    common_ijk = common_ij & neighbors[k_idx]
                    l = common_ijk & ~((1 << (k_idx+1)) - 1)
                    if l:
                        lb_l = l & -l
                        l_idx = lb_l.bit_length() - 1
                        cliques.append((i, j_idx, k_idx, l_idx))
        return cliques

    def is_valid_graph(adj, n):
        cliques_4 = get_cliques_4_from_adj(adj, n)
        if check_k5(adj, n, cliques_4):
            return False
        
        # Complement
        full_mask = (1 << n) - 1
        adj_comp = [full_mask ^ adj[i] ^ (1 << i) for i in range(n)]
        cliques_4_comp = get_cliques_4_from_adj(adj_comp, n)
        if check_k5(adj_comp, n, cliques_4_comp):
            return False
        return True

    def try_add_vertex(adj, n):
        # Returns new_adj if successful, else None
        # We need to find N_new such that:
        # 1. N_new contains no K4 from G
        # 2. Non-N_new contains no K4 from G_comp
        
        # Precompute K4s in G and G_comp for current n
        cliques_4 = get_cliques_4_from_adj(adj, n)
        full_mask = (1 << n) - 1
        adj_comp = [full_mask ^ adj[i] ^ (1 << i) for i in range(n)]
        cliques_4_comp = get_cliques_4_from_adj(adj_comp, n)
        
        # Convert cliques to masks for fast checking
        masks_k4 = [((1<<a)|(1<<b)|(1<<c)|(1<<d)) for a,b,c,d in cliques_4]
        masks_k4_comp = [((1<<a)|(1<<b)|(1<<c)|(1<<d)) for a,b,c,d in cliques_4_comp]
        
        # Randomized greedy construction of N_new
        # Start with random subset
        for attempt in range(100):
            # Initialize N_new with random probability 0.5
            N_new = 0
            for i in range(n):
                if random.random() < 0.5:
                    N_new |= (1 << i)
            
            # Check validity of N_new
            valid = True
            # Check K4 in N_new
            for m in masks_k4:
                if (N_new & m) == m:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check K4 in Non-N_new
            Non_N_new = full_mask ^ N_new
            for m in masks_k4_comp:
                if (Non_N_new & m) == m:
                    valid = False
                    break
            if not valid:
                continue
            
            # If valid, construct new adj
            new_n = n + 1
            new_adj = adj[:] + [0]
            # Set edges for new vertex
            new_adj[new_n - 1] = N_new
            for i in range(n):
                if (N_new >> i) & 1:
                    new_adj[i] |= (1 << (new_n - 1))
            
            return new_adj, new_n
        return None, None

    # Initialization
    q, edges = paley_graph(37)
    n = q
    adj = build_bitsets(n, edges)
    
    # Verify initial
    if not is_valid_graph(adj, n):
        # Fallback to smaller if 37 is invalid (should not happen)
        return 37, edges

    best_n = n
    best_adj = adj[:]
    
    while time.time() - start_time < timeout:
        # Try to add vertex
        new_adj, new_n = try_add_vertex(adj, n)
        if new_adj is not None:
            adj = new_adj
            n = new_n
            best_n = n
            best_adj = adj[:]
            continue
        
        # Flip random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        old_u = adj[u]
        old_v = adj[v]
        
        # Flip
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        if is_valid_graph(adj, n):
            # Keep flip
            best_n = n
            best_adj = adj[:]
        else:
            # Undo
            adj[u] = old_u
            adj[v] = old_v

    # Return edges from best_adj
    best_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                best_edges.append((i, j))
    
    return best_n, best_edges
