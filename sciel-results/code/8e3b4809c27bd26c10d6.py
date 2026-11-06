import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 18
    
    # Precompute subsets for triangle-free check
    # We only need to check subsets of size > 8 or so.
    # Let's precompute all subsets of size 9 to 18.
    # Actually, we can just generate them on the fly or precompute once.
    # Precomputing all subsets of size 9..18 is memory heavy?
    # Sum(18Ck for k=9..18) = 2^18 - Sum(18Ck for k=0..8)
    # 2^18 = 262144.
    # It's fine to store them as tuples.
    
    subsets_by_size = {}
    for k in range(1, n+1):
        subsets_by_size[k] = list(itertools.combinations(range(n), k))
        
    # We will check from high k down to 1.
    # But we can stop early.
    
    def is_triangle_free_subset(subset, adj):
        mask_subset = 0
        for v in subset:
            mask_subset |= (1 << v)
            
        for i in subset:
            neigh_i = adj[i] & mask_subset
            m = neigh_i
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    # Check for common neighbor in S
                    if (adj[i] & adj[j] & mask_subset):
                        return False
                m ^= lsb
        return True

    def get_tf(adj):
        # Find max k such that there exists a triangle-free subset of size k
        for k in range(n, 0, -1):
            for subset in subsets_by_size[k]:
                if is_triangle_free_subset(subset, adj):
                    return k
        return 0

    def is_k4_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    # Check if any k in common is connected to all i,j
                    # Actually, if i,j,k is a triangle, and k has a neighbor l in common(i,j,k), then K4.
                    # Simpler: for each k in common, if (adj[i] & adj[j] & adj[k]) != 0, then K4.
                    m = common
                    while m:
                        lsb = m & -m
                        k = lsb.bit_length() - 1
                        if (adj[i] & adj[j] & adj[k]):
                            return False
                        m ^= lsb
        return True

    def generate_random_k4_free():
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            common = adj[u] & adj[v]
            m = common
            has_k4 = False
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                if (adj[u] & adj[v] & adj[k]):
                    has_k4 = True
                    break
                m ^= lsb
            if not has_k4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_adj = generate_random_k4_free()
    best_score = get_tf(best_adj)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        adj_copy = best_adj[:]
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        if adj_copy[u] & (1 << v):
            adj_copy[u] ^= (1 << v)
            adj_copy[v] ^= (1 << u)
            new_adj = adj_copy
        else:
            common = adj_copy[u] & adj_copy[v]
            m = common
            has_k4 = False
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                if (adj_copy[u] & adj_copy[v] & adj_copy[k]):
                    has_k4 = True
                    break
                m ^= lsb
            if has_k4:
                continue
            adj_copy[u] |= (1 << v)
            adj_copy[v] |= (1 << u)
            new_adj = adj_copy
            
        new_score = get_tf(new_adj)
        
        delta = new_score - best_score
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            best_adj = new_adj
            best_score = new_score
        
        T *= alpha
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges
