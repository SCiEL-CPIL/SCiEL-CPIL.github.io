import random
import time

def is_k4_free(adj, n):
    """Fast K4-free check using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common_ab = adj[a] & adj[b]
            common_ab &= ~((1 << (b + 1)) - 1)
            while common_ab:
                lsb = common_ab & -common_ab
                c = lsb.bit_length() - 1
                common_abc = adj[a] & adj[b] & adj[c]
                common_abc &= ~((1 << (c + 1)) - 1)
                if common_abc:
                    return False
                common_ab ^= lsb
    return True

def has_triangle_mask(mask, adj, n):
    """Check if induced subgraph on mask has a triangle."""
    m = mask
    while m:
        lsb_u = m & -m
        u = lsb_u.bit_length() - 1
        m ^= lsb_u
        nbrs_u = adj[u] & mask & ~((1 << (u + 1)) - 1)
        while nbrs_u:
            lsb_v = nbrs_u & -nbrs_u
            v = lsb_v.bit_length() - 1
            nbrs_u ^= lsb_v
            if adj[u] & adj[v] & mask:
                return True
    return False

def compute_largest_tf(adj, n, deadline):
    """
    Find the largest triangle-free induced subgraph using branch and bound.
    """
    def exists_tf_subset(k):
        """Check if there exists a triangle-free induced subgraph of size k."""
        vertices = list(range(n))
        
        def backtrack(start_idx, chosen_count, chosen_mask):
            if time.time() > deadline:
                return False
            if chosen_count == k:
                return not has_triangle_mask(chosen_mask, adj, n)
            if start_idx >= n:
                return False
            if n - start_idx < k - chosen_count:
                return False
            
            v = vertices[start_idx]
            
            # Try including v
            new_mask = chosen_mask | (1 << v)
            # Check if adding v creates a triangle with already chosen vertices
            nbrs_v_in_chosen = adj[v] & chosen_mask
            creates_tri = False
            m = nbrs_v_in_chosen
            while m:
                lsb_u = m & -m
                u = lsb_u.bit_length() - 1
                m ^= lsb_u
                common = adj[u] & adj[v] & chosen_mask
                if common & ~(1 << u) & ~(1 << v):
                    creates_tri = True
                    break
            if not creates_tri:
                if backtrack(start_idx + 1, chosen_count + 1, new_mask):
                    return True
            
            # Try excluding v
            if backtrack(start_idx + 1, chosen_count, chosen_mask):
                return True
            
            return False
        
        return backtrack(0, 0, 0)
    
    for k in range(n, 0, -1):
        if exists_tf_subset(k):
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 18
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    best_l = n + 1
    best_adj = None
    
    def update_best(adj, l):
        nonlocal best_l, best_adj
        if l < best_l:
            best_l = l
            best_adj = adj[:]
    
    random.seed(42)
    
    # Generate candidates from 4-partite structures
    candidates = []
    
    # Try different 4-partite splits
    part_schemes = [
        [4, 4, 5, 5],
        [4, 5, 4, 5],
        [5, 4, 4, 5],
        [4, 4, 4, 6],
        [4, 4, 6, 4],
        [4, 6, 4, 4],
        [6, 4, 4, 4],
        [5, 5, 4, 4],
        [4, 5, 5, 4],
        [5, 4, 5, 4],
    ]
    
    for scheme in part_schemes:
        parts = []
        start = 0
        for size in scheme:
            parts.append(list(range(start, start + size)))
            start += size
        
        for trial in range(20):
            adj = [0] * n
            # Start with complete 4-partite
            for i in range(4):
                for j in range(i + 1, 4):
                    for u in parts[i]:
                        for v in parts[j]:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            
            # Repair to K4-free by removing edges
            removed = set()
            for _ in range(2000):
                if is_k4_free(adj, n):
                    break
                found_k4 = False
                # Find a K4 and remove a random edge from it
                for a in range(n):
                    for b in range(a + 1, n):
                        if not (adj[a] & (1 << b)):
                            continue
                        for c in range(b + 1, n):
                            if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                                continue
                            common = adj[a] & adj[b] & adj[c]
                            common &= ~((1 << (c + 1)) - 1)
                            if common:
                                lsb = common & -common
                                d = lsb.bit_length() - 1
                                # Remove a random edge from the K4
                                edge_choices = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                e = random.choice(edge_choices)
                                key = (min(e), max(e))
                                if key not in removed:
                                    adj[e[0]] &= ~(1 << e[1])
                                    adj[e[1]] &= ~(1 << e[0])
                                    removed.add(key)
                                    found_k4 = True
                                    break
                        if found_k4:
                            break
                    if found_k4:
                        break
                if not found_k4:
                    break
            
            if is_k4_free(adj, n):
                candidates.append(adj)
    
    # Also add Turan T(18,3) as baseline
    adj_turan = [0] * n
    parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts3[i]:
                for v in parts3[j]:
                    adj_turan[u] |= (1 << v)
                    adj_turan[v] |= (1 << u)
    candidates.append(adj_turan)
    
    # Evaluate candidates
    for adj in candidates:
        if time.time() > deadline:
            break
        l = compute_largest_tf(adj, n, deadline)
        update_best(adj, l)
    
    # Local search with simulated annealing
    if best_adj is not None and time.time() < deadline - 10:
        current_adj = best_adj[:]
        current_l = best_l
        T = 1.0
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            i, j = random.sample(range(n), 2)
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
            
            if not is_k4_free(current_adj, n):
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
                continue
            
            new_l = compute_largest_tf(current_adj, n, deadline)
            
            if new_l < current_l:
                current_l = new_l
                update_best(current_adj, new_l)
            elif new_l == current_l:
                pass
            else:
                delta = new_l - current_l
                if random.random() < pow(2.71828, -delta / T):
                    pass
                else:
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
            
            T *= 0.998
    
    if best_adj is None:
        return adj_to_edges(candidates[-1])
    
    return adj_to_edges(best_adj)
