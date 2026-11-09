import random
import time
import itertools

def is_k4_free(adj):
    """Check if graph (bitmask adjacency) is K4-free using bitmasks."""
    n = len(adj)
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            common = adj[a] & adj[b]
            m = common
            while m:
                lsb = m & -m
                c = lsb.bit_length() - 1
                rest = common & ~((1 << (c + 1)) - 1)
                if rest & adj[c]:
                    return False
                m ^= lsb
    return True

def has_triangle_in_mask(adj, mask):
    """Check if the induced subgraph on mask has a triangle."""
    m = mask
    while m:
        lsb = m & -m
        u = lsb.bit_length() - 1
        rest = mask & ~((1 << (u + 1)) - 1)
        if rest & adj[u]:
            # Check if any edge in rest ∩ adj[u]
            common = rest & adj[u]
            if common:
                # For each pair in common, check edge
                cm = common
                while cm:
                    lsb2 = cm & -cm
                    v = lsb2.bit_length() - 1
                    rest2 = common & ~((1 << (v + 1)) - 1)
                    if rest2 & adj[v]:
                        return True
                    cm ^= lsb2
        m ^= lsb
    return False

def largest_tf_size(adj, upper_bound=None):
    """
    Find the size of the largest triangle-free induced subgraph.
    Check from upper_bound down to 1.
    """
    n = len(adj)
    if upper_bound is None:
        upper_bound = n
    
    for k in range(upper_bound, 0, -1):
        # Check if any subset of size k is triangle-free
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if not has_triangle_in_mask(adj, mask):
                return k
    return 0

def repair_k4_free(adj):
    """Remove edges to make graph K4-free."""
    adj = adj[:]
    for _ in range(200):
        if is_k4_free(adj):
            break
        found = False
        n = len(adj)
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)):
                    continue
                common = adj[a] & adj[b]
                m = common
                while m:
                    lsb = m & -m
                    c = lsb.bit_length() - 1
                    rest = common & ~((1 << (c + 1)) - 1)
                    if rest & adj[c]:
                        d_mask = rest & adj[c]
                        d = (d_mask & -d_mask).bit_length() - 1
                        adj[c] &= ~(1 << d)
                        adj[d] &= ~(1 << c)
                        found = True
                        break
                    m ^= lsb
                if found:
                    break
            if found:
                break
        if not found:
            break
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    
    n = 18
    best_l = n + 1
    best_adj = None
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    # Candidate 1: 4-partite K(5,5,4,4)
    adj_4part = [0] * n
    parts = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13], [14,15,16,17]]
    for i in range(4):
        for j in range(i + 1, 4):
            for u in parts[i]:
                for v in parts[j]:
                    adj_4part[u] |= (1 << v)
                    adj_4part[v] |= (1 << u)
    
    # Candidate 2: 5-partite K(4,4,4,3,3)
    adj_5part = [0] * n
    parts5 = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14], [15,16,17]]
    for i in range(5):
        for j in range(i + 1, 5):
            for u in parts5[i]:
                for v in parts5[j]:
                    adj_5part[u] |= (1 << v)
                    adj_5part[v] |= (1 << u)
    
    # Candidate 3: 3-partite K(6,6,6) - L=12
    adj_3part = [0] * n
    parts3 = [[0,1,2,3,4,5], [6,7,8,9,10,11], [12,13,14,15,16,17]]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts3[i]:
                for v in parts3[j]:
                    adj_3part[u] |= (1 << v)
                    adj_3part[v] |= (1 << u)
    
    candidates = [adj_4part, adj_5part, adj_3part]
    
    # Add random candidates
    random.seed(42)
    for seed in range(8):
        random.seed(seed * 7 + 3)
        adj_rand = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_rand[i] |= (1 << j)
                    adj_rand[j] |= (1 << i)
        candidates.append(repair_k4_free(adj_rand))
    
    # Evaluate and local search
    for adj in candidates:
        if time.time() > deadline:
            break
        
        l = largest_tf_size(adj, n)
        if l < best_l:
            best_l = l
            best_adj = adj[:]
        
        if time.time() > deadline:
            continue
        
        # Local search
        current_adj = adj[:]
        current_l = l
        T = 1.0
        
        for step in range(3000):
            if time.time() > deadline:
                break
            
            i, j = random.sample(range(n), 2)
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
            
            if not is_k4_free(current_adj):
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
                continue
            
            new_l = largest_tf_size(current_adj, current_l + 1)
            
            if new_l < current_l:
                current_l = new_l
                if new_l < best_l:
                    best_l = new_l
                    best_adj = current_adj[:]
            elif new_l == current_l:
                if random.random() < 0.3:
                    pass
                else:
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
            else:
                delta = new_l - current_l
                if random.random() < 0.005 ** (delta / T):
                    pass
                else:
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
            
            T *= 0.995
    
    if best_adj is None:
        return adj_to_edges(candidates[0])
    
    return adj_to_edges(best_adj)
