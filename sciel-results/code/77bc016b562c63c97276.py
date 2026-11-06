import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9, maximizing n.
    Uses proven 8xK4 (n=32) as base, then attempts local search for n=33+.
    """
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    # Proven n=32 solution: 8 disjoint K4's
    def make_k4_solution():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    fallback = make_k4_solution()
    
    # Try to find n=33, 34, or 35 solutions using structured search
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 10:
            break
        result = _search_graph(target_n, deadline)
        if result is not None:
            return result
    
    return fallback

def _has_c5_fast(n, adj):
    """Check if graph has a C5 using adjacency matrix. Returns True if C5 exists."""
    for v0 in range(n):
        for v1 in range(n):
            if v1 <= v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 == v0 or v2 == v1 or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2 or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if adj[v3][v4] and adj[v4][v0]:
                            return True
    return False

def _exact_alpha(n, adj, timeout=5.0):
    """Exact maximum independent set size using branch and bound with bitsets."""
    best = [0]
    start_time = time.time()
    
    # Build bitset adjacency
    adj_bits = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        adj_bits[i] = mask
        
    def backtrack(candidates, current_size):
        if time.time() - start_time > timeout:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound: current_size + popcount(candidates)
        if current_size + bin(candidates).count('1') <= best[0]:
            return
        
        # Pick vertex with max degree in candidates
        max_deg = -1
        v = -1
        temp_c = candidates
        while temp_c:
            lsb = temp_c & -temp_c
            u = (lsb.bit_length() - 1)
            # Degree of u in candidates
            d = bin(adj_bits[u] & candidates).count('1')
            if d > max_deg:
                max_deg = d
                v = u
            temp_c ^= lsb
            
        # Include v
        new_candidates = candidates & ~adj_bits[v] & ~(1 << v)
        backtrack(new_candidates, current_size + 1)
        
        # Exclude v
        backtrack(candidates & ~(1 << v), current_size)
    
    backtrack((1 << n) - 1, 0)
    return best[0]

def _find_one_c5(n, adj):
    """Find one C5, return tuple or None."""
    for v0 in range(n):
        for v1 in range(n):
            if v1 <= v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 == v0 or v2 == v1 or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2 or not adj[v2][v3]:
                        continue
                        for v4 in range(n):
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                                continue
                            if adj[v3][v4] and adj[v4][v0]:
                                return (v0, v1, v2, v3, v4)
    return None

def _search_graph(target_n, deadline):
    """Structured search for valid graph with target_n vertices."""
    random.seed(12345)
    
    for attempt in range(1000):
        if time.time() > deadline - 5:
            break
        
        adj = [[False] * target_n for _ in range(target_n)]
        
        p = 0.4
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Break C5's
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            c5 = _find_one_c5(target_n, adj)
            if c5 is None:
                break
            v0, v1, v2, v3, v4 = c5
            edge_idx = random.randint(0, 4)
            edges_c5 = [(v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v0)]
            u, v = edges_c5[edge_idx]
            adj[u][v] = False
            adj[v][u] = False
        
        if not _has_c5_fast(target_n, adj):
            alpha = _exact_alpha(target_n, adj, timeout=2.0)
            if alpha < 9:
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': target_n, 'edges': edges}
    
    return None
