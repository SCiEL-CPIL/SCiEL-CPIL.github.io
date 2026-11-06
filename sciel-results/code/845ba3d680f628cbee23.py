import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    n = 40
    best_edges = []
    best_count = 0
    
    # Precompute all pairs for speed
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def check_c6_with_edge(mask, u, v):
        """
        Check if adding edge (u,v) to graph 'mask' creates a C6.
        A new C6 must use the new edge (u,v).
        So we need a path of length 5 from u to v in the current graph:
        u - a - b - c - d - v, with a,b,c,d distinct and not u,v.
        """
        nu = mask[u]
        nv = mask[v]
        
        # a must be a neighbor of u (and not v)
        # d must be a neighbor of v (and not u)
        # a != d
        # Need path a-b-c-d in graph, with b,c distinct from {u,v,a,d}
        
        a_candidates = nu & ~(1 << v)
        d_candidates = nv & ~(1 << u)
        
        while a_candidates:
            a_bit = a_candidates & (-a_candidates)
            a = a_bit.bit_length() - 1
            a_candidates ^= a_bit
            
            # d in d_candidates, d != a
            d_cand = d_candidates & ~(1 << a)
            while d_cand:
                d_bit = d_cand & (-d_cand)
                d = d_bit.bit_length() - 1
                d_cand ^= d_bit
                
                # b in N(a)\{u,v,d}
                na = mask[a]
                b_cand = na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                while b_cand:
                    b_bit = b_cand & (-b_cand)
                    b = b_bit.bit_length() - 1
                    b_cand ^= b_bit
                    
                    # c in N(d)\{u,v,a,b} and c~b
                    # c in (mask[d] & mask[b]) & ~((1<<u)|(1<<v)|(1<<a)|(1<<b))
                    c_cand = (mask[d] & mask[b]) & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                    if c_cand:
                        return True
        return False

    def is_edge_in_c6(mask, u, v):
        """Check if existing edge (u,v) is part of a C6 in mask."""
        orig_u = mask[u]
        orig_v = mask[v]
        mask[u] &= ~(1 << v)
        mask[v] &= ~(1 << u)
        res = check_c6_with_edge(mask, u, v)
        mask[u] = orig_u
        mask[v] = orig_v
        return res

    def random_graph(p):
        mask = [0] * n
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    mask[i] |= (1 << j)
                    mask[j] |= (1 << i)
                    edges.append((i, j))
        return mask, edges

    def prune_c6(mask, edges):
        """Remove edges that are part of C6s."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            random.shuffle(edges)
            for i in range(len(edges)):
                u, v = edges[i]
                if is_edge_in_c6(mask, u, v):
                    mask[u] &= ~(1 << v)
                    mask[v] &= ~(1 << u)
                    edges.pop(i)
                    changed = True
                    break # Restart scan
        return mask, edges

    def greedy_add(mask, edges):
        """Add all possible non-edges that don't create C6."""
        added = True
        while added and time.time() < deadline:
            added = False
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (mask[i] >> j) & 1:
                        non_edges.append((i, j))
            
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if not check_c6_with_edge(mask, u, v):
                    mask[u] |= (1 << v)
                    mask[v] |= (1 << u)
                    edges.append((u, v))
                    added = True
        return mask, edges

    # Main Loop: Multiple restarts with different densities
    densities = [0.10, 0.12, 0.15, 0.18, 0.20]
    
    while time.time() < deadline:
        p = random.choice(densities)
        mask, edges = random_graph(p)
        
        mask, edges = prune_c6(mask, edges)
        mask, edges = greedy_add(mask, edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [[u, v] for u, v in edges]
            
        # If we found a very good graph, we might want to try local search on it
        # But for now, restarts are sufficient given the time limit.
        
    return best_edges
