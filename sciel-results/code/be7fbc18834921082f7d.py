import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n = 40

    def bitset_to_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 containing (u,v) would be u-a-b-c-d-v-u.
        So we need to check if there is a path of length 4 from u to v
        using only vertices other than u and v, AND the path vertices
        a,b,c,d are all distinct and not equal to u,v.
        
        Actually, we need to check if there exists a simple path of length 4 from u to v
        that doesn't use u or v internally.
        
        Path: u - a - b - c - d - v
        a in N(u) \ {v}
        b in N(a) \ {u, v, a}
        c in N(b) \ {u, v, a, b}
        d in N(c) \ {u, v, a, b, c} and d in N(v)
        
        We can use bitsets for efficiency.
        """
        # a in N(u) \ {v}
        S1 = adj[u] & ~(1 << v)
        while S1:
            lsb = S1 & -S1
            S1 ^= lsb
            a = lsb.bit_length() - 1
            
            # b in N(a) \ {u, v, a}
            S2 = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            while S2:
                lsb2 = S2 & -S2
                S2 ^= lsb2
                b = lsb2.bit_length() - 1
                
                # c in N(b) \ {u, v, a, b}
                S3 = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                
                # d in N(c) \ {u, v, a, b, c} AND d in N(v)
                # For each c in S3, check if N(c) & N(v) & ~(mask) is non-empty
                # where mask = {u, v, a, b, c}
                S3_iter = S3
                while S3_iter:
                    lsb3 = S3_iter & -S3_iter
                    S3_iter ^= lsb3
                    c = lsb3.bit_length() - 1
                    
                    # d must be in N(c) & N(v) and not in {u, v, a, b, c}
                    mask = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c)
                    if (adj[c] & adj[v] & ~mask):
                        return False
        return True

    def greedy_init(seed_val):
        random.seed(seed_val)
        adj = [0] * n
        # Start with 8 K5
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Add all possible edges greedily
        changed = True
        while changed and time.time() < DEADLINE - 10:
            changed = False
            candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
            random.shuffle(candidates)
            for u, v in candidates:
                if time.time() > DEADLINE - 10:
                    return adj
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    def local_search(adj, deadline):
        m = count_edges(adj)
        best_m = m
        best_adj = adj[:]
        
        T = 1.0
        lambda_pen = 5.0
        steps = 0
        max_steps = 100000
        
        while time.time() < deadline and steps < max_steps:
            steps += 1
            # Pick move: 1-for-1 edge swap or add/remove
            r = random.random()
            if r < 0.4:
                # Add edge
                candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
                if not candidates:
                    continue
                u, v = random.choice(candidates)
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    m += 1
                    if m > best_m:
                        best_m = m
                        best_adj = adj[:]
                else:
                    # Reject
                    pass
            elif r < 0.8:
                # Remove edge
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
                if not edges:
                    continue
                u, v = random.choice(edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                m -= 1
                # Accept with some probability to allow exploration
                if random.random() < 0.3:
                    pass
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    m += 1
            else:
                # 1-for-1 edge swap: remove (u1,v1) and add (u2,v2)
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
                if not edges or not non_edges:
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                
                # Temporarily remove (u1,v1)
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                # Try to add (u2,v2)
                if can_add_edge(adj, u2, v2):
                    adj[u2] |= (1 << v)
                    adj[v2] |= (1 << u2)
                    # m stays the same
                    # Check if this improves things (it doesn't directly, but allows future additions)
                    # We accept if it doesn't break C6-freeness
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
            
            # Cool down
            T *= 0.999
            if T < 0.01:
                T = 0.01
        
        return best_adj, best_m

    best_adj = None
    best_m = 0
    
    # Run multiple restarts
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > DEADLINE - 15:
            break
        adj = greedy_init(12345 + i * 77)
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_adj = adj
        
        # Run local search on this initial solution
        remaining = DEADLINE - 15 - time.time()
        if remaining > 5:
            ls_deadline = time.time() + remaining * 0.7
            ls_adj, ls_m = local_search(adj, ls_deadline)
            if ls_m > best_m:
                best_m = ls_m
                best_adj = ls_adj

    if best_adj is None:
        # Fallback
        best_adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)

    return bitset_to_edges(best_adj)
