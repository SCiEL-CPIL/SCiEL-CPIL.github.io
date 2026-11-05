import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    n = 40
    
    def bitset_to_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    def count_paths_4(adj, u, v):
        """
        Count simple paths of length 4 from u to v.
        u-a-b-c-d-v
        """
        count = 0
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
                # Let D_v = N(v) \ {u, a, b}
                # We need to count edges (c, d) with c in S3, d in D_v.
                # Note: c != d is guaranteed if we handle sets correctly?
                # If c in D_v, then c is in N(v).
                # If we count edges between S3 and D_v, we might count (c, c) if c in S3 and c in D_v?
                # No, (c, c) is not an edge.
                # But we must ensure d != c.
                # If c in S3 and c in D_v, then c is a valid candidate for d?
                # No, d must be distinct from c.
                # So if c in D_v, we must exclude c from D_v when counting neighbors of c?
                # Actually, we are counting pairs (c, d).
                # For a fixed c, d must be in N(c) & D_v.
                # If c in D_v, is c in N(c)? No.
                # So c is not counted as a neighbor of itself.
                # So we can just count edges between S3 and D_v.
                
                D_v = adj[v] & ~(1 << u) & ~(1 << a) & ~(1 << b)
                # We want to count edges between S3 and D_v.
                # Iterate over the smaller set.
                if bin(S3).count('1') < bin(D_v).count('1'):
                    iter_set = S3
                    other_set = D_v
                else:
                    iter_set = D_v
                    other_set = S3
                
                iter_bits = iter_set
                while iter_bits:
                    lsb_iter = iter_bits & -iter_bits
                    iter_bits ^= lsb_iter
                    node = lsb_iter.bit_length() - 1
                    # Count neighbors of node in other_set
                    count += bin(adj[node] & other_set).count('1')
                    
        return count

    # Initial Greedy Construction
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
        
        # Add all possible edges
        changed = True
        while changed and time.time() < DEADLINE - 10:
            changed = False
            candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
            random.shuffle(candidates)
            for u, v in candidates:
                if time.time() > DEADLINE - 10:
                    return adj
                # Check if adding (u,v) creates C6
                # This is slow, so we do it carefully
                if count_paths_4(adj, u, v) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    # Simulated Annealing
    def simulated_annealing(adj, deadline):
        m = count_edges(adj)
        c6 = 0
        # Calculate initial C6 count
        # This is expensive, but we only do it once per restart
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    c6 += count_paths_4(adj, u, v)
        c6 //= 6 # Each C6 counted 6 times
        
        best_m = m
        best_adj = adj[:]
        
        T = 1.0
        lambda_pen = 10.0
        steps = 0
        max_steps = 50000
        
        while time.time() < deadline and steps < max_steps:
            steps += 1
            # Pick move
            if random.random() < 0.5:
                # Add edge
                candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
                if not candidates:
                    continue
                u, v = random.choice(candidates)
                k = count_paths_4(adj, u, v)
                delta_m = 1
                delta_c6 = k
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                m += 1
                c6 += k
            else:
                # Remove edge
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
                if not edges:
                    continue
                u, v = random.choice(edges)
                k = count_paths_4(adj, u, v)
                delta_m = -1
                delta_c6 = -k
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                m -= 1
                c6 -= k
            
            delta_score = delta_m - lambda_pen * delta_c6
            if delta_score > 0 or random.random() < pow(2.718, delta_score / T):
                # Accept
                if c6 == 0 and m > best_m:
                    best_m = m
                    best_adj = adj[:]
            else:
                # Reject: Revert
                if delta_m == 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    m -= 1
                    c6 -= k
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    m += 1
                    c6 += k
            
            # Cool down
            T *= 0.999
            if T < 0.01:
                T = 0.01
        
        return best_adj, best_m

    best_adj = None
    best_m = 0
    
    # Run a few restarts
    for i in range(5):
        if time.time() > DEADLINE - 20:
            break
        adj = greedy_init(12345 + i)
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_adj = adj
        
        # Run SA on this initial solution
        # Allocate remaining time / 2 for SA
        sa_deadline = min(time.time() + 10, DEADLINE - 5)
        sa_adj, sa_m = simulated_annealing(adj, sa_deadline)
        if sa_m > best_m:
            best_m = sa_m
            best_adj = sa_adj

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
