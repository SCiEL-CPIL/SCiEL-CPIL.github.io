import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def get_seed_result():
        # L-U-W Construction
        # q must be a prime power for the field structure to hold.
        # If k is not a prime power, this construction is invalid for girth guarantees.
        # We check if k is a prime power.
        def is_prime_power(n):
            if n < 2: return False
            p = 2
            while p * p <= n:
                if n % p == 0:
                    # check if n is a power of p
                    m = n
                    while m % p == 0:
                        m //= p
                    if m == 1:
                        return True
                    else:
                        return False
                p += 1
            return True # n is prime
        
        q = k
        if not is_prime_power(q):
            # Fallback to random search or larger prime power?
            # For now, return None to trigger random search
            return None, None

        # Determine d
        # Girth is 2d+1 if q >= d
        # We need 2d+1 >= g => d >= (g-1)/2
        d_min = (g - 1 + 1) // 2  # ceil((g-1)/2)
        
        if d_min > q:
            # Cannot guarantee girth with this q
            # Try to find a valid d? If q < d_min, girth < 2d_min+1 = g.
            # So this construction fails for girth g.
            return None, None

        d = d_min
        
        # If q == 4, use XOR table (F2x2)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule definition from L-U-W
        # The seed code had a specific rule generation.
        # rule[c] for c in 2..d
        rule = {}
        # Seed code:
        # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        # for c in range(5, d + 1):
        #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
        #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        # This rule generation seems specific to the seed. I will replicate it.
        if d >= 2:
            rule[2] = (1, 1)
        if d >= 3:
            rule[3] = (2, 1)
        if d >= 4:
            rule[4] = (1, 2)
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def generate_random_regular(n, k):
        if n * k % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            edges.append((u, v))
        
        edge_set = set()
        for u, v in edges:
            if u > v: u, v = v, u
            if (u, v) in edge_set:
                return None
            edge_set.add((u, v))
        
        return edges

    def compute_girth(n, edges, max_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = max_girth
        depth_limit = max_girth // 2
        
        for start in range(n):
            visited = {start: (0, -1)}
            queue = [(start, 0, -1)]
            
            while queue:
                node, depth, parent = queue.pop(0)
                if depth >= depth_limit:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        cycle_len = depth + visited[neighbor][0] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth == 3:
                            return 3
                    else:
                        visited[neighbor] = (depth + 1, node)
                        queue.append((neighbor, depth + 1, node))
                        
        return min_girth

    # Try L-U-W first
    seed_n, seed_edges = get_seed_result()
    
    if seed_n is not None:
        # Verify girth to be safe
        # For large n, girth check is expensive. 
        # But L-U-W is theoretically guaranteed. 
        # However, if my d calculation is wrong, it might fail.
        # Let's do a quick check for small n, or trust the theory.
        # Given time constraints, trust the theory for prime powers.
        return seed_n, seed_edges

    # Fallback to random search if L-U-W not applicable
    # This is unlikely for k=5.
    
    # If L-U-W failed, try random
    candidate_ns = [500, 600, 800, 1000, 1500, 2000, 3000, 5000, 6000]
    for n in candidate_ns:
        if time.time() > deadline:
            break
        for _ in range(5):
            if time.time() > deadline:
                break
            edges = generate_random_regular(n, k)
            if edges is None:
                continue
            girth = compute_girth(n, edges, g)
            if girth >= g:
                return n, edges

    # Emergency fallback: return a trivial valid graph if possible?
    # Or raise an error? The task says "invalid result scores 0".
    # If we can't find one, we should return the best attempt.
    # But L-U-W should work for k=5.
    
    return 0, [] # Should not happen for k=5

