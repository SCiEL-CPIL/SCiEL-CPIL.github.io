"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 135  # 15s safety margin before 150s kill

    def check_girth_fast(n, adj, k, g):
        """
        Check if graph has girth >= g.
        For a graph to have girth >= g (even), the BFS tree from any vertex
        up to depth r = g//2 - 1 must be a tree (no cycles, correct vertex counts).
        """
        r = g // 2 - 1
        expected_counts = [1] + [k * (k - 1) ** i for i in range(r)]
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            counts = [0] * (r + 1)
            counts[0] = 1
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= r:
                            counts[dist[v]] += 1
                            queue.append(v)
            
            if counts != expected_counts:
                return False
        return True

    def random_regular_graph(n, k):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def margulis_graph(p):
        """
        Construct the Margulis 3-regular graph on SL(2, Z_p).
        """
        mats = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    if a == 0:
                        for d in range(p):
                            if (b * c) % p == (p - 1) % p:
                                mats.append((0, b, c, d))
                    else:
                        inv_a = pow(a, p - 2, p)
                        for b in range(p):
                            for c in range(p):
                                d = (1 + b * c) * inv_a % p
                                mats.append((a, b, c, d))
        
        mat_to_idx = {m: i for i, m in enumerate(mats)}
        
        def mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            a = (a1 * a2 + b1 * c2) % p
            b = (a1 * b2 + b1 * d2) % p
            c = (c1 * a2 + d1 * c2) % p
            d = (c1 * b2 + d1 * d2) % p
            return (a, b, c, d)
        
        gen_A = (1, 1, 0, 1)
        gen_B = (1, 0, 1, 1)
        gen_C = (0, 1, p - 1, 0)
        
        n = len(mats)
        adj = [[] for _ in range(n)]
        
        for i, m in enumerate(mats):
            m1 = mul(m, gen_A)
            m2 = mul(m, gen_B)
            m3 = mul(m, gen_C)
            
            j1 = mat_to_idx[m1]
            j2 = mat_to_idx[m2]
            j3 = mat_to_idx[m3]
            
            adj[i].append(j1)
            adj[i].append(j2)
            adj[i].append(j3)
            
        return adj

    # Strategy:
    # 1. Try Margulis for small primes (7, 11) - fast deterministic check.
    # 2. If that fails, try random search for n in [512, 1000] with a time budget.
    # 3. If that fails, fall back to LUW construction (guaranteed valid, but large).

    # 1. Margulis
    for p in [7, 11]:
        if time.time() > deadline - 10:
            break
        try:
            adj = margulis_graph(p)
            n = len(adj)
            if check_girth_fast(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        except Exception:
            pass

    # 2. Random search
    # We want to find a graph with n as small as possible.
    # Start from lower bound 512.
    # Use a time budget for random search, e.g., 100 seconds.
    # If we find a valid graph, return it.
    # If not, fall back to LUW.
    
    # To maximize the chance of finding a small graph, we can iterate n from 512 upwards.
    # For each n, try a few random graphs.
    
    # However, the probability of finding a girth-16 graph in a random 3-regular graph
    # with n=512 is likely extremely low.
    # The cage has n=936.
    # So we should probably focus on n around 900-1000 if we are doing random search.
    # But let's try a range.
    
    # Let's allocate 100 seconds for random search.
    t_start = time.time()
    t_budget = 100
    
    # Try n from 512 to 1200 (step 2, since n must be even for k=3)
    # We can prioritize larger n if we suspect smaller ones are impossible,
    # but the lower bound is 512.
    # Let's just iterate.
    
    # To speed up, we can do a quick check for small cycles first?
    # check_girth_fast is already quite fast (BFS).
    # For n=1000, g=16, r=7.
    # BFS from each vertex visits up to 1+3+6+12+24+48+96+192 = 382 vertices.
    # 1000 * 382 * 3 (degree) ~ 1M operations.
    # In Python, this might take ~0.1-0.5s.
    # So we can do ~200-1000 checks in 100s.
    
    # Let's try to be smart:
    # If n < 936, it's likely impossible (cage is 936).
    # But maybe there's a graph with n=900? Unlikely if 936 is the cage.
    # The cage is the smallest. So n must be >= 936.
    # So we should start searching from 936!
    
    # Wait, the task says "The smallest known such graph has 936 vertices".
    # So n must be at least 936.
    # Let's start from 936.
    
    found = False
    for n in range(936, 1300, 2):
        if time.time() - t_start > t_budget:
            break
        
        # Try a few random graphs for this n
        # How many?
        # Let's say 10 per n value.
        # 10 checks * 0.5s = 5s per n.
        # 100s / 5s = 20 n values.
        # 936, 938, ..., 974.
        
        for _ in range(10):
            if time.time() - t_start > t_budget:
                break
            
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
            
            if check_girth_fast(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # 3. Fallback: LUW
    # This is guaranteed to be valid (girth >= g) but large.
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
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

# EVOLVE-BLOCK-END
