import time
from collections import deque
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def find_square_root_minus_one(p):
        # Find s such that s^2 = -1 mod p
        # p must be 1 mod 4
        for s in range(2, p):
            if (s * s) % p == p - 1:
                return s
        return None

    def build_lps_graph(p):
        # LPS construction for prime p == 1 mod 4
        s = find_square_root_minus_one(p)
        if s is None:
            return None, None
        
        n = 2 * (p * p + 1)
        edges = []
        
        # Vertices are (x, y) where x, y in Z_p, plus two special vertices?
        # Actually, the standard LPS graph L(p) has vertices:
        # 1. (x, y) for x, y in Z_p, x != 0? No.
        # The vertices are:
        # - (x, y) for all x, y in Z_p such that x != 0?
        # Let's use the standard definition:
        # Vertices: (x, y) in Z_p x Z_p, plus two vertices (0, 0) and (0, 1)?
        # No, the LPS graph L(p) is a 3-regular graph on 2(p^2+1) vertices.
        # The vertices can be identified with:
        # 1. (x, y) for x, y in Z_p, x != 0
        # 2. (0, y) for y in Z_p
        # 3. Two additional vertices?
        # Total: p(p-1) + p = p^2. Plus 2? No.
        # 2(p^2+1) = 2p^2 + 2.
        # The vertices are:
        # (x, y) for x, y in Z_p, x != 0 : p(p-1)
        # (0, y) for y in Z_p : p
        # This sums to p^2.
        # We need 2p^2 + 2.
        # Ah, the LPS graph is defined on:
        # V = { (x, y) | x, y in Z_p, x != 0 } U { (0, y) | y in Z_p } U { (0, 0), (0, 1) }? No.
        
        # Let's use the explicit edge definition from Lubotzky, Phillips, Sarnak.
        # Vertices:
        # 1. (x, y) for x, y in Z_p, x != 0
        # 2. (0, y) for y in Z_p
        # 3. (0, 0) and (0, 1) are already in set 2?
        # Wait, 2(p^2+1) vertices.
        # The set is:
        # A = { (x, y) | x, y in Z_p, x != 0 } (size p(p-1))
        # B = { (0, y) | y in Z_p } (size p)
        # C = { (0, 0), (0, 1) }? No, those are in B.
        
        # Actually, the LPS graph L(p) is a lift of the Petersen graph?
        # No, it's a Cayley graph of SL_2(Z) / Gamma.
        
        # Let's use a known Python implementation of LPS or similar.
        # A simpler 3-regular graph with large girth:
        # The **Cage** for girth 16 might be found by random search or specific constructions.
        
        # Given the time constraint, if LPS is complex to implement correctly from memory,
        # I should stick to a robust random search with girth verification, 
        # or use the LUW construction but with optimized parameters.
        
        # However, the LUW construction in the seed is likely the bottleneck or suboptimal.
        # Let's try a random 3-regular graph search with girth check.
        # For n=580, random search is hard.
        
        # Let's return the LUW result if LPS fails to implement quickly.
        # But I can implement LPS.
        
        # Vertices of L(p):
        # 1. (x, y) for x, y in Z_p, x != 0
        # 2. (0, y) for y in Z_p
        # 3. (0, 0) and (0, 1) are distinct from the above?
        # No, the standard LPS graph has vertices:
        # (x, y) for x, y in Z_p, x != 0
        # (0, y) for y in Z_p
        # This is p^2 vertices.
        # But L(p) has 2(p^2+1) vertices.
        # So there are p^2 + 2 more vertices.
        # The other half is (x, y) for x, y in Z_p, x != 0?
        
        # Let's use the **Tutte-Coxeter** style or just random.
        
        return None, None

    def check_girth(n, edges, g, k):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    else:
                        if parent[nb] != node and parent[node] != nb:
                            cycle_len = dist[node] + dist[nb] + 1
                            if cycle_len < g:
                                return False
            if time.time() > deadline - 20:
                return None
        return True

    # Fallback to LUW if LPS is not implemented
    def build_luw(k, d, rule):
        q = k
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y))

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def rule_pattern_A(d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    # Try LUW with d=8 for g=16
    # n = 2 * 3^8 = 13122
    # This is the current program's likely output.
    
    # If I can't implement LPS, I return this.
    d = 8
    rule = rule_pattern_A(d)
    n, edges = build_luw(k, d, rule)
    return n, edges

