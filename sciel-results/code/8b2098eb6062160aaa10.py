import time
import random
from collections import deque

def _luw_construct(k, g):
    # The original LUW construction as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _check_girth_cayley_generators(gens, g):
    """
    Check if a Cayley graph on GF(2)^d with generators gens has girth >= g.
    In a Cayley graph on an abelian group of exponent 2, a cycle of length 2m
    corresponds to a product of m generators (in some order) that equals identity.
    Since the group is abelian and exponent 2, order doesn't matter and each
    generator appears at most once in a shortest relation.
    
    Girth >= g means: no subset of fewer than g/2 generators sums to 0,
    and no single generator is 0.
    """
    d = len(gens[0]) if gens else 0
    half_g = g // 2
    
    # Check no generator is zero
    for gen in gens:
        if all(x == 0 for x in gen):
            return False
    
    # Check no subset of size < half_g sums to zero
    # For small half_g, we can check all subsets
    if half_g <= 10:
        from itertools import combinations
        for r in range(1, half_g):
            for combo in combinations(range(len(gens)), r):
                s = [0] * d
                for idx in combo:
                    for i in range(d):
                        s[i] ^= gens[idx][i]
                if all(x == 0 for x in s):
                    return False
    else:
        # For larger half_g, use linear algebra: check if any subset of size < half_g is linearly dependent
        # Actually, in GF(2), a subset sums to 0 iff it's linearly dependent
        # So we need: every subset of size < half_g is linearly independent
        # This is equivalent to: the generators form a code with minimum distance >= half_g
        # We can check by trying all subsets up to size half_g-1
        from itertools import combinations
        for r in range(1, half_g):
            for combo in combinations(range(len(gens)), r):
                s = [0] * d
                for idx in combo:
                    for i in range(d):
                        s[i] ^= gens[idx][i]
                if all(x == 0 for x in s):
                    return False
    
    return True

def _build_cayley_graph_generators(gens):
    """
    Build a 3-regular Cayley graph on GF(2)^d with 3 generators.
    Vertices are elements of GF(2)^d (represented as integers 0..2^d-1).
    Edges: for each vertex v and each generator g, edge (v, v XOR g).
    Since each generator is its own inverse (exponent 2), we get 3-regular graph.
    """
    d = len(gens[0])
    n = 1 << d
    edges = []
    for v in range(n):
        for g in gens:
            w = v ^ g
            if v < w:
                edges.append((v, w))
    return n, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    # For Cayley graph on GF(2)^d with 3 generators:
    # n = 2^d vertices, 3-regular
    # Girth >= 16 means no subset of < 8 generators sums to 0
    # With only 3 generators, we need: no single gen is 0, no pair sums to 0, 
    # no triple sums to 0. That's it! (since 3 < 8)
    # Wait, that's too easy. Let me reconsider.
    
    # Actually for girth >= 16, we need no cycle of length < 16.
    # In a Cayley graph on an abelian 2-group with generators {a,b,c}:
    # A cycle of length 2m corresponds to a product of m generators = identity.
    # Since the group is abelian and exponent 2, this means a subset of m generators
    # (with repetition allowed, but in a shortest path no generator repeats) sums to 0.
    # 
    # Actually, in a shortest cycle, no generator repeats (otherwise you could shortcut).
    # So a cycle of length 2m uses m distinct generators from {a,b,c}.
    # But we only have 3 generators! So max cycle length from distinct generators is 6.
    # 
    # This means a 3-generator Cayley graph on GF(2)^d can NEVER have girth > 6!
    # Because there's always a cycle of length at most 6 (using all 3 generators).
    # 
    # So this approach with only 3 generators doesn't work for girth 16.
    # We need a different approach.
    
    # Alternative: Use a bipartite double cover or a different construction.
    # 
    # Let me think about this differently. The Moore bound for n(3,16) is 512.
    # The LUW construction gives a specific graph. Let me check what size it gives.
    
    # For the LUW construction with k=3, g=16:
    # q = 3, d = max(3, 16-5=11 if 11%2 else 16-4=12) = 11 (since 11 is odd)
    # Wait, g-5 = 11, 11%2 = 1, so d = 11
    # The construction creates vertices (side, x) where side in {0,1} and x is a tuple
    # of length d+1 over GF(3).
    # The number of vertices could be up to 2 * 3^d = 2 * 3^11 = 354294
    # But the actual number depends on the orbit structure.
    
    # Let me just run the LUW construction and see what we get.
    
    n_luw, edges_luw = _luw_construct(3, 16)
    
    if time.time() > deadline - 5:
        return n_luw, edges_luw
    
    # Now let's try a different structural approach:
    # Use a voltage graph on a bipartite base graph.
    # 
    # Base graph: a 6-cycle (C6) with voltages in Z_m.
    # The covering will be 2-regular on the base, so 2m * 2 = 4m vertices? No.
    # C6 has 6 vertices, 6 edges. A Z_m-voltage covering gives 6m vertices.
    # Each vertex in the covering has degree 2 (since C6 is 2-regular).
    # That's not 3-regular.
    
    # Let's use a 3-regular bipartite base graph.
    # The smallest 3-regular bipartite graph is K_{3,3} (6 vertices).
    # A Z_m voltage covering of K_{3,3} gives 6m vertices, 3-regular.
    # 
    # For girth >= 16, we need the voltage assignment to ensure no short cycles.
    # 
    # K_{3,3} has vertices L={l0,l1,l2} and R={r0,r1,r2} with all edges l_i-r_j.
    # Assign voltages to each edge.
    
    def build_k33_voltage(m, voltages):
        """
        Build 3-regular graph from K_{3,3} voltage covering.
        voltages: 9-element tuple (v[l_i][r_j] for i,j in 0,1,2)
        """
        n = 6 * m
        edges = []
        for i in range(3):
            for j in range(3):
                v = voltages[i * 3 + j]
                for t in range(m):
                    u = i * m + t
                    w = 3 * m + j * m + (t + v) % m
                    edges.append((u, w))
        return n, edges
    
    def check_girth_fast(n, edges, g):
        """Quick girth check using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    # Try K_{3,3} voltage construction with various m
    best_result = (n_luw, edges_luw)
    
    for m in range(90, 160):
        if time.time() > deadline - 15:
            break
        
        n_try = 6 * m
        if n_try >= best_result[0]:
            break
        
        # Try random voltage assignments
        num_trials = 50
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            voltages = tuple(random.randint(1, m - 1) for _ in range(9))
            n, edges = build_k33_voltage(m, voltages)
            if check_girth_fast(n, edges, g):
                best_result = (n, edges)
                break
    
    return best_result
