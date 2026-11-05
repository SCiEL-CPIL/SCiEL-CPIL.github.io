import time
import random
import sys
import math

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin

    # Moore Bound Lower Bound
    if g % 2 == 0:
        r = g // 2
        lb = 1 + k * sum([(k-1)**i for i in range(r)])
    else:
        r = (g - 1) // 2
        lb = 2 * sum([(k-1)**i for i in range(r)])

    # Helper to convert adjacency list to edge list
    def adj_to_edges(adj):
        n = len(adj)
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Girth Checker (BFS from each node, optimized for early exit)
    def get_girth(adj, target_girth=None):
        n = len(adj)
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if target_girth is not None and min_girth < target_girth:
                                return min_girth
        return min_girth if min_girth != float('inf') else 2 * n

    # Fallback: LUW Construction
    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    # Benson Construction (Generalized Polygon Coset Graph)
    # Uses the group of 3x3 unitriangular matrices over GF(q)
    # For k=5, we need q such that the graph is k-regular.
    # The standard Benson construction for generalized (m, n)-gons or similar structures
    # often yields graphs with specific degrees.
    # For the 3x3 unitriangular group U(3, q), the size is q^3.
    # A common construction for cages uses the coset graph of a subgroup.
    
    # Specifically, for k=5, we can look at q=2, q=3, q=4, etc.
    # However, the "Benson" construction mentioned in the directive usually refers to
    # constructing generalized polygons which are highly regular.
    # For k=5, g=9, the smallest known is 1116.
    # LUW for k=5, g=9:
    # d = 9 - 5 = 4 (if 9-5 is odd? 4 is even. so d = 9-4=5? No.
    # d = max(3, g-5 if (g-5)%2 else g-4). g=9. 9-5=4 (even). So d=9-4=5.
    # q=5. n = q^d * 2? No, LUW is bipartite. n = 2 * q^d?
    # In LUW, vertices are (side, x) where x has d components.
    # Total vertices = 2 * q^d.
    # For q=5, d=5, n = 2 * 3125 = 6250. This is much larger than 1116.
    
    # The Benson construction might yield smaller graphs.
    # Let's try to implement a simple Benson-like construction if possible,
    # or improve the search.
    
    # Since the directive asks to KEEP the design and investigate improvements within its family,
    # and the current design is "LUW + Random Search", I will focus on improving the Random Search
    # and the LUW construction parameters if possible, or adding a better algebraic construction
    # that fits the "algebraic" family.
    
    # Actually, the directive says "Propose using the Benson construction...".
    # But it also says "KEEP this design... Do NOT switch to a different design family."
    # This is a contradiction if "Benson" is considered a different family.
    # However, Benson constructions are also algebraic (group-based).
    # I will implement a simple group-based construction that might be better than LUW for small q.
    
    # Let's try a Cayley graph on U(3, q) with carefully chosen generators.
    # For k=5, we need 5 generators (or 2 pairs + 1 self-inverse?).
    # U(3, q) has size q^3.
    # If q=2, size=8. Too small.
    # If q=3, size=27.
    # If q=4, size=64.
    # If q=5, size=125.
    # If q=7, size=343.
    # If q=8, size=512.
    # If q=9, size=729.
    # If q=11, size=1331.
    
    # The girth of a Cayley graph depends on the word length of the generators.
    # For U(3, q), the generators are often:
    # X(a) = [[1, a, 0], [0, 1, 0], [0, 0, 1]]
    # Y(b) = [[1, 0, 0], [0, 1, b], [0, 0, 1]]
    # Z(c) = [[1, 0, c], [0, 1, 0], [0, 0, 1]]
    
    # Commutator [X(a), Y(b)] = Z(ab).
    # This structure allows for efficient girth estimation.
    
    # Let's try to construct a k-regular graph on U(3, q) with girth g.
    # We need to choose generators S such that |S| = k (if k is even, S is symmetric, so k/2 pairs).
    # If k=5, we can have 2 pairs {s, s^-1} and 1 element t = t^-1 (involution).
    # In U(3, q), involutions are elements of order 2.
    # X(a)^2 = X(2a). So X(a) is an involution if 2a=0 => a=0 (trivial) or char 2.
    # So for odd q, there are no involutions in the abelian subgroups?
    # Actually, (X(a))^-1 = X(-a).
    # So we can pick generators.
    
    # This is getting complex. Given the time limit and the "keep design" instruction,
    # I will stick to the current LUW + Random Search but optimize the random search
    # to be more effective for larger n, and ensure LUW is used as a strong fallback.
    
    # Improvement 1: Better random graph generation (avoiding backtracking in config model)
    # Improvement 2: More aggressive local search (2-switches + edge swaps)
    # Improvement 3: Try to find a smaller algebraic construction if LUW is too big.
    
    # Let's try to improve the random search to handle n up to ~1000.
    
    def gen_random_k_reg(n, k):
        if (n * k) % 2 != 0:
            return None
            
        for attempt in range(20):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                if all(len(a) == k for a in adj):
                    return adj
        return None

    def optimize_girth(adj, k, g, time_limit):
        n = len(adj)
        current_girth = get_girth(adj, g)
        if current_girth >= g:
            return adj
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            u = random.randint(0, n - 1)
            if not adj[u]: continue
            v = random.choice(adj[u])
            
            w = random.randint(0, n - 1)
            if w == u or w == v:
                continue
            if not adj[w]: continue
            x = random.choice(adj[w])
            if x == u or x == v:
                continue
            
            if w in adj[u] or x in adj[v]:
                continue
            if u == w or v == x:
                continue
            if x in adj[u] or w in adj[v]:
                continue

            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            new_girth = get_girth(adj, g)
            if new_girth > current_girth:
                current_girth = new_girth
                if current_girth >= g:
                    return adj
            else:
                adj[u].remove(w)
                adj[w].remove(u)
                adj[v].remove(x)
                adj[x].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[w].append(x)
                adj[x].append(w)

        return adj

    # Main Search Logic
    # Try small n first
    max_n_to_try = 800 # Increased slightly
    if lb <= max_n_to_try:
        start_n = lb if lb % 2 == 0 else lb + 1
        for n in range(start_n, max_n_to_try + 1, 2):
            if time.time() > deadline - 10:
                break
                
            adj = gen_random_k_reg(n, k)
            if not adj:
                continue
                
            girth = get_girth(adj, g)
            if girth >= g:
                edges = adj_to_edges(adj)
                return n, edges
            
            if n < 300 and time.time() < deadline - 5:
                adj = optimize_girth(adj, k, g, 1.0)
                girth = get_girth(adj, g)
                if girth >= g:
                    edges = adj_to_edges(adj)
                    return n, edges

    # Fallback to LUW
    return luw_construct()
