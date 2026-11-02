import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 105  # 15s margin before 120s kill

    # Moore bound for k=3, g=16 is 766.
    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # Strategy: try algebraic constructions first (much higher girth potential),
    # then fall back to L-U-W if time runs out.

    # Phase 1: Try algebraic constructions with various group sizes
    # Vertices are pairs (u, v) in Z_m x Z_m, edges via generators
    # For k=3: generators A = {a1, a2, a3} with f: A -> Z_m
    # Edge: (u,v) -- (u+ai, v+f(ai))
    # For this to be 3-regular and undirected, we need:
    #   - The 3 generators must come in "inverse" pairs or be self-inverse
    #   - Actually for undirected: if (u,v)->(u+a, v+f(a)), then (u+a,v+f(a))->(u,v)
    #     requires (u+a - a, v+f(a) - f(-a)) = (u, v), so f(-a) = f(a)
    #   - Simpler: use generators that are involutions or pair them up
    #   - For k=3: use generators {a, -a, b} where f(a)=f(-a) and f(b)=-f(b) (i.e., 2*f(b)=0)
    #   - Or: use 3 generators a1,a2,a3 with a_i + a_j + a_k = 0 and f satisfying constraints
    #
    # Alternative simpler approach: use the "lifting" construction
    # Take a base 3-regular graph H on m vertices, lift to m*n vertices
    # This is complex. Let's try a different algebraic approach.
    #
    # Better: Use a circulant-like construction
    # Vertices: 0..n-1 where n = m * s
    # For k=3, g=16: try n values and construct via random 3-regular with girth check
    #
    # Most practical: generate random 3-regular graphs and check girth quickly
    # The key insight: random 3-regular graphs have girth ~ log(n), so for n~1000,
    # girth ~ 5-7 typically. We need girth 16, which is much harder.
    #
    # The LUW construction gives girth 16 but with n ~ 3^8 = 6561 vertices.
    # We need to do better.
    #
    # Let's try: for each candidate n, generate many random 3-regular graphs
    # and use a fast girth check. If any has girth >= 16, return it.
    # Probability is very low for n < 1000, but let's try.

    moore_bound = 1 + k * sum((k-1)**i for i in range(g//2))

    # Try algebraic construction: vertices (i, j) in Z_m x Z_m
    # Edges: (i,j) connected to (i+a, j+b) for (a,b) in S where S has 3 elements
    # and S is symmetric (if (a,b) in S then (-a,-b) in S) and no (0,0)
    # For k=3: S must have 3 elements, but symmetric set of size 3 in Z_m x Z_m
    # means one element is self-inverse: (a,b) with 2a=0, 2b=0
    # So S = {(a,b), (-a,-b), (c,d)} where (c,d) = -(c,d), i.e., 2c=0, 2d=0
    # This requires m even.
    #
    # Let's try m=2 for simplicity: Z_2 x Z_2 = {00, 01, 10, 11}
    # Self-inverse elements: (0,0), (1,0), (0,1), (1,1) - all are self-inverse in Z_2
    # S of size 3 from {(1,0),(0,1),(1,1)}: 
    #   S1 = {(1,0),(0,1),(1,1)} - this gives complete graph K4, girth 3
    # Not useful.
    #
    # For larger m: try m=4, S = {(1,0),(3,0),(2,1)} where (2,1)+(-2,-1)=(0,0)
    # Wait, (2,1) is not self-inverse unless 2*2=0 mod 4 (yes, 4=0) and 2*1=0 mod 4 (no, 2!=0)
    # So (2,1) is not self-inverse. We need (c,d) with 2c=0 and 2d=0 mod m.
    # For m=4: (0,0),(2,0),(0,2),(2,2) are self-inverse.
    # S = {(1,0),(3,0),(2,1)}: (2,1) not self-inverse. Invalid.
    # S = {(1,0),(3,0),(2,0)}: (2,0) self-inverse. But (1,0)+(3,0)+(2,0)=(6,0)=(2,0)≠(0,0)
    # That's fine, we just need each edge to be valid.
    # Check: from (i,j): neighbors are (i+1,j),(i+3,j),(i+2,j) mod 4
    # This is a circulant on Z_4 x Z_4 with steps (1,0),(3,0),(2,0)
    # = circulant with steps 1,3,2 in the first coordinate, all same second coord
    # This decomposes into 4 copies of circulant Z_4 with steps {1,2,3} = K4, girth 3. Bad.
    #
    # Let's try S = {(1,1),(3,3),(2,2)}: (2,2) self-inverse? 2*2=4=0 mod 4, yes.
    # From (i,j): neighbors (i+1,j+1),(i+3,j+3),(i+2,j+2)
    # This is like a circulant on Z_4 x Z_4 with 3 steps.
    # Let's check girth... this might work for small m.
    #
    # Actually, let me just try random 3-regular graphs with fast girth check.
    # For n around 1000-2000, random 3-regular has girth ~6-8. We need 16.
    # This is extremely unlikely randomly.
    #
    # The LUW construction is the best known practical approach.
    # Let's just use it but try to minimize n by choosing optimal parameters.

    # Try to find a smaller LUW-like construction
    # LUW: q=3, d such that girth >= 16
    # Girth of LUW(q,d) is at least 2d-2 for d >= 4 (roughly)
    # For g=16, we need d >= 9
    # n = 2 * q^d = 2 * 3^9 = 39366 for d=9
    # For d=8: girth ~ 14, n = 2*3^8 = 13122
    #
    # The known best is 936 vertices. We can't beat that with LUW.
    # Let's just return LUW with minimal d that gives girth >= 16.

    # Determine minimum d for LUW with q=3 to have girth >= 16
    # LUW girth is at least 2*d - 2 (conservative)
    # So d >= 9 for girth >= 16
    # But actual girth may be higher. Let's compute exactly.

    # For time efficiency, let's try a few approaches:
    # 1. Quick random search for small n (very unlikely to succeed)
    # 2. LUW construction with optimal parameters

    # Phase 1: Quick random attempts for n in [moore_bound, moore_bound+100]
    # Limit to ~20 seconds
    phase1_deadline = start_time + 20
    for n in range(moore_bound, moore_bound + 100):
        if time.time() > phase1_deadline:
            break
        for attempt in range(5):
            if time.time() > phase1_deadline:
                break
            edges = _generate_random_regular_graph(n, 3)
            if edges is None:
                continue
            if _check_girth_fast(edges, n, g):
                return n, edges

    # Phase 2: LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original code
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

def _generate_random_regular_graph(n, k):
    """Generate a random simple k-regular graph using configuration model with retries."""
    if n * k % 2 != 0 or n < k:
        return None
    
    max_retries = 50
    for _ in range(max_retries):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        edges = set()
        adj = [[] for _ in range(n)]
        valid = True
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
        
        if valid:
            return list(edges)
    
    return None

def _check_girth_fast(edges, n, g):
    """Fast girth check using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking BFS: state is (vertex, previous_vertex)
    # We want to find if there's a cycle of length < g
    # A cycle of length L means: starting from (v0, v_{L-1}), after L-1 steps we reach (v0, v1)
    # where v1 != v_{L-1} (non-backtracking)
    
    depth_limit = g - 1  # We check up to depth g-1 to detect cycles of length < g
    
    for start in range(n):
        # BFS state: (current_vertex, previous_vertex)
        # visited[state] = depth
        visited = {}
        queue = []
        for nb in adj[start]:
            state = (start, nb)
            visited[state] = 1
            queue.append((start, nb, 1))
        
        while queue:
            u, prev, depth = queue.pop(0)
            if depth >= depth_limit:
                continue
            for v in adj[u]:
                if v == prev:
                    continue  # non-backtracking
                new_state = (u, v)
                if new_state in visited:
                    # Cycle detected
                    # The cycle length is depth + visited[new_state] + 1
                    # Wait, let me think more carefully
                    # We're at state (u, v) at depth depth
                    # Previously visited (u, v) at depth visited[new_state]
                    # This means there's a closed non-backtracking walk
                    # of length depth - visited[new_state]
                    # which corresponds to a cycle of that length
                    cycle_len = depth - visited[new_state]
                    if cycle_len < g:
                        return False
                else:
                    visited[new_state] = depth + 1
                    queue.append((u, v, depth + 1))
    return True

def _improve_girth(edges, n, g, max_iter, time_limit):
    return None
