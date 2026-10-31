"""Construct small 3-regular graphs of large girth via random voltage lifts."""
import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 135  # 15s safety margin before 150s kill
    rng = random.Random(42)
    
    # Moore bound for (3, g): 2 * sum_{i=0}^{floor((g-1)/2)} 3^i
    # For g=16: 2*(1+3+9+27+81+243) = 2*364 = 728... let me compute properly
    # Actually for girth g, radius r = floor((g-1)/2)
    # Moore bound = 2 * sum_{i=0}^{r} (k-1)^i for k-regular
    # For k=3, g=16: r = floor(15/2) = 7, sum = 1+3+9+27+81+243+729+2187 = 3279, *2 = 6558
    # Wait, that's too high. Let me reconsider.
    # The Moore bound for (d, g) regular graph with girth g:
    # If g = 2r+1 (odd): n >= d * sum_{i=0}^{r-1} (d-1)^i
    # If g = 2r (even): n >= 2 * sum_{i=0}^{r-1} (d-1)^i
    # For k=3, g=16 (even, r=8): n >= 2 * sum_{i=0}^{7} 3^i = 2*(1+3+9+27+81+243+729+2187) = 2*3280 = 6560
    # Hmm, but the task says best lower bound is 512 and smallest known is 936.
    # Let me just use a reasonable lower bound and scan upward.
    
    # Actually, let me use the standard Moore bound formula:
    # For girth g and degree k:
    # If g is even, g=2r: n >= 2*(1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
    # If g is odd, g=2r+1: n >= k*(1 + (k-1) + ... + (k-1)^(r-1))
    
    def moore_bound(k, g):
        if g % 2 == 0:
            r = g // 2
            s = sum((k-1)**i for i in range(r))
            return 2 * s
        else:
            r = (g - 1) // 2
            s = sum((k-1)**i for i in range(r))
            return k * s
    
    lb = moore_bound(k, g)
    
    # For k=3, g=16: lb = 2*(1+3+9+27+81+243+729+2187) = 2*3280 = 6560
    # But the known answer is 936, so the Moore bound is very loose here.
    # The task says best lower bound is 512. Let me start from a smaller value.
    # Actually, let me start from a practical lower bound.
    # For practical purposes, let's start from max(lb, 100) but the known best is 936.
    # Let me start scanning from a reasonable point.
    
    # Strategy: Use voltage lifts over small base graphs
    # Base graph: 3-regular, small
    # The lift of a base graph with m voltage levels has n = |V_base| * m vertices
    
    # Try different base graphs and m values
    # For a 3-regular base graph on v vertices, the lift has v*m vertices
    
    # Let's try base graphs of various sizes
    # Key insight: the girth of the lift depends on the voltages assigned to edges
    
    # A simpler approach: directly construct random 3-regular graphs and check girth
    # But for large n this is slow. Let me use the voltage graph approach.
    
    # Voltage graph approach:
    # 1. Pick a small 3-regular base graph B with v vertices
    # 2. Assign voltages (elements of Z_m) to each directed edge
    # 3. The lift has v*m vertices, vertex (u, a) where u in V(B), a in Z_m
    # 4. Edge from (u, a) to (v, a + voltage(u,v)) for each edge (u,v) in B
    # 5. Girth check: no non-backtracking closed walk in B with length < g has voltage sum 0 mod m
    
    # For efficiency, let's use the fact that we need girth >= g
    # So we need: for all non-backtracking walks of length 1 to g-1 in B, 
    # the voltage sum is NOT 0 mod m
    
    # This is a constraint satisfaction problem on the voltages.
    
    # Let me try a different approach: random 3-regular graph generation + girth check
    # For n up to ~1000, we can generate random 3-regular graphs and check girth
    
    # Actually, the most practical approach given the time constraint:
    # 1. Start with the known construction or a good heuristic
    # 2. Try to find smaller graphs by random search
    
    # Let me use a hybrid: try voltage lifts with small base graphs
    
    # Base graph options: 
    # - K_4 (4 vertices, 3-regular, girth 3) - too small girth
    # - Cube Q_3 (8 vertices, 3-regular, girth 4)
    # - Petersen (10 vertices, 3-regular, girth 5)
    # - Dodecahedron (20 vertices, 3-regular, girth 5)
    
    # For the lift to have girth >= 16, the base graph's girth doesn't matter as much
    # as the voltage assignments avoiding short zero-sum walks.
    
    # Let me try: Petersen graph as base, m from some minimum upward
    # n = 10 * m, so for n=936, m=93.6... not integer. 
    # For n=100, m=10; n=940, m=94
    
    # Actually, let me just do random 3-regular graph search with girth verification.
    # For each n starting from a lower bound, try random constructions.
    
    def check_girth(n, edges, g):
        """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
        if n == 0:
            return True
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do non-backtracking BFS up to depth (g-1)//2
        # If we find a cycle of length < g, return False
        # Non-backtracking: when at vertex v coming from u, don't go back to u
        
        max_depth = (g - 1) // 2  # We need to check up to this depth for cycles
        
        for start in range(n):
            # BFS state: (current_vertex, previous_vertex, depth)
            # Use a set to track visited (vertex, prev) pairs to avoid cycles in BFS
            visited = set()
            queue = [(start, -1, 0)]
            visited.add((start, -1))
            
            while queue:
                v, prev, depth = queue.pop(0)
                if depth >= max_depth:
                    continue
                for w in adj[v]:
                    if w == prev:
                        continue  # non-backtracking
                    if depth + 1 >= g:  # shouldn't happen with max_depth
                        continue
                    # If we've seen (w, v) before, we found a cycle
                    if (w, v) in visited:
                        # Cycle length is depth + 1 + (depth when we first visited)
                        # Actually, this means we found a cycle of length < g
                        return False
                    if (w, v) not in visited:
                        visited.add((w, v))
                        queue.append((w, v, depth + 1))
        
        return True
    
    # This BFS approach is O(n * degree * max_depth) which is fine for n~1000
    
    # Now, generate random 3-regular graphs. Use configuration model + rejection.
    def random_3_regular(n, rng):
        """Generate a random 3-regular graph on n vertices using configuration model."""
        if n < 4 or n % 2 == 1:
            return None
        # Create stubs: 3 per vertex
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        rng.shuffle(stubs)
        # Pair up stubs
        edges = []
        edge_set = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None  # loop, reject
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return None  # multi-edge, reject
            edge_set.add(key)
            edges.append(key)
        return edges
    
    # For larger n, the configuration model rejection rate is low.
    # Let me scan n from a lower bound upward.
    
    # Practical lower bound: the task mentions 512 as best lower bound.
    # Let's start from there but also try to go lower.
    # Actually, let me start from a smaller value to see if we can beat 936.
    
    # The known best is 936. Let me try to find something smaller.
    # Start from 512 and go up in steps.
    
    best_n = None
    best_edges = None
    
    # Try n values from a lower bound
    # For 3-regular, n must be even
    start_n = max(512, lb)  # But lb is 6560 which is too high, use 512
    start_n = 512  # Use the stated lower bound
    
    # Actually, let me be smarter. Try a range of n values.
    # For each n, try several random graphs.
    
    # Time management
    attempts_per_n = 50  # number of random graphs to try per n
    
    for n in range(start_n, min(start_n + 500, 2000), 2):  # even n only
        if time.time() > deadline:
            break
        for attempt in range(attempts_per_n):
            if time.time() > deadline:
                break
            edges = random_3_regular(n, rng)
            if edges is None:
                continue
            # Check girth
            if check_girth(n, edges, g):
                best_n = n
                best_edges = edges
                return n, edges
        # If we didn't find one for this n, try next
    
    # If we didn't find anything in the scan, fall back to a known construction
    # Use the L-U-W construction from the seed
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
