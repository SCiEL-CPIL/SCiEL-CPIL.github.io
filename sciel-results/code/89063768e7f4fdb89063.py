import time
import random
import numpy as np
from collections import deque

_DEADLINE = 110.0

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    # Target: Find a 4-regular graph with girth >= 10 on 320 vertices
    if k == 4 and g == 10:
        n = 320
        t0 = time.time()
        
        # Deterministic backtracking search to construct 4-regular graph with girth 10
        # Build layer by layer from root, maintaining BFS tree structure
        # Girth 10 means: no cycles of length 3,4,5,6,7,8,9
        # Equivalently: in BFS from any vertex, vertices at distance i and j can only
        # be adjacent if |i-j| <= 1, and we need to avoid creating short cycles
        
        # Strategy: Build the graph using a BFS-like construction from vertex 0
        # Layer 0: {0}
        # Layer 1: 4 neighbors of 0
        # Layer 2: each vertex in layer 1 has 3 new neighbors (one back to layer 0)
        # Layer 3: each vertex in layer 2 has 4 new neighbors (one back to layer 1)
        # ...
        # For girth 10, the BFS tree from any vertex has depth at least 4 (since 2*4+1=9 < 10)
        # So we need to ensure no shortcuts create cycles shorter than 10
        
        # Key insight: For a 4-regular graph with girth 10:
        # - From any vertex, BFS layers grow as 1, 4, 12, 36, ... up to depth 4
        # - At depth 4, we have 1+4+12+36+108 = 161 vertices in the "ball" of radius 4
        # - Total vertices must be at least 161 (actually more due to overlapping balls)
        # - The Moore bound for (4,10) is 243, known best is 320
        
        # Backtracking approach:
        # 1. Start with vertex 0
        # 2. Add 4 neighbors (layer 1): vertices 1,2,3,4
        # 3. For each vertex in layer 1, add 3 new neighbors (layer 2)
        # 4. Continue building layers, ensuring no short cycles are created
        # 5. At the end, we need to "close" the graph by adding edges that
        #    complete the degree-4 requirement without creating short cycles
        
        # Simpler approach: Use a known construction pattern
        # The (4,10)-cage is known to have 320 vertices
        # We'll try to construct it using a systematic backtracking
        
        # Let's use a different strategy: start with a random 4-regular graph
        # and use local search to eliminate short cycles
        
        def gen_random_4reg(n):
            """Generate a random 4-regular simple graph on n vertices."""
            while True:
                stubs = []
                for i in range(n):
                    for _ in range(4):
                        stubs.append(i)
                random.shuffle(stubs)
                edges = []
                valid = True
                seen = set()
                for i in range(0, len(stubs), 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        valid = False
                        break
                    seen.add(key)
                    edges.append(key)
                if valid:
                    return edges

        edges = gen_random_4reg(n)
        
        # Build adjacency structure
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Build adjacency matrix for fast cycle detection
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        # Cost function: Count of short cycles (lengths 3-9)
        # Use trace of A^k / (2*k) for cycles of length k
        def calc_short_cycle_count(A):
            """Count total number of cycles of length 3 through 9."""
            count = 0
            Ak = A.copy()
            for k in range(3, 10):
                Ak = Ak @ A
                # trace(Ak) = k * (number of cycles of length k) * 2 / 2 = k * cycles
                # Actually trace(A^k) counts closed walks of length k
                # For simple cycles of length k, each cycle is counted 2k times
                # (k starting points, 2 directions)
                # But there are also non-simple closed walks
                # For our purposes, we just want to minimize this
                count += np.trace(Ak)
            return count

        current_cost = calc_short_cycle_count(A)
        
        # Local search: try to eliminate short cycles by edge swaps
        # A move: pick two edges (u,v) and (x,y), replace with (u,x) and (v,y)
        # This preserves 4-regularity
        
        # Precompute for faster move evaluation
        # We'll use a smarter approach: focus on vertices involved in short cycles
        
        def find_short_cycle_vertices(A, adj):
            """Find vertices that are part of short cycles."""
            # Use BFS from each vertex to find short cycles
            # This is expensive, so we'll use a heuristic
            # Vertices with many common neighbors are likely in triangles
            in_short_cycle = set()
            for u in range(n):
                for v in adj[u]:
                    # Check if u and v have common neighbors (triangle)
                    common = adj[u] & adj[v]
                    if common:
                        in_short_cycle.add(u)
                        in_short_cycle.add(v)
                        for w in common:
                            in_short_cycle.add(w)
            return in_short_cycle

        # Simulated annealing with focus on eliminating short cycles
        T = 2.0
        cooling = 0.9995
        best_cost = current_cost
        best_adj = [set(s) for s in adj]
        best_A = A.copy()
        
        # Precompute edge list for efficient move generation
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # Track which vertices are in short cycles for targeted moves
        short_cycle_verts = find_short_cycle_vertices(A, adj)
        
        step = 0
        while time.time() - t0 < _DEADLINE:
            step += 1
            
            if current_cost == 0:
                # Found a valid graph!
                final_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            final_edges.append((i, j))
                return n, final_edges

            # Periodically refresh the set of vertices in short cycles
            if step % 100 == 0:
                short_cycle_verts = find_short_cycle_vertices(A, adj)
                if not short_cycle_verts:
                    # No obvious short cycles from heuristic, do full check
                    if current_cost == 0:
                        final_edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    final_edges.append((i, j))
                        return n, final_edges

            # Generate a move: pick two edges to swap
            # Prefer moves that involve vertices in short cycles
            max_attempts = 50
            moved = False
            
            for _ in range(max_attempts):
                # Pick first edge, prefer vertices in short cycles
                if short_cycle_verts and random.random() < 0.7:
                    u = random.choice(list(short_cycle_verts))
                else:
                    u = random.randint(0, n-1)
                
                if not adj[u]:
                    continue
                v = random.choice(list(adj[u]))
                
                # Pick second edge, prefer vertices in short cycles
                if short_cycle_verts and random.random() < 0.7:
                    x = random.choice(list(short_cycle_verts))
                else:
                    x = random.randint(0, n-1)
                
                # Ensure x is not u or v
                attempts_x = 0
                while x in (u, v) and attempts_x < 10:
                    if short_cycle_verts and random.random() < 0.7:
                        x = random.choice(list(short_cycle_verts))
                    else:
                        x = random.randint(0, n-1)
                    attempts_x += 1
                if x in (u, v):
                    continue
                
                if not adj[x]:
                    continue
                y = random.choice(list(adj[x]))
                
                # Ensure y is not u or v
                attempts_y = 0
                while y in (u, v) and attempts_y < 10:
                    y = random.choice(list(adj[x]))
                    attempts_y += 1
                if y in (u, v):
                    continue
                
                # Check that (u,x) and (v,y) are not already edges
                if A[u, x] == 1 or A[v, y] == 1:
                    continue
                
                # Perform swap
                # Remove edges (u,v) and (x,y)
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                A[u, v] = 0; A[v, u] = 0
                A[x, y] = 0; A[y, x] = 0
                
                # Add edges (u,x) and (v,y)
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                A[u, x] = 1; A[x, u] = 1
                A[v, y] = 1; A[y, v] = 1
                
                new_cost = calc_short_cycle_count(A)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = [set(s) for s in adj]
                        best_A = A.copy()
                    moved = True
                    break
                else:
                    # Revert
                    adj[u].remove(x)
                    adj[x].remove(u)
                    adj[v].remove(y)
                    adj[y].remove(v)
                    A[u, x] = 0; A[x, u] = 0
                    A[v, y] = 0; A[y, v] = 0
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[x].add(y)
                    adj[y].add(x)
                    A[u, v] = 1; A[v, u] = 1
                    A[x, y] = 1; A[y, x] = 1
                    
                    # SA acceptance
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        # Accept
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        A[u, v] = 0; A[v, u] = 0
                        A[x, y] = 0; A[y, x] = 0
                        
                        adj[u].add(x)
                        adj[x].add(u)
                        adj[v].add(y)
                        adj[y].add(v)
                        A[u, x] = 1; A[x, u] = 1
                        A[v, y] = 1; A[y, v] = 1
                        
                        current_cost = new_cost
                        moved = True
                        break
            
            if not moved:
                pass
                
            T *= cooling
            if T < 0.1:
                T = 0.1

        # If we didn't find a valid graph, return the best we found
        # But it must be valid! So we need to fall back to LUW if invalid.
        # Check if best is valid
        if best_cost == 0:
            final_edges = []
            for i in range(n):
                for j in best_adj[i]:
                    if i < j:
                        final_edges.append((i, j))
            return n, final_edges

    return luw_construct(k, g)
