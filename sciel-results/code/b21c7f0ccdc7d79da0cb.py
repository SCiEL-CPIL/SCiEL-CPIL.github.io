import time
import random
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target: 3-regular graph, girth 16.
    # Known optimal: 936 vertices.
    # Strategy: Search for a voltage lift of an 8-cycle with group size 117.
    # If found, n=936.
    
    if k != 3 or g != 16:
        # Fallback to seed logic for other parameters
        return _seed_construct(k, g)
        
    # Try to find Boutin-like construction
    group_size = 117
    # Base graph: 8-cycle. Vertices 0..7.
    # Edges: (i, i+1) with voltage v_i, (i+1, i) with voltage -v_i.
    # Lifted graph vertices: (i, x) where i in 0..7, x in 0..116.
    # Total vertices: 8 * 117 = 936.
    
    # We need to choose v_0...v_7 such that no closed walk of length < 16 sums to 0 mod 117.
    # Closed walks on 8-cycle of length < 16:
    # Length 4: i, i+1, i+2, i+1, i -> v_i + v_{i+1} - v_{i+1} - v_i = 0. Always 0!
    # Wait, the base graph is a cycle. A closed walk of length 4 in the lift corresponds to a walk in the base.
    # In the base 8-cycle, a walk of length 4 can be i -> i+1 -> i+2 -> i+1 -> i.
    # Sum: v_i + v_{i+1} - v_{i+1} - v_i = 0.
    # This implies a cycle of length 4 in the lift?
    # No, the lift of a back-and-forth walk is a cycle of length 4 only if the voltages sum to 0.
    # Here they always sum to 0.
    # So a simple 8-cycle lift will ALWAYS have girth 4 (from back-and-forth walks).
    # Thus, the base graph cannot be a simple cycle if we want girth > 4.
    # The base graph must be a **tree** or a graph with no short cycles?
    # But a tree lift is a tree (infinite) or a graph with cycles only from the group.
    # To get a finite graph, the base must have cycles.
    # If the base has a cycle of length L, the lift has cycles of length L if the voltage sum is 0.
    # To avoid short cycles, the base graph should have **large girth**.
    # But we want a small base to get a small lift.
    # Contradiction?
    # Actually, Boutin's construction uses a **specific base graph** that is not a simple cycle.
    # It uses a **tree** of depth 3? Or a **cage** itself?
    
    # Alternative: Use the **Lazebnik-Ustimenko-Woldar** construction but with **optimized parameters**.
    # The seed uses q=k. For k=3, q=3.
    # Maybe we can use a **different q**?
    # LUW graphs are q-regular. So q must be 3.
    # The girth is determined by d.
    # For q=3, the girth of CD(d, 3) is roughly 2d+4?
    # If we want girth 16, we need d such that 2d+4 >= 16 => d >= 6.
    # If d=6, girth is ~16.
    # Number of vertices is 2 * 3^d = 2 * 729 = 1458.
    # This is smaller than 350,000 (which was d=11).
    # So using d=6 in LUW gives n=1458.
    # This is a valid graph with girth >= 16 (needs verification).
    # 1458 is better than 350,000.
    # Is 1458 better than 936? No.
    # But it's easy to implement.
    
    # Let's try to implement LUW with d=6 for k=3, g=16.
    # If girth is < 16, we fail.
    # LUW girth is exactly d+5 for odd d, d+6 for even d?
    # For d=6 (even), girth is 6+6=12? Or 6+5=11?
    # If girth is 12, it's invalid for g=16.
    # We need d such that girth >= 16.
    # If girth = d+5, we need d >= 11.
    # Then n = 2 * 3^11 = 354294.
    # So LUW is not good for k=3, g=16.
    
    # Let's try a **Randomized Local Search** for a 3-regular graph with girth 16.
    # We will start with a random 3-regular graph of size 936.
    # We will perform edge swaps to eliminate short cycles.
    
    n_target = 936
    if time.time() > deadline:
        return _seed_construct(k, g)
        
    # Generate random 3-regular graph
    # Using configuration model
    def random_3_reg(n):
        # n must be even
        if n % 2 != 0:
            n += 1
        half_edges = n * 3 // 2
        # Stubs
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        seen = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                continue # Skip loops for now, handle later
            if (min(u,v), max(u,v)) in seen:
                continue # Skip multi-edges
            seen.add((min(u,v), max(u,v)))
            edges.append((u, v))
        # This might not be exactly 3-regular if we skipped edges.
        # Better: use a standard random 3-regular generator.
        return n, edges

    # This is getting too complex and risky for 120s.
    # Let's stick to the **LUW** but with a **smaller d** if possible, or a **different construction**.
    
    # Actually, the **Boutin** construction for (3,16) is the only way to get 936.
    # Since I cannot reliably implement Boutin from memory, I will use the **LUW** construction
    # but I will **verify the girth** and if it's too small, I will **increase d**.
    # The seed already does this.
    
    # Is there a **simpler** algebraic construction?
    # **Circulant graphs**?
    # A circulant graph on 936 vertices with 3 generators.
    # Girth of circulant graphs is hard to control.
    
    # Let's try **Greedy Construction** with Girth Check.
    # Build a 3-regular graph by adding edges one by one, ensuring no short cycles.
    # This is O(n * girth) per edge.
    # For n=936, this is feasible.
    
    def greedy_3_reg_girth(n, g):
        # Initialize empty graph
        adj = [set() for _ in range(n)]
        edges = []
        
        # We need to ensure every vertex ends up with degree 3.
        # We can start with a random matching, then add edges.
        
        # 1. Create a random 1-regular graph (perfect matching)
        #    Since n is even, this is possible.
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(0, n, 2):
            u, v = vertices[i], vertices[i+1]
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            
        # 2. Add edges to reach degree 3.
        #    We need to add 2 more edges per vertex.
        #    Total edges to add: n.
        #    We pick pairs (u, v) where deg(u)<3 and deg(v)<3, and (u,v) is not an edge.
        #    And adding (u,v) does not create a cycle of length < g.
        
        def has_short_cycle(u, v, adj, g):
            # Check if adding (u,v) creates a cycle of length < g.
            # This is equivalent to checking if dist(u, v) < g - 1 in the current graph.
            # BFS from u, stop at depth g-2. If v is reached, cycle is too short.
            # Actually, if dist(u, v) <= g-2, then adding (u,v) creates a cycle of length dist+1 <= g-1.
            # So we need dist(u, v) >= g-1.
            
            visited = {u: 0}
            queue = deque([(u, 0)])
            while queue:
                curr, dist = queue.popleft()
                if dist >= g - 1:
                    break
                for nb in adj[curr]:
                    if nb not in visited:
                        if nb == v:
                            # Found v at distance dist+1
                            # Cycle length = dist + 1 + 1 (new edge) = dist + 2
                            # We need cycle length >= g
                            # So dist + 2 >= g => dist >= g - 2
                            # If dist < g - 2, then cycle is too short.
                            if dist + 2 < g:
                                return True
                            else:
                                return False
                        visited[nb] = dist + 1
                        queue.append((nb, dist + 1))
            return False
            
        # We need to add n edges.
        # We will try to add edges in a random order.
        # If we get stuck, we backtrack? No, just restart.
        
        for _ in range(100): # 100 restarts
            if time.time() > deadline:
                break
            # Reset graph
            adj = [set() for _ in range(n)]
            edges = []
            vertices = list(range(n))
            random.shuffle(vertices)
            for i in range(0, n, 2):
                u, v = vertices[i], vertices[i+1]
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                
            success = True
            # Add remaining edges
            # We need to add n edges.
            # We can pick random pairs of vertices with degree < 3.
            for _ in range(n):
                if time.time() > deadline:
                    success = False
                    break
                candidates = []
                # Collect all vertices with degree < 3
                low_deg = [i for i in range(n) if len(adj[i]) < 3]
                if not low_deg:
                    break
                # Pick random u, v from low_deg
                # To ensure we make progress, we can just pick random pairs
                # and check validity.
                found = False
                for _ in range(1000): # Try 1000 random pairs
                    u, v = random.sample(low_deg, 2)
                    if v in adj[u]:
                        continue
                    if has_short_cycle(u, v, adj, g):
                        continue
                    # Add edge
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    found = True
                    break
                if not found:
                    success = False
                    break
            
            if success:
                # Verify degree
                for i in range(n):
                    if len(adj[i]) != 3:
                        success = False
                        break
                if success:
                    return n, edges
                    
        return None

    result = greedy_3_reg_girth(936, 16)
    if result:
        return result
        
    # Fallback to seed
    return _seed_construct(k, g)

def _seed_construct(k, g):
    # Original seed code
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
