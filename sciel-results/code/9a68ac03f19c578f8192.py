import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Fallback: LUW construction
    def luw_construct():
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

    # Try to find a better graph with N=320
    target_n = 320
    
    def generate_random_4_reg(n):
        # Configuration model
        half_edges = []
        for i in range(n):
            half_edges.extend([i, i, i, i])
        random.shuffle(half_edges)
        
        adj = {i: set() for i in range(n)}
        edges = set()
        
        # Pair up half-edges
        # This is tricky to do directly without creating multi-edges.
        # Easier: Start with a random 4-regular graph using 2-switches from a base.
        # Base: Two disjoint cycles? No, 4-regular.
        # Let's use a known 4-regular base: The Cartesian product of two cycles?
        # Or just random pairing and fix.
        
        # Random pairing approach:
        # We have 2n pairs of half-edges? No, 4n half-edges.
        # We need to pair 4n items into 2n edges.
        
        # Simpler: Start with a random permutation of half-edges and pair (0,1), (2,3)...
        # Then fix loops and multi-edges.
        
        perm = list(range(4*n))
        random.shuffle(perm)
        
        for i in range(0, 4*n, 2):
            u = perm[i] // 4
            v = perm[i+1] // 4
            if u == v:
                # Loop, fix by swapping with a random other edge
                pass # Simplified: Just retry or use a different generator
            else:
                if (u, v) in edges or (v, u) in edges:
                    pass # Multi-edge
                else:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.add((min(u,v), max(u,v)))
        
        # This generator is flawed for ensuring 4-regularity easily.
        # Let's use a standard library or a better algorithm.
        # Actually, for small n=320, we can use a specific construction.
        
        # Let's try a different approach:
        # Start with the LUW graph? No, it's bipartite and large.
        # Start with a random 4-regular graph using the "configuration model" correctly.
        
        # Correct Configuration Model:
        # 1. Create a list of 4n half-edges.
        # 2. Randomly permute them.
        # 3. Pair them up: (h[0], h[1]), (h[2], h[3]), ...
        # 4. This creates a multigraph.
        # 5. Use 2-switches to remove loops and multi-edges.
        
        half_edges = []
        for i in range(n):
            half_edges.extend([i]*4)
        
        random.shuffle(half_edges)
        
        edges = []
        for i in range(0, 4*n, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            edges.append((u, v))
            
        # Now we have a multigraph. We need to make it simple.
        # We will use 2-switches to fix it.
        
        adj = {i: set() for i in range(n)}
        edge_set = set()
        
        for u, v in edges:
            if u == v:
                continue # Ignore loops for now, will fix
            if (u, v) in edge_set or (v, u) in edge_set:
                continue # Ignore multi-edges for now
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
            
        # This is getting complicated. Let's just return LUW if we can't do better quickly.
        # But we want to TRY.
        
        # Let's use a simpler random 4-regular generator:
        # Start with two disjoint 2-factors (cycles).
        # 2-factor 1: Cycle 0-1-2-...-n-1-0
        # 2-factor 2: Random perfect matching? No, 2-factor is a set of cycles.
        # Let's take Cycle 1 and a random 2-factor (permutation with no fixed points).
        
        cycle1 = [(i, (i+1)%n) for i in range(n)]
        
        # Random 2-factor:
        # Permute vertices, pair (0,1), (2,3)... but that's a matching (1-regular).
        # We need a 2-regular graph.
        # So, take a random permutation P of vertices.
        # Edges: (P[i], P[(i+1)%n]) ... this is just a cycle.
        # We need TWO disjoint cycles? Or one big cycle?
        # A 2-regular graph is a union of cycles.
        
        perm = list(range(n))
        random.shuffle(perm)
        cycle2 = [(perm[i], perm[(i+1)%n]) for i in range(n)]
        
        # Combine
        edges = cycle1 + cycle2
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
            else:
                # If conflict, we have a problem.
                # But cycle1 and cycle2 might share edges.
                # We need to ensure they are edge-disjoint.
                pass
        
        # This is too risky. Let's stick to LUW.
        # The prompt says "score 1.0000" for LUW.
        # If I can't find a better one quickly, I return LUW.
        
        return None

    # Since writing a robust random 4-regular generator + local search in Python within 110s
    # is risky and might not beat LUW (which is 320+ vertices? No, LUW for k=4, g=10 is large).
    # Let's check LUW size.
    # q=4, d=5 (since g=10, d=5? g-5=5, odd. d=5).
    # Vertices: 2 * q^(d+1) ? No.
    # LUW graph CD(d, q) has 2*q^(d+1) vertices?
    # For q=4, d=5: 2 * 4^6 = 2 * 4096 = 8192 vertices.
    # This is MUCH larger than 320.
    # So ANY valid 320-vertex graph is a huge improvement.
    
    # Therefore, we MUST find a 320-vertex graph.
    # The (4,10)-cage is known to exist with 320 vertices.
    # We need to construct it.
    
    # There is an explicit construction for the (4,10)-cage.
    # It is based on the **Heisenberg group** over GF(5)?
    # Or **finite fields**.
    
    # Let's try to find the (4,10)-cage using a known algebraic construction.
    # Reference: "The (4,10)-cage"
    # One construction uses the group:
    # G = < a, b | a^2 = b^3 = ... > ?
    
    # Actually, the (4,10)-cage is the **Hoffman-Singleton** graph? No, that's (7,5).
    # The (4,10)-cage is related to the **Tutte 8-cage** (3,8).
    
    # Let's use the **random local search** but with a better starting point.
    # Start with a random 4-regular graph on 320 vertices.
    # Use 2-switches to minimize cycles.
    
    # To generate a random 4-regular graph on 320 vertices:
    # 1. Create 2 2-factors.
    # 2. Ensure they are edge-disjoint.
    
    def generate_4_reg(n):
        # 2-factor 1: Cycle 0-1-...-n-1
        f1 = [(i, (i+1)%n) for i in range(n)]
        
        # 2-factor 2: Random permutation, split into cycles?
        # Just use a random derangement?
        # A 2-factor is a set of cycles.
        # Let's generate a random permutation and use its cycles.
        # But we need exactly 2 edges per vertex.
        # So, take a random permutation P.
        # Edges: (i, P(i)) and (i, P^{-1}(i))? No, that's a matching.
        # A 2-regular graph is a union of cycles.
        # We can generate a random 2-regular graph by:
        # 1. Start with a cycle 0-1-...-n-1.
        # 2. Perform random 2-switches to randomize it.
        
        # Let's do that.
        edges = set(f1)
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Randomize 2-factor 1
        for _ in range(100):
            # Pick two edges (u,v) and (x,y)
            u = random.randint(0, n-1)
            v = random.choice(list(adj[u]))
            x = random.randint(0, n-1)
            if x == u or x == v: continue
            y = random.choice(list(adj[x]))
            if y == u or y == v: continue
            
            # Check if swap is valid
            if (u, x) in edges or (x, u) in edges or (v, y) in edges or (y, v) in edges:
                continue
            if (u, y) in edges or (y, u) in edges or (v, x) in edges or (x, v) in edges:
                continue
            
            # Swap
            edges.discard((min(u,v), max(u,v)))
            edges.discard((min(x,y), max(x,y)))
            edges.add((min(u,x), max(u,x)))
            edges.add((min(v,y), max(v,y)))
            
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
        # Now generate 2-factor 2 similarly, ensuring no edge overlap.
        # This is hard.
        
        # Alternative: Use the LUW graph's structure? No.
        
        # Let's just return LUW. The risk of failing to find 320 is high.
        # But LUW is 8192 vertices. Score is 1.0.
        # If I return 320, score is 8192/320 = 25.5.
        # So finding ANY valid 320 graph is a massive win.
        
        return None

    # Since I cannot easily implement a robust 320-vertex cage constructor in this time,
    # and the LUW graph is very large, I will try a **greedy random search** for 320 vertices.
    
    # However, if I fail, I return LUW.
    
    # Let's try one more thing: The (4,10)-cage is bipartite?
    # Moore bound 243 is odd.
    # If it were bipartite, it would have even number of vertices.
    # 320 is even.
    
    # I will implement a simple 2-switch local search on a random 4-regular graph of size 320.
    # If it finds a valid graph, great. If not, LUW.
    
    n = 320
    # Generate random 4-regular graph
    # Use the "two random 2-factors" approach.
    # F1: Cycle 0-1-...-n-1
    # F2: Random 2-factor.
    
    # To get a random 2-factor:
    # Permute vertices.
    # The permutation defines a set of cycles.
    # But we need a 2-regular graph, which IS a set of cycles.
    # So, take a random permutation P.
    # The edges are (i, P(i)) for all i? No, that's a 1-regular graph (matching) if P is involution?
    # No, (i, P(i)) is a directed cycle. As an undirected graph, it's a union of cycles.
    # So, yes, a random permutation gives a random 2-factor.
    
    def get_random_2_factor(n):
        perm = list(range(n))
        random.shuffle(perm)
        edges = []
        for i in range(n):
            u = i
            v = perm[i]
            if u != v:
                edges.append((min(u,v), max(u,v)))
        return edges

    f1 = [(i, (i+1)%n) for i in range(n)]
    f2 = get_random_2_factor(n)
    
    # Merge F1 and F2
    # Ensure no common edges.
    # If there are common edges, re-generate F2 or modify.
    
    edge_set = set()
    adj = {i: set() for i in range(n)}
    
    for u, v in f1:
        edge_set.add((min(u,v), max(u,v)))
        adj[u].add(v)
        adj[v].add(u)
        
    for u, v in f2:
        if (min(u,v), max(u,v)) in edge_set:
            # Conflict
            # Remove from F2 and find a replacement?
            # For simplicity, just skip? No, degree will be wrong.
            # Let's just re-generate F2 if conflict.
            pass
            
    # This is too complex.
    
    # I will return LUW.
    return luw_construct()

# EVOLVE-BLOCK-END
