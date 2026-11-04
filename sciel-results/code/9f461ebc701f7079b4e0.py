import random
import time
import numpy as np

def construct(k, g):
    # k is 3, g is 16
    # Target: Find a 3-regular graph of size N with girth >= g.
    # Strategy: Randomized local search (2-switches) to increase girth.
    
    # Fallback: LUW construction (from seed) if search fails
    # We will implement a simplified LUW or just use the seed's logic if needed.
    # But first, try the search.
    
    # We want to minimize N.
    # Known best for n(3,16) is 936.
    # Moore bound is 766.
    # Let's try N in [936, 950, 1000, 1100, 1200, 1500]
    
    target_ns = [936, 950, 1000, 1100, 1200, 1500]
    time_limit_total = 110  # Leave 10s for overhead
    time_per_n = time_limit_total / len(target_ns)
    
    best_graph = None
    best_n = float('inf')
    
    for N in target_ns:
        if time.time() - start_time > time_limit_total: break
        
        # Time budget for this N
        t_start = time.time()
        t_limit = t_start + time_per_n
        
        while time.time() < t_limit:
            # Generate random 3-regular graph
            G = generate_random_3_regular(N)
            if G is None: continue
            
            # Local search
            success = local_search_girth(G, g, t_limit)
            if success:
                best_graph = G
                best_n = N
                break # Found for this N, try smaller N? No, we iterate N in increasing order.
                # So if we find for 936, we return it.
        
        # If we found a graph for this N, we can return it immediately since we want min N.
        if best_graph is not None and best_n == N:
            break

    if best_graph is not None:
        edges = []
        n = len(best_graph)
        for u in range(n):
            for v in best_graph[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges
    
    # Fallback: Use the LUW construction from the seed
    return construct_luw(k, g)

def generate_random_3_regular(N):
    # Configuration model
    # 3N stubs
    # Randomly pair them
    # Fix loops and multi-edges
    
    # Simple approach: Start with a random 3-regular multigraph, then fix.
    # Or use a known algorithm for simple 3-regular graphs.
    # For N up to 1500, simple random generation + repair is fast.
    
    # 1. Create list of stubs
    stubs = []
    for i in range(N):
        stubs.extend([i, i, i])
    
    random.shuffle(stubs)
    
    # 2. Pair them
    edges = []
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Loop, try to fix later
            pass
        edges.append((u, v))
    
    # Build adjacency list
    adj = {i: set() for i in range(N)}
    # Count degrees and check for multi-edges/loops
    # This simple generation often creates multi-edges.
    # We need to ensure simple graph.
    
    # Better: Use a loop to generate a simple 3-regular graph.
    # Algorithm:
    # 1. Start with empty graph.
    # 2. While not all vertices have degree 3:
    #    Pick two vertices u, v with degree < 3.
    #    If u != v and (u,v) not in edges: add edge.
    #    Else: backtrack or swap.
    
    # This is complex to implement robustly.
    # Let's use a simpler heuristic:
    # Generate a random permutation, connect i to i+1, i+2, i+3 mod N?
    # That's a circulant graph C_N(1, 2, 3)?
    # Girth of C_N(1,2,3) is small.
    
    # Let's stick to the configuration model + repair.
    # We will generate the edge list, then repair.
    
    # Actually, for small N, we can just use a library-like approach if available? No.
    # Let's implement a robust random 3-regular generator.
    
    # Method: Random 2-switches on a random multigraph.
    # 1. Generate random multigraph (pair stubs).
    # 2. While has_loops_or_multi_edges:
    #    Find a loop or multi-edge.
    #    Perform a 2-switch to remove it.
    
    # This is standard and works well for N < 1000.
    
    # Implementation:
    # adj as list of sets.
    
    adj = [set() for _ in range(N)]
    
    # 1. Random pairing
    stubs = []
    for i in range(N):
        stubs.extend([i, i, i])
    random.shuffle(stubs)
    
    for i in range(0, 2*N, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Loop
            # We need to fix this.
            # For now, just add it and fix later.
            pass
        # Add edge
        # We need to handle multi-edges.
        # Let's just build the multigraph first.
        
    # This is getting complicated.
    # Alternative: Use the fact that N is even? 3-regular graphs must have even N.
    # 936 is even.
    
    # Let's use a simpler generator:
    # Start with a random 3-regular graph from a known construction?
    # No, let's just use the LUW graph as the *starting* point for local search?
    # LUW graph is huge.
    
    # Let's try a different tactic:
    # The seed program IS the LUW graph.
    # If we can't find a small graph, we return the seed's graph.
    # The score will be 1.0.
    # To beat 1.0, we MUST find a graph with n < seed_n.
    # seed_n is ~100,000.
    # Any valid graph with n < 100,000 scores > 1.0.
    # We can easily find a graph with n=1000 and girth 16?
    # Maybe not easily.
    # But we can find a graph with n=10,000 and girth 16?
    # Random graph of size 10,000 has expected girth ~ log2(10000) ~ 13.
    # Still not 16.
    
    # We really need the local search to work.
    
    # Let's implement the local search carefully.
    
    pass

def construct_luw(k, g):
    # Copy of the seed logic to ensure fallback works
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
