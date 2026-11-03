import random
import time
import sys

def construct(k, g):
    # The task is specifically for k=4, g=10.
    # The known smallest graph has n=320.
    # We aim to construct a 320-vertex graph using a lift of K4.
    
    if k != 4 or g != 10:
        # Fallback for other cases (not the main focus, but keeps the function general)
        return _construct_fallback(k, g)

    # Base graph: K4
    # Vertices: 0, 1, 2, 3
    # Edges:
    # 0: (0,1), 1: (0,2), 2: (0,3), 3: (1,2), 4: (1,3), 5: (2,3)
    base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    num_base_vertices = 4
    num_base_edges = 6
    M = 80  # Target modulus for n = 4 * 80 = 320

    # Build adjacency list for base graph
    adj = [[] for _ in range(num_base_vertices)]
    for i, (u, v) in enumerate(base_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    # Generate non-backtracking closed walks of length L < g
    # We need to ensure no such walk has a voltage sum of 0 mod M.
    # A walk is a sequence of edges.
    # We store the sequence of edge indices.
    
    walks_by_length = {L: set() for L in range(3, g)}
    
    # DFS to find closed walks
    # State: (current_vertex, prev_edge_idx, path_edges)
    # Start from each vertex and each incident edge
    for start_v in range(num_base_vertices):
        for next_v, e0 in adj[start_v]:
            # Start with edge e0
            # Current vertex is next_v
            # We need to close back to start_v
            # Length L means L edges.
            # We already have 1 edge.
            
            def dfs(current_v, prev_e, path, depth):
                if depth >= g:
                    return
                if depth >= 3: # Minimum cycle length is 3
                    if current_v == start_v:
                        # Closed walk found
                        # Store the tuple of edges
                        walks_by_length[depth].add(tuple(path))
                    # Continue extending if depth < g-1
                    if depth < g - 1:
                        for next_v2, e_next in adj[current_v]:
                            if e_next == prev_e:
                                continue
                            dfs(next_v2, e_next, path + [e_next], depth + 1)
            
            dfs(next_v, e0, [e0], 1)

    # Flatten walks for faster checking
    # We only care about walks of length 3 to 9
    # We can pre-compute the "signature" of each walk: the count of each edge in the walk?
    # No, the sum is additive, so we just need the sum of voltages.
    # sum(v[e] for e in walk) % M != 0
    
    all_walks = []
    for L in range(3, g):
        for w in walks_by_length[L]:
            all_walks.append(w)
    
    # Optimize: Represent walks as counts of each edge?
    # Since the walk is a sequence, and addition is commutative, 
    # the sum depends only on the multiset of edges in the walk.
    # However, different walks might have the same multiset.
    # We can just check the sum directly.
    # To speed up, we can pre-compute a list of "edge count vectors" for each walk?
    # Or just keep the list of edge indices.
    
    # Let's convert walks to a format that allows fast sum calculation.
    # Since M=80 and there are only 6 edges, we can represent a walk by a tuple of 6 counts?
    # No, the sum is linear.
    # Let's just keep the list of edges.
    
    # Search for voltages
    # voltages = [v0, v1, v2, v3, v4, v5]
    
    deadline = time.time() + 100 # Leave 20s margin
    
    best_voltages = None
    
    # Randomized Local Search
    # Objective: Minimize the number of "bad" walks (sum == 0)
    # If count is 0, we are done.
    
    def count_bad_walks(volts):
        count = 0
        for w in all_walks:
            s = 0
            for e in w:
                s += volts[e]
            if s % M == 0:
                count += 1
        return count

    # Try multiple random restarts
    num_restarts = 100
    steps_per_restart = 500
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize voltages
        # Strategy: Random, or try to set them to distinct values to avoid small sums
        voltages = [random.randint(1, M-1) for _ in range(num_base_edges)]
        # Ensure gcd-like properties? Not strictly necessary for random search.
        
        current_bad = count_bad_walks(voltages)
        
        if current_bad == 0:
            best_voltages = voltages
            break
            
        # Hill Climbing
        for _ in range(steps_per_restart):
            if time.time() > deadline:
                break
            if current_bad == 0:
                best_voltages = voltages
                break
                
            # Pick a random edge to change
            e_idx = random.randint(0, num_base_edges - 1)
            old_volt = voltages[e_idx]
            new_volt = random.randint(0, M - 1)
            
            if new_volt == old_volt:
                continue
                
            voltages[e_idx] = new_volt
            new_bad = count_bad_walks(voltages)
            
            if new_bad < current_bad:
                current_bad = new_bad
            else:
                voltages[e_idx] = old_volt
        
        if current_bad == 0:
            best_voltages = voltages
            break

    if best_voltages is None:
        # If search failed, we cannot guarantee n=320.
        # Fall back to a known valid graph, even if larger.
        # The LUW graph is a safe fallback.
        return _construct_luw(k, g)

    # Construct the graph
    n = num_base_vertices * M
    edges = []
    
    # Map edge index to voltage
    # base_edges order:
    # 0: (0,1), 1: (0,2), 2: (0,3), 3: (1,2), 4: (1,3), 5: (2,3)
    
    for u in range(num_base_vertices):
        for v, e_idx in adj[u]:
            if u < v: # Add each edge once
                volt = best_voltages[e_idx]
                for x in range(M):
                    y = (x + volt) % M
                    i1 = u * M + x
                    i2 = v * M + y
                    edges.append((i1, i2))
                    
    return n, edges

def _construct_luw(k, g):
    # Original LUW code as fallback
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

def _construct_fallback(k, g):
    # Simple fallback for non-(4,10) cases
    # Just return a random regular graph if possible, or LUW
    return _construct_luw(k, g)
