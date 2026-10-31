"""Construct small 3-regular graphs with large girth using a backtracking
(non-backtracking) search with symmetry breaking and early cycle detection."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    # Deadline guard: must return before ~110s to be safe (evaluator kills at 120s)
    deadline = time.time() + 105.0
    rng = random.Random(42)

    # For k=3, g=16: we need a 3-regular graph with girth >= 16.
    # Strategy: build the graph vertex-by-vertex using a non-backtracking
    # BFS-like approach. We maintain partial assignments and check that
    # no non-backtracking walk of length < g returns to a vertex already
    # in the graph (which would create a cycle shorter than g).
    #
    # Key insight: In a graph of girth g, the non-backtracking tree of
    # depth (g-1)//2 around any vertex is a tree. For g=16, depth=7.
    # We use a constraint-satisfaction / backtracking approach.
    #
    # However, for k=3, g=16, the known best is 936 vertices. Building
    # from scratch via backtracking is extremely hard. Instead, we use
    # a different structural approach: the "cage" construction via
    # voltage graphs / covering graphs of small base graphs.
    #
    # Alternative: Use the known Exoo et al. construction or a
    # generalized Petersen-like / Coxeter-type construction.
    #
    # Most practical approach for this task: use a non-backtracking
    # walk-based verification and a greedy + local search to build
    # a 3-regular graph with girth >= g.
    #
    # Let's try: start with a cycle, then add chords/edges to make it
    # 3-regular while maintaining girth. This is essentially finding
    # a 3-regular graph with large girth.
    #
    # Actually, the most reliable approach within time limits:
    # Use the "girth-verified greedy construction" where we build
    # vertices one at a time, assigning edges to satisfy degree k,
    # and using BFS from each vertex to ensure no short cycles form.
    #
    # For k=3, g=16: Moore bound is ~ 2*(3^7 - 3^6)/2 + ... 
    # Actually Moore bound for (3,16): n >= 2 * (3^7 - 3^6)/2 + 1... 
    # Let me compute: for bipartite, n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # = 2 * sum_{i=0}^{7} 2^i = 2 * (2^8 - 1) = 510. So n >= 512 (even).
    # Known best: 936.
    #
    # Practical approach: Try to construct using the "non-backtracking
    # expansion" method. We'll use a randomized greedy approach with
    # backtracking on a limited depth.

    if k != 3:
        # Fallback: use the original Lazebnik-Ustimenko-Woldar for other k
        return _luw_fallback(k, g)

    # For k=3, g=16: use a structured construction.
    # The key challenge: build a 3-regular graph on n vertices with girth >= 16.
    #
    # Approach: Use a "lift" of a small graph. A common technique is to
    # take a base graph and "lift" it using a voltage group. If the base
    # graph has a certain structure, the lift can have large girth.
    #
    # Simpler approach that works: Use the "random greedy" with verification.
    # Start with n vertices, try to add edges to make 3-regular, checking
    # girth at each step. This is too slow for n=936.
    #
    # Best practical approach: Use the known algebraic construction.
    # The (3,16)-cage has 936 vertices. It can be constructed as a
    # covering graph. Let me use a specific known construction.
    #
    # Actually, let me use a different strategy: the "non-backtracking
    # matrix" approach. For a 3-regular graph, the non-backtracking
    # matrix B has size 6n x 6n (2 directed edges per undirected edge,
    # but non-backtracking means we exclude the reverse). Actually B is
    # (2m) x (2m) where m = 3n/2... no, m = 3n/2 edges, 2m directed edges.
    # The girth is related to the smallest r such that B^r has a nonzero
    # diagonal entry.
    #
    # Let me just use a well-known explicit construction for (3,16):
    # The graph can be described as follows (from Exoo et al.):
    # It's a 3-regular graph on 936 vertices. One construction uses
    # the group Z_2 x Z_2 x Z_3 x Z_7 x Z_11 or similar.
    #
    # For simplicity and correctness, let me use the following approach:
    # 1. Generate a random 3-regular graph on n vertices (for various n).
    # 2. Check its girth.
    # 3. If girth >= g, return it.
    # 4. Use local search (edge swaps) to increase girth.
    #
    # But this is unlikely to find girth 16 on 936 vertices randomly.
    #
    # Let me use the "greedy with backtracking" on a tree-like structure:
    # Build the graph as a "non-backtracking tree" unfolded to depth 7,
    # then identify boundary vertices to close it up into a 3-regular graph.
    # This is essentially the "cage" construction method.

    # For time efficiency, let me use a specific known construction.
    # The (3,16)-cage: I'll use the construction from the literature.
    # One way: it's a normal Cayley graph or a covering graph.
    #
    # Let me try: the graph is a covering of K_4 (or another small graph)
    # with a voltage group. 
    #
    # Actually, the most reliable approach within 100s:
    # Use a "non-backtracking BFS" to verify, and construct via
    # the "Bose-Chowla" type or "Sylvester" type construction.
    #
    # Let me use the following explicit construction for (3,16):
    # Based on the paper by Exoo et al., the graph has 936 vertices.
    # I'll implement a construction that produces a valid (3,16)-cage
    # or close to it.
    #
    # Given the difficulty, let me use a pragmatic approach:
    # 1. Try to use the original LUW construction (gives girth ~10-12).
    # 2. Try a "blow-up" or "lifting" to increase girth.
    # 3. Verify with non-backtracking walks.
    #
    # Actually, the simplest correct approach: use the original LUW
    # but with larger parameters to get higher girth, accepting more
    # vertices. For k=3, the LUW graph CD(11, 3) has girth 16 and
    # 3^11 * 2 = 177147 vertices. That's way too many.
    #
    # Let me use a different known construction: the "Coxeter" type
    # or the "generalized Petersen" type.
    #
    # For (3,16), one known construction uses the group:
    # G = Z_2 x Z_2 x Z_3 x Z_7 x Z_11 (order 924... not 936)
    # 
    # Let me just use the original LUW as a baseline and try to improve
    # with a local search that increases girth.

    # First, get the LUW graph
    n, edges = _luw_fallback(3, 16)
    
    # Verify girth of LUW graph
    girth = _compute_girth(n, edges)
    
    if girth >= 16:
        return n, edges

    # Try to improve via edge swaps (2-opt) to increase girth
    # This is a local search: find edges whose removal and reconnection
    # increases the girth.
    
    edge_set = set()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
        adj[u].add(v)
        adj[v].add(u)
    
    # Try random edge swaps to increase girth
    best_girth = girth
    best_edges = list(edges)
    
    elapsed = time.time() - (deadline - 105.0)
    max_time = 90.0  # leave 15s margin
    
    while time.time() < deadline - 15.0:
        # Pick two random edges, remove them, reconnect
        e1 = rng.choice(list(edge_set))
        e2 = rng.choice(list(edge_set))
        if e1 == e2:
            continue
        u1, v1 = e1
        u2, v2 = e2
        # Try reconnecting: (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
        for new_e1, new_e2 in [((min(u1,u2),max(u1,u2)), (min(v1,v2),max(v1,v2))),
                                 ((min(u1,v2),max(u1,v2)), (min(v1,u2),max(v1,u2)))]:
            if new_e1 in edge_set or new_e2 in edge_set:
                continue
            if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                continue
            # Apply swap
            edge_set.discard(e1)
            edge_set.discard(e2)
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            
            # Update adjacency
            adj[u1].discard(v1); adj[v1].discard(u1)
            adj[u2].discard(v2); adj[v2].discard(u2)
            adj[new_e1[0]].add(new_e1[1]); adj[new_e1[1]].add(new_e1[0])
            adj[new_e2[0]].add(new_e2[1]); adj[new_e2[1]].add(new_e2[0])
            
            new_girth = _compute_girth(n, list(edge_set))
            if new_girth > best_girth:
                best_girth = new_girth
                best_edges = list(edge_set)
                if best_girth >= 16:
                    return n, best_edges
            
            # Revert
            edge_set.discard(new_e1)
            edge_set.discard(new_e2)
            edge_set.add(e1)
            edge_set.add(e2)
            adj[u1].add(v1); adj[v1].add(u1)
            adj[u2].add(v2); adj[v2].add(u2)
            adj[new_e1[0]].discard(new_e1[1]); adj[new_e1[1]].discard(new_e1[0])
            adj[new_e2[0]].discard(new_e2[1]); adj[new_e2[1]].discard(new_e2[0])
    
    return n, best_edges


def _compute_girth(n, edges):
    """Compute the girth of a graph using BFS from each vertex (non-backtracking)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_cycle = float('inf')
    
    for start in range(n):
        # BFS tracking parent to avoid backtracking
        # dist[v] = (distance, parent)
        dist = [(-1, -1) for _ in range(n)]
        dist[start] = (0, -1)
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            d_u, p_u = dist[u]
            if d_u >= min_cycle:
                continue
            for v in adj[u]:
                if v == p_u:
                    continue
                if dist[v][0] == -1:
                    dist[v] = (d_u + 1, u)
                    queue.append(v)
                elif dist[v][1] != u:  # found a cycle
                    cycle_len = d_u + 1 + dist[v][0]
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
    
    return min_cycle if min_cycle != float('inf') else 2 * n


def _luw_fallback(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction."""
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
# EVOLVE-BLOCK-END
