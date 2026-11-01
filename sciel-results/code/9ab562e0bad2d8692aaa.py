import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)

    # Voltage graph search for n(5,9)
    # Moore bound is 426, so N >= 426
    # We search for base graphs B (5-regular) and voltage assignments in Z_m
    # such that the lift has girth >= 9

    best_n = float('inf')
    best_edges = None

    # Generate candidate base graphs
    base_graphs = []

    # K6
    edges_k6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    base_graphs.append((6, edges_k6))

    # K5,5
    edges_k55 = [(i, j+5) for i in range(5) for j in range(5)]
    base_graphs.append((10, edges_k55))

    # Random 5-regular graphs
    for V in [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]:
        stubs = [i for i in range(V) for _ in range(5)]
        random.shuffle(stubs)
        edges = set()
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
        if valid and len(edges) == 5 * V // 2:
            base_graphs.append((V, list(edges)))

    base_graphs.sort(key=lambda x: x[0])

    for V, base_edges in base_graphs:
        if time.time() > deadline - 15:
            break

        target_m = 426 // V
        for m in range(max(2, target_m - 5), target_m + 6):
            N = V * m
            if N < 426:
                continue

            # Try structured + random voltage assignments
            for trial in range(30):
                if time.time() > deadline - 5:
                    break

                # Build adjacency for the base graph (directed)
                base_adj = {u: [] for u in range(V)}
                for u, v in base_edges:
                    base_adj[u].append(v)
                    base_adj[v].append(u)

                # Assign voltages
                voltages = {}
                if trial < 5:
                    # Structured: use a cyclic pattern
                    for u, v in base_edges:
                        w = (u * 7 + v * 13) % m
                        voltages[(u, v)] = w
                        voltages[(v, u)] = (m - w) % m
                else:
                    # Random
                    for u, v in base_edges:
                        w = random.randint(0, m-1)
                        voltages[(u, v)] = w
                        voltages[(v, u)] = (m - w) % m

                # Build lift adjacency
                lift_adj = [[] for _ in range(N)]
                for u, v in base_edges:
                    w = voltages[(u, v)]
                    for i in range(m):
                        u_idx = u * m + i
                        v_idx = v * m + (i + w) % m
                        lift_adj[u_idx].append(v_idx)
                        lift_adj[v_idx].append(u_idx)

                # Check girth using non-backtracking walks
                if _check_girth_nb(lift_adj, N, g):
                    edges = []
                    for u in range(N):
                        for v in lift_adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if N < best_n:
                        best_n = N
                        best_edges = edges
                        if N <= 430:
                            return N, edges

    if best_edges is not None:
        return best_n, best_edges

    return _luw_construct(k, g, deadline)


def _check_girth_nb(adj, n, min_girth):
    """
    Check girth >= min_girth using non-backtracking walks.
    A non-backtracking walk of length L starting at vertex s with first step to v
    is a sequence s, v, w1, ..., w_{L-1} where no step immediately backtracks.
    If there's a closed non-backtracking walk of length < min_girth, girth is violated.
    We check for cycles of length 3 through min_girth-1.
    """
    # For each vertex, do a BFS on the state space (current_vertex, previous_vertex)
    # to find the shortest non-backtracking cycle through that vertex.
    # But this is expensive. Instead, use the fact that for girth >= 9,
    # we need to check for cycles of length 3,4,5,6,7,8.
    
    # Efficient approach: for each vertex s, explore non-backtracking walks
    # up to length (min_girth-1)//2 and check if they return to s.
    # Actually, a cycle of length L through s means a non-backtracking walk
    # of length L starting and ending at s.
    
    # For small girth checks, we can do:
    # For length 3: for each edge (u,v), check if u and v share a common neighbor
    # For length 4: for each pair of vertices at distance 2, check...
    
    # Let's use a direct non-backtracking walk enumeration for lengths 3 to min_girth-1
    # State: (current, previous) where previous is the vertex we came from
    # We want to find if there's a walk of length L that starts at s, 
    # goes through non-backtracking steps, and returns to s.
    
    # More efficient: for each starting vertex s, 
    # do a BFS in the "line graph" sense:
    # State is (v, p) meaning we're at v, came from p
    # Start states: (v, s) for each neighbor v of s
    # Transition: from (v, p), go to (w, v) for each neighbor w of v where w != p
    # A cycle of length L through s exists if we can reach (s, v) from (v, s) in L-1 steps
    
    # But we need the TOTAL walk length to be L.
    # Walk: s -> v -> w1 -> ... -> s has length L
    # In state space: start at (v, s), after L-2 transitions we're at (s, u)
    # where u is the second-to-last vertex, and the last step goes s->u... 
    # Wait, let me think again.
    
    # Walk of length L: v_0=s, v_1, ..., v_L=s
    # Non-backtracking: v_{i-1} != v_{i+1} for all i
    # State after step 1: (v_1, v_0) = (v_1, s)
    # State after step i: (v_i, v_{i-1})
    # We want v_L = s, so state after step L is (s, v_{L-1})
    # So we need to reach a state (s, u) from initial state (v, s) in L-1 steps
    # where u is a neighbor of s and u != v (non-backtracking on the last step)
    
    # For each s, for each neighbor v of s:
    #   BFS from (v, s) in the non-backtracking state space
    #   Check if we can reach (s, u) for any neighbor u of s, u != v, in exactly L-1 steps
    #   for L = 3, 4, ..., min_girth-1
    
    # This is still expensive. Let's use a simpler approach for girth up to 8:
    # Count triangles, 4-cycles, etc. using adjacency matrix multiplication.
    
    # For now, use a practical approach:
    # Check for cycles of length 3 to min_girth-1 by BFS with parent tracking
    # but optimized: only need to check up to depth (min_girth-1)//2 from each vertex
    # and look for back-edges.
    
    # Actually, the simplest correct approach for small graphs:
    # For each vertex s, do BFS. When we find a vertex v already visited,
    # the cycle length is dist[s->u] + dist[s->v] + 1 where u is parent of v... 
    # No, that's not right either.
    
    # Let me just use the non-backtracking walk approach but limit depth.
    # For girth 9, we need to rule out cycles of length 3-8.
    # Max depth we need to explore: 4 (since a cycle of length 8 through s 
    # means going out 4 steps and back 4 steps, but non-backtracking...)
    
    # Simpler: use adjacency matrix and check (A^L)[s,s] for L=3..8
    # But that counts walks, not necessarily simple cycles.
    # However, for girth checking, if there's ANY closed walk of length L < 9,
    # there's a cycle of length <= L < 9. So checking closed walks suffices!
    
    # A closed walk of length L implies a cycle of length <= L.
    # So we just need to check if there's a closed non-backtracking walk of length < 9.
    # Actually, even a closed walk (with backtracking) of length L implies a cycle <= L.
    # So: if (A^L)[s][s] > 0 for any s and L < 9, girth < 9.
    
    # But (A^L)[s][s] counts ALL closed walks of length L, including those with backtracking.
    # A closed walk of length 2 is just going s->v->s, which is a "cycle" of length 2 (not valid).
    # A closed walk of length 3 is a triangle.
    # A closed walk of length 4 could be s->a->b->a->s (backtracking) which contains a 2-cycle... 
    # No, s->a->b->a->s: the edges are (s,a),(a,b),(b,a),(a,s). The underlying cycle is s-a-b-a-s which isn't simple.
    # But it does contain the edge (a,b) traversed twice, so the girth is at most... well, 
    # if there's a closed walk of length L, there's a cycle of length <= L.
    # Proof: take the shortest cycle in the walk.
    
    # So: girth >= g iff there is no closed walk of length L for any 3 <= L < g.
    # (Length 2 closed walks are just back-and-forth on an edge, not cycles.)
    
    # We can compute A^L for L=3..8 and check diagonal entries.
    # For n=426, A is 426x426. A^2 is O(n^3) = 77M ops, feasible.
    # But we need A^3 through A^8, which is 6 matrix multiplications.
    # That's 6 * 426^3 ≈ 463M ops. With numpy, should be fast.
    
    # Actually, let's use a different approach: 
    # For each vertex s, count the number of closed walks of length L starting/ending at s
    # for L = 3, 4, 5, 6, 7, 8.
    # If any count > 0, girth < 9.
    
    # Use iterative multiplication:
    # W_1 = A
    # W_L = W_{L-1} * A
    # Check diag(W_L) for L=3..8
    
    import numpy as np
    
    # Convert to numpy
    A = np.zeros((n, n), dtype=np.int64)
    for u in range(n):
        for v in adj[u]:
            A[u, v] = 1
    
    # Compute A^L for L=3..8
    W = A.copy()  # A^1
    W2 = W @ A    # A^2
    for L in range(3, min_girth):
        W = W2 @ A  # A^L
        W2 = W
        # Check diagonal
        if np.any(np.diag(W) > 0):
            return False
    
    return True


def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges
