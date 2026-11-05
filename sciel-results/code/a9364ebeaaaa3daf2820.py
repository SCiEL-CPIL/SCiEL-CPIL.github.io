import random
import time
import math
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # LUW fallback construction
    def _luw_graph(k, g):
        q = k
        d = max(3, (g - 2) // 2)
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

    # Verify girth using non-backtracking BFS from every vertex (optimized)
    def _verify_girth(n, edges, girth_req):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Use BFS with early termination: if we find a cycle of length < girth_req, fail
        # Optimize by checking degree first, then using efficient BFS
        for start in range(n):
            # Non-backtracking BFS: state is (current_vertex, prev_vertex, distance)
            # We track visited states as (vertex, prev) pairs
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            while queue:
                curr, prev, dist = queue.popleft()
                if dist >= girth_req:
                    continue
                if curr == start and dist >= 3:
                    return False
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nb, curr, dist + 1))
        return True

    # Petersen graph: 10 vertices, 15 edges
    # Outer cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle (0-4)
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star (5-9)
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes (10-14)
    ]
    n_base = 10
    n_edges_base = len(base_edges)
    
    # Build adjacency with edge indices
    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(base_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    # Generate all closed non-backtracking walks of length 3..g-1 in Petersen
    # Represent each walk as a list of (edge_id, sign) where sign=+1 if traversed u->v, -1 if v->u
    def _get_closed_nb_walks(max_len):
        walks = set()
        for start_v in range(n_base):
            # BFS on (current_vertex, prev_edge, length)
            stack = []
            for next_v, e_id in base_adj[start_v]:
                u, v = base_edges[e_id]
                sign = 1 if u == start_v else -1
                stack.append((next_v, e_id, 1, [(e_id, sign)]))
            while stack:
                curr_v, prev_e, length, path = stack.pop()
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    # Store as tuple for set membership
                    walks.add(tuple(path))
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = base_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_path = path + [(e_id, sign)]
                    stack.append((next_v, e_id, length + 1, new_path))
        return list(walks)

    closed_walks = _get_closed_nb_walks(g)

    # Precompute sparse coefficient vectors for each walk
    # For each walk, compute the coefficient of each edge (sum of signs)
    walk_coeffs = []
    for w in closed_walks:
        coeff = [0] * n_edges_base
        for e_id, sign in w:
            coeff[e_id] += sign
        # Only keep non-zero coefficients
        sparse = [(eid, c) for eid, c in enumerate(coeff) if c != 0]
        if sparse:
            walk_coeffs.append(sparse)

    # Remove duplicate walks (same coefficient vector)
    seen = set()
    unique_walk_coeffs = []
    for sparse in walk_coeffs:
        key = tuple(sorted(sparse))
        if key not in seen:
            seen.add(key)
            unique_walk_coeffs.append(sparse)
    walk_coeffs = unique_walk_coeffs

    best_n = float('inf')
    best_edges = None

    # Search for voltage assignments
    # Moore bound for k=3, g=16 is 512, so n = 10*m >= 512 => m >= 52
    # Try smaller m first to find the smallest valid graph
    m_candidates = list(range(52, 100)) + list(range(100, 200, 2)) + list(range(200, 400, 5))

    random.seed(42)

    # Precompute walk constraint matrix for faster checking
    # walk_coeffs is list of lists of (edge_id, coefficient)
    # Convert to numpy for vectorized dot products
    n_walks = len(walk_coeffs)
    n_edges = n_edges_base
    # Build constraint matrix: each row is a walk, columns are edge indices
    # Use sparse representation for efficiency
    walk_eids = np.zeros((n_walks, n_edges), dtype=np.int32)
    walk_coefs = np.zeros((n_walks, n_edges), dtype=np.int32)
    for i, sparse in enumerate(walk_coeffs):
        for eid, c in sparse:
            walk_eids[i, eid] = 1
            walk_coefs[i, eid] = c

    for m in m_candidates:
        if time.time() > deadline - 20.0:
            break
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue

        # Try multiple voltage assignments
        # Use a mix of systematic and random approaches
        max_tries = 30000 if m < 60 else 15000
        found = False

        # First try structured voltages: use small prime-like values
        # Then fall back to random
        tried_voltages = set()

        for attempt in range(max_tries):
            if time.time() > deadline - 10.0:
                break

            if attempt < 500:
                # Structured: use values that are relatively prime to m
                # Try combinations of small generators
                if m % 2 == 1:
                    # m is odd, use even numbers
                    base_vals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
                else:
                    # m is even, use odd numbers
                    base_vals = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
                # Pick from base_vals with some randomness
                volts = [base_vals[random.randint(0, len(base_vals) - 1)] % m for _ in range(n_edges)]
                # Ensure at least one value is coprime to m
                g_val = m
                for v in volts:
                    g_val = math.gcd(g_val, v)
                    if g_val == 1:
                        break
                if g_val != 1:
                    # Replace one value with a known generator
                    for gen in [1, 2, 3, 5, 7, 11, 13]:
                        if math.gcd(gen, m) == 1:
                            volts[random.randint(0, n_edges - 1)] = gen
                            break
            else:
                # Random voltages
                volts = [random.randint(1, m - 1) for _ in range(n_edges)]
                g_val = m
                for v in volts:
                    g_val = math.gcd(g_val, v)
                    if g_val == 1:
                        break
                if g_val != 1:
                    continue

            # Check all walk constraints using vectorized operations
            volts_arr = np.array(volts, dtype=np.int64)
            # Compute sum of c * v for each walk
            sums = np.sum(walk_coefs * volts_arr[None, :], axis=1)
            # Check if any sum is 0 mod m
            if not np.any(sums % m == 0):
                # Valid! Construct the lift
                lift_edges = []
                for i, (u, v) in enumerate(base_edges):
                    vol = volts[i]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + vol) % m
                        lift_edges.append((u_lift, v_lift))

                # Verify girth before accepting
                if _verify_girth(candidate_n, lift_edges, g):
                    best_n = candidate_n
                    best_edges = lift_edges
                    found = True
                    break

        if found:
            break

    if best_edges is not None:
        return best_n, best_edges

    # Fallback to LUW
    return _luw_graph(k, g)
