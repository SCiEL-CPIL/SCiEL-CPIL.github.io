import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # Moore bound
    if k == 3:
        if g % 2 == 0:
            moore_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
        else:
            moore_bound = (k - 1) * sum((k - 1) ** i for i in range((g - 1) // 2))
    else:
        moore_bound = 1

    def check_girth_fast(adj, n, g):
        """Exact girth check using BFS from each vertex."""
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start and dist[u] + dist[v] + 1 < g:
                        return False
        return True

    def build_petersen_voltage_lift(q, voltage_seed):
        """
        Build a 3-regular graph by voltage-lifting the Petersen graph over Z_q.
        The Petersen graph has 10 vertices and 15 edges, girth 5.
        A voltage lift over Z_q gives 10*q vertices.
        The girth of the lift depends on the voltage assignment.
        """
        rng = random.Random(voltage_seed)

        # Petersen graph edges (standard labeling)
        # Outer pentagon: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]

        # Assign random voltages to each edge
        voltages = {}
        for u, v in pet_edges:
            voltages[(u, v)] = rng.randint(0, q - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % q

        # Build the lift: vertices are (base_vertex, voltage)
        # Total vertices: 10 * q
        n = 10 * q
        adj = [[] for _ in range(n)]

        for u, v in pet_edges:
            for a in range(q):
                b = (a + voltages[(u, v)]) % q
                # Vertex (u, a) connects to (v, b)
                idx_u = u * q + a
                idx_v = v * q + b
                adj[idx_u].append(idx_v)

        # Add reverse edges (make undirected)
        for u, v in pet_edges:
            for a in range(q):
                b = (a + voltages[(u, v)]) % q
                idx_u = u * q + a
                idx_v = v * q + b
                adj[idx_v].append(idx_u)

        # Verify 3-regular and simple
        for i in range(n):
            if len(adj[i]) != 6:  # 3 edges * 2 (undirected)
                return None
            if len(set(adj[i])) != 6:
                return None

        # Convert to proper adjacency (each undirected edge appears once per endpoint)
        # Actually the above already has each edge in both directions.
        # Let's rebuild properly.
        adj2 = [[] for _ in range(n)]
        for u, v in pet_edges:
            for a in range(q):
                b = (a + voltages[(u, v)]) % q
                idx_u = u * q + a
                idx_v = v * q + b
                adj2[idx_u].append(idx_v)
                adj2[idx_v].append(idx_u)

        # Verify
        for i in range(n):
            if len(adj2[i]) != 3:
                return None
            if len(set(adj2[i])) != 3:
                return None

        return adj2

    def build_luw_fallback(d):
        """LUW construction fallback."""
        q = 3
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

        rule = {}
        if d >= 2: rule[2] = (1, 1)
        if d >= 3: rule[3] = (2, 1)
        if d >= 4: rule[4] = (1, 2)
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

    best_n = float('inf')
    best_edges = None

    # Strategy: Voltage-lift Petersen graph over Z_q for various q values.
    # The lift has 10*q vertices. For girth 16, we need q large enough.
    # Try q from small to large, with multiple random voltage assignments.
    # Also try q values that give n near the Moore bound (254) and above.

    # Try various q values
    q_candidates = []
    for q in range(20, 200):
        if 10 * q >= moore_bound:
            q_candidates.append(q)

    # Sort by resulting n (which is 10*q)
    q_candidates.sort()

    # For each q, try several random voltage assignments
    for q in q_candidates:
        if time.time() > deadline - 15:
            break

        n = 10 * q
        # Try multiple seeds for this q
        max_seeds = 50 if n < 500 else 20 if n < 800 else 10
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break

            adj = build_petersen_voltage_lift(q, seed)
            if adj is None:
                continue

            if check_girth_fast(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_n = n
                best_edges = edges
                break  # Found for this q, try smaller q? No, we go in increasing order.

        # If we found one, we can stop since we're going in increasing n order
        if best_edges is not None:
            break

    # If voltage lift didn't work, try larger q with more attempts
    if best_edges is None:
        for q in range(100, 300):
            if time.time() > deadline - 5:
                break
            n = 10 * q
            for seed in range(30):
                if time.time() > deadline - 3:
                    break
                adj = build_petersen_voltage_lift(q, seed)
                if adj is None:
                    continue
                if check_girth_fast(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    break
            if best_edges is not None:
                break

    # Fallback: LUW construction
    if best_edges is None:
        best_n, best_edges = build_luw_fallback(11)

    return best_n, best_edges
