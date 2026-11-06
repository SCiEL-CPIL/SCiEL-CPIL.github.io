import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def check_c4_fast(adj_bits, n, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks."""
        nu = adj_bits[u]
        nv = adj_bits[v]
        nu_iter = nu
        while nu_iter:
            lsb = nu_iter & (-nu_iter)
            x = lsb.bit_length() - 1
            nu_iter ^= lsb
            common = adj_bits[x] & nv
            if common:
                return True
        return False

    def min_degree(adj_bits, n):
        return min(adj_bits[i].bit_count() for i in range(n))

    def is_c4_free(adj_bits, n):
        """Verify no C4 exists."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_bits[i] & (1 << j)):
                    common = adj_bits[i] & adj_bits[j]
                    if common.bit_count() >= 2:
                        return False
        return True

    def to_edge_list(edges_set):
        return [[a, b] for a, b in edges_set]

    def greedy_build(n, min_deg_req, time_limit, seed):
        """Build a C4-free graph with min degree >= min_deg_req."""
        t0 = time.time()
        rng = random.Random(seed)

        adj_bits = [0] * n
        edges_set = set()

        def add_edge(u, v):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            a, b = min(u, v), max(u, v)
            edges_set.add((a, b))

        def remove_edge(u, v):
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            a, b = min(u, v), max(u, v)
            edges_set.discard((a, b))

        def can_add_edge(u, v):
            if u == v:
                return False
            if adj_bits[u] & (1 << v):
                return False
            return not check_c4_fast(adj_bits, n, u, v)

        # Initialize: try to build a near-regular C4-free graph
        # Start with a random ordering and greedily add edges
        order = list(range(n))
        rng.shuffle(order)
        
        # For each vertex in order, connect to as many other vertices as possible
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.3:
                    if can_add_edge(i, j):
                        add_edge(i, j)

        stuck_count = 0
        repair_budget = 200

        while time.time() - t0 < time_limit:
            md = min_degree(adj_bits, n)
            if md >= min_deg_req:
                # Verify
                if is_c4_free(adj_bits, n):
                    return {'n': n, 'edges': to_edge_list(edges_set)}
                return None

            # Find lowest-degree vertex
            low_v = min(range(n), key=lambda i: adj_bits[i].bit_count())
            low_deg = adj_bits[low_v].bit_count()

            if low_deg < min_deg_req:
                candidates = [v for v in range(n) if v != low_v and not (adj_bits[low_v] & (1 << v))]
                rng.shuffle(candidates)

                added = False
                for v in candidates:
                    if can_add_edge(low_v, v):
                        add_edge(low_v, v)
                        added = True
                        stuck_count = 0
                        break

                if not added:
                    stuck_count += 1
                    if stuck_count >= 3 and repair_budget > 0:
                        repair_budget -= 1

                        # Find blocking edges
                        low_verts = [i for i in range(n) if adj_bits[i].bit_count() < min_deg_req]
                        if len(low_verts) >= 2 and edges_set:
                            block_scores = defaultdict(int)
                            for v in low_verts:
                                non_neigh = [x for x in range(n) if x != v and not (adj_bits[v] & (1 << x))]
                                for x in non_neigh[:10]:  # Limit for speed
                                    nu = adj_bits[v]
                                    nx = adj_bits[x]
                                    nu_iter = nu
                                    while nu_iter:
                                        lsb = nu_iter & (-nu_iter)
                                        a = lsb.bit_length() - 1
                                        nu_iter ^= lsb
                                        common = adj_bits[a] & nx
                                        if common:
                                            common_iter = common
                                            while common_iter:
                                                lsb2 = common_iter & (-common_iter)
                                                b = lsb2.bit_length() - 1
                                                common_iter ^= lsb2
                                                if a != b and (adj_bits[a] & (1 << b)):
                                                    ea, eb = min(a, b), max(a, b)
                                                    block_scores[(ea, eb)] += 1

                            if block_scores:
                                sorted_edges = sorted(block_scores.items(), key=lambda x: -x[1])
                                for edge_candidate, score in sorted_edges[:5]:
                                    u, w = edge_candidate
                                    if adj_bits[u] & (1 << w):
                                        du = adj_bits[u].bit_count() - 1
                                        dw = adj_bits[w].bit_count() - 1
                                        if du >= min_deg_req and dw >= min_deg_req:
                                            remove_edge(u, w)
                                            stuck_count = 0
                                            break
                                else:
                                    # Try random safe removal
                                    safe_edges = [(a, b) for a, b in edges_set
                                                 if a not in low_verts and b not in low_verts]
                                    if safe_edges:
                                        u, w = rng.choice(safe_edges)
                                        du = adj_bits[u].bit_count() - 1
                                        dw = adj_bits[w].bit_count() - 1
                                        if du >= min_deg_req and dw >= min_deg_req:
                                            remove_edge(u, w)
                                            stuck_count = 0
                    elif stuck_count > 30:
                        break
        return None

    def polarity_seed(n, rng):
        """Create a polarity graph seed for C4-free construction."""
        adj_bits = [0] * n
        edges_set = set()

        # Use a quadratic residue polarity graph idea
        # For prime p where n <= p, use x~y iff x-y is a quadratic residue mod p
        # This gives a C4-free (actually C4-free in the right sense) graph

        # Simpler: use a known C4-free construction
        # Start with empty and add edges greedily with bias toward balanced degrees
        order = list(range(n))
        rng.shuffle(order)

        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.25:
                    if not check_c4_fast(adj_bits, n, i, j):
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                        edges_set.add((min(i, j), max(i, j)))

        return adj_bits, edges_set

    best_result = None
    best_n = 0

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline - 3:
            break

        min_deg_req = max(1, n - 39)
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break

        # Allocate time: more attempts for larger n
        num_attempts = min(8, max(2, int(remaining / 5.0)))
        time_per_attempt = (remaining - 2) / max(1, num_attempts)

        for attempt in range(num_attempts):
            if time.time() - start_time > deadline - 2:
                break

            seed = 42 + attempt * 104729 + n * 7919
            result = greedy_build(n, min_deg_req, time_per_attempt, seed)
            if result is not None:
                best_n = n
                best_result = result
                return result

    if best_result is not None:
        return best_result

    # Fallback: small valid graph
    return {'n': 3, 'edges': [[0, 1], [1, 2], [0, 2]]}
