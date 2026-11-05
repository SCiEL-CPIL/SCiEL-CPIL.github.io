import time
import numpy as np
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (non-backtracking BFS from every vertex) ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return -1
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift with constraint-directed SA ----------------
    def voltage_lift(base_edges, base_name):
        if k != 3:
            return None

        n_base = max(max(u, v) for u, v in base_edges) + 1
        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(n_base)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Verify base graph is 3-regular
        for i in range(n_base):
            if len(adj[i]) != 3:
                return None

        n_edges = len(base_edges)

        # Generate all non-backtracking closed walks of length [5, g-1]
        max_L = g - 1
        walk_vectors = set()
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * n_edges)]
            while stack:
                current, prev, length, counts = stack.pop()
                if length >= max_L:
                    continue
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = base_edges[e_idx]
                    d = 1 if current == u else -1
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    if next_v == start_v and new_l >= 5:
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(n_edges):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and vec[first_nz] < 0:
                                vec = tuple(-x for x in vec)
                            walk_vectors.add(vec)
                    stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        C = np.array(list(walk_vectors), dtype=np.int64)
        if C.size == 0:
            return None

        n_constraints = C.shape[0]
        constraint_nonzero_indices = [np.nonzero(C[i])[0] for i in range(n_constraints)]

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        def solve_voltage(m, max_iters, seed):
            rng = random.Random(seed)
            rs = np.random.RandomState(seed)
            v = rs.randint(1, m, size=n_edges).astype(np.int64)

            def cost(vec):
                residues = np.mod(C @ vec, m)
                return int(np.sum(residues == 0))

            best_v = v.copy()
            best_cost = cost(v)
            if best_cost == 0:
                return v, 0

            T = 1.0
            cooling = 0.9998
            min_T = 0.0003
            for it in range(max_iters):
                if remaining() < 4:
                    break
                if best_cost > 0:
                    residues = np.mod(C @ v, m)
                    violated_idx = np.nonzero(residues == 0)[0]
                    if len(violated_idx) == 0:
                        break
                    c_idx = violated_idx[rng.randrange(len(violated_idx))]
                    nz_indices = constraint_nonzero_indices[c_idx]
                    if len(nz_indices) == 0:
                        continue
                    j = nz_indices[rng.randrange(len(nz_indices))]
                else:
                    j = rng.randrange(n_edges)
                old_val = v[j]
                v[j] = rng.randrange(1, m)
                new_cost = cost(v)
                delta = new_cost - best_cost
                if delta < 0 or (T > min_T and rng.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
                else:
                    v[j] = old_val
                T *= cooling
            return best_v, best_cost

        def polish(v, m, max_iters, seed):
            """Local search polish: try all single-edge changes and 2-edge swaps."""
            rng = random.Random(seed)
            v = v.copy()

            def cost(vec):
                residues = np.mod(C @ vec, m)
                return int(np.sum(residues == 0))

            best_cost = cost(v)
            if best_cost == 0:
                return v, 0

            improved = True
            while improved and max_iters > 0 and remaining() > 5:
                improved = False
                for j in range(n_edges):
                    for new_val in range(1, m):
                        if new_val == v[j]:
                            continue
                        old = v[j]
                        v[j] = new_val
                        nc = cost(v)
                        if nc < best_cost:
                            best_cost = nc
                            improved = True
                            if best_cost == 0:
                                return v, 0
                        else:
                            v[j] = old
                        max_iters -= 1
                        if max_iters <= 0:
                            break
                    if max_iters <= 0:
                        break
            return v, best_cost

        def build_and_verify(v, m):
            n = n_base * m
            edges = []
            for u, v_base in base_edges:
                e_idx = edge_index[(u, v_base)]
                vol = int(v[e_idx])
                for layer in range(m):
                    u_lift = u * m + layer
                    v_lift = v_base * m + (layer + vol) % m
                    if u_lift < v_lift:
                        edges.append((u_lift, v_lift))
                    else:
                        edges.append((v_lift, u_lift))
            gc = check_girth(n, edges, g, deadline - 3)
            if gc >= g:
                return (n, edges)
            return None

        # Moore bound for (3,16) is 766
        # n = n_base * m, so m >= ceil(766 / n_base)
        m_floor = max(2, -(-766 // n_base))  # ceil division
        m = m_floor
        tried = 0
        while m < 300 and remaining() > 15 and tried < 50:
            # Try all m values, but prioritize primes
            if is_prime(m) or m >= m_floor + 10:
                for restart in range(6):
                    if remaining() < 10:
                        break
                    v, c = solve_voltage(m, max_iters=8000, seed=restart * 1000 + m * 7 + n_base)
                    if c > 0:
                        v, c = polish(v, m, max_iters=5000, seed=restart * 5000 + m)
                    if c == 0:
                        res = build_and_verify(v, m)
                        if res is not None:
                            return res
            m += 1
            tried += 1
        return None

    # ---------------- Base graphs ----------------
    # Möbius-Kantor graph: 16 vertices, 24 edges, girth 7, 3-regular
    # It's the voltage lift of the 8-cycle with Z_2 voltages, or equivalently
    # the cubical graph. Let me construct it properly.
    # Möbius-Kantor: vertices 0..15. Edges form two 8-cycles with a twist.
    # Actually, the Möbius-Kantor graph is the (3,7)-cage. Let me list its edges.
    # It can be constructed as follows: take two copies of C8 and connect them.
    # Standard construction: vertices (i, j) where i in Z_8, j in Z_2.
    # Edges: (i,j)-(i+1,j) [cyclic in i], (i,j)-(i+1,1-j) [twisted connection]
    # Wait, that gives degree 3 but let me verify girth.
    # Actually the Möbius-Kantor graph edges:
    # Let me use a known edge list.
    mobius_kantor_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),  # top 8-cycle
        (8, 10), (10, 12), (12, 14), (14, 15), (15, 11), (11, 9), (9, 13), (13, 8),  # bottom 8-cycle (twisted)
        (0, 8), (1, 11), (2, 9), (3, 13), (4, 15), (5, 14), (6, 12), (7, 10)  # connections
    ]

    # Heawood graph: 14 vertices, 21 edges, girth 6, 3-regular
    # It's the (3,6)-cage. Constructed as voltage lift of K4 with Z_2.
    # Standard: vertices (i,j) for i in Z_7, j in Z_2.
    # Edges: (i,j)-(i+1,j), (i,j)-(i+2,j), (i,j)-(i,1-j)
    heawood_edges = []
    for i in range(7):
        for j in range(2):
            u = i * 2 + j
            v1 = ((i + 1) % 7) * 2 + j
            v2 = ((i + 2) % 7) * 2 + j
            v3 = i * 2 + (1 - j)
            heawood_edges.append((u, v1))
            heawood_edges.append((u, v2))
            heawood_edges.append((u, v3))
    # Remove duplicates
    heawood_edges = list(set(tuple(sorted(e)) for e in heawood_edges))
    heawood_edges = [tuple(e) for e in heawood_edges]

    # Petersen graph: 10 vertices, 15 edges, girth 5, 3-regular
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    # Try base graphs in order of expected efficiency
    # Möbius-Kantor first (girth 7, 16 vertices)
    best = None
    r = voltage_lift(mobius_kantor_edges, "mobius_kantor")
    if r is not None:
        best = r

    if remaining() > 10:
        r = voltage_lift(heawood_edges, "heawood")
        if r is not None:
            if best is None or r[0] < best[0]:
                best = r

    if remaining() > 10:
        r = voltage_lift(petersen_edges, "petersen")
        if r is not None:
            if best is None or r[0] < best[0]:
                best = r

    if best is not None:
        return best
    return (0, [])


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
