import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # Check degree
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

    # ---------------- Voltage lift on Petersen graph ----------------
    def voltage_lift_search():
        if k != 3 or g != 16:
            return None

        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]

        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate constraints: all non-backtracking closed walks of length < g
        # on the base graph. For each such walk, the sum of voltages (with signs)
        # must be non-zero mod p.
        max_L = g - 1
        walk_vectors = set()

        for start_v in range(n_base):
            # DFS for non-backtracking closed walks
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()

                if length >= max_L:
                    continue

                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1

                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1

                    if next_v == start_v and new_l >= 5:
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            # Normalize: make first non-zero component positive
                            first_nz = -1
                            for i in range(15):
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
        num_constraints = C.shape[0]

        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        def solve_voltage(m, time_limit, rng):
            """
            Find voltage assignment v in (Z/mZ)^15 such that C @ v has no zero
            entries mod m. Use simulated annealing with a smooth surrogate.
            """
            v = rng.randint(1, m, size=15).astype(np.int64)

            def cost_exact(v):
                residues = np.mod(C @ v, m)
                return int(np.sum(residues == 0))

            best_v = v.copy()
            best_cost = cost_exact(v)

            if best_cost == 0:
                return best_v

            # Simulated annealing
            T = 2.0
            cooling = 0.9995
            start_t = time.time()
            no_improve = 0

            while time.time() - start_t < time_limit:
                if remaining() < 5:
                    break

                # Random perturbation: change 1-3 components
                num_changes = rng.randint(1, 3)
                old_vals = []
                for _ in range(num_changes):
                    j = rng.randint(0, 14)
                    old_vals.append((j, v[j]))
                    v[j] = rng.randint(1, m - 1)

                new_cost = cost_exact(v)
                delta = new_cost - best_cost

                if delta < 0 or (delta == 0) or (rng.random() < np.exp(-delta / max(T, 0.01))):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                        no_improve = 0
                        if best_cost == 0:
                            return best_v
                    else:
                        no_improve += 1
                else:
                    # Revert
                    for j, old_val in old_vals:
                        v[j] = old_val
                    no_improve += 1

                T *= cooling

                # If stuck for too long, try a bigger jump
                if no_improve > 5000:
                    # Restart with a fresh random assignment but keep best
                    for j in range(15):
                        v[j] = rng.randint(1, m - 1)
                    no_improve = 0
                    T = max(T, 1.0)

            return best_v if best_cost == 0 else None

        def try_prime(m, time_budget, rng):
            """Try to solve for prime m."""
            # Multiple restarts
            max_restarts = 5
            for r in range(max_restarts):
                if remaining() < 3:
                    return None
                t_alloc = time_budget / max_restarts
                v = solve_voltage(m, t_alloc, rng)
                if v is not None:
                    return v
            return None

        # Moore bound for k=3, g=16: 2 + 3*(2^8 - 1) = 766
        # n = 10 * m, so m >= 77. Smallest prime >= 77 is 79.
        # Try primes in increasing order.
        primes_to_try = []
        m = 79
        while m < 200:
            if is_prime(m):
                primes_to_try.append(m)
            m += 1

        for m in primes_to_try:
            if remaining() < 15:
                break

            n = 10 * m
            # Allocate more time to smaller m (more likely to work)
            t_budget = min(12.0, remaining() * 0.3)

            rng = random.Random(m * 7919 + 31)
            v = try_prime(m, t_budget, rng)

            if v is not None:
                # Construct the lifted graph
                edges = []
                for u, v_base in pet_edges:
                    e_idx = edge_index[(u, v_base)]
                    vol = int(v[e_idx])
                    for layer in range(m):
                        u_lift = u * m + layer
                        v_lift = v_base * m + (layer + vol) % m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))

                # Verify
                gc = check_girth(n, edges, g, deadline - 2)
                if gc >= g:
                    return (n, edges)

        return None

    # ---------------- Cayley graph / Schreier graph approach ----------------
    def cayley_graph_search():
        if k != 3 or g != 16:
            return None

        # For n vertices, we need 3 involutions a, b, c in S_n
        # such that the Schreier graph has girth >= 16.
        # n must be even for 3-regular graph.
        # Moore bound: 766, so try n from 768 upward.

        def random_involution(n, rng):
            """Generate a random fixed-point-free involution."""
            if n % 2 != 0:
                return None
            perm = list(range(n))
            # Randomly pair up elements
            unpaired = list(range(n))
            rng.shuffle(unpaired)
            for i in range(0, n, 2):
                a, b = unpaired[i], unpaired[i+1]
                perm[a] = b
                perm[b] = a
            return perm

        def check_schreier_girth(n, gens, g, time_limit):
            """Check that the Schreier graph has girth >= g."""
            adj = [[] for _ in range(n)]
            for gen in gens:
                for i in range(n):
                    j = gen[i]
                    if i != j:
                        adj[i].append(j)

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
            return min_cycle

        def build_from_generators(n, gens):
            edges = set()
            for gen in gens:
                for i in range(n):
                    j = gen[i]
                    if i < j:
                        edges.add((i, j))
            return list(edges)

        # Try a few n values
        for n_try in [768, 770, 772, 774, 776, 778, 780]:
            if remaining() < 20:
                break

            rng = random.Random(n_try * 99991 + 7)
            max_attempts = 30
            for attempt in range(max_attempts):
                if remaining() < 10:
                    break

                gens = []
                valid = True
                for _ in range(3):
                    gen = random_involution(n_try, rng)
                    if gen is None:
                        valid = False
                        break
                    gens.append(gen)

                if not valid:
                    continue

                # Quick connectivity/girth check
                gc = check_schreier_girth(n_try, gens, g, deadline - 5)
                if gc >= g:
                    edges = build_from_generators(n_try, gens)
                    gc2 = check_girth(n_try, edges, g, deadline - 2)
                    if gc2 >= g:
                        return (n_try, edges)

        return None

    # Try voltage lift first (most promising for small n)
    res = voltage_lift_search()
    if res is not None:
        return res

    # Fallback: Cayley/Schreier graph approach
    res2 = cayley_graph_search()
    if res2 is not None:
        return res2

    return (0, [])


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
