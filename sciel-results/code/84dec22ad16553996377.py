import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- Girth Verification (fast) ----------------
    def check_girth(n, edges, g, time_limit):
        """
        Returns the girth if < g, else g (or larger).
        Uses BFS from every vertex; each edge is checked exactly once
        (when the later-discovered endpoint is processed).
        """
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        min_cycle = float('inf')
        g_minus_1 = g - 1

        for s in range(n):
            if time.time() > time_limit:
                return -1

            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0

            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]

                # If d >= g_minus_1, any cycle through this node has length >= g
                if d >= g_minus_1:
                    continue

                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = d + 1
                        if d + 1 < g_minus_1:
                            queue.append(nb)
                    elif dist[nb] != d + 1:
                        # Found a cycle: dist[node] + dist[nb] + 1
                        cycle_len = d + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle

        return min_cycle if min_cycle != float('inf') else g

    # ---------------- Voltage Lift of Petersen ----------------
    def voltage_lift_petersen():
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

        # Generate all non-backtracking closed walks of length [5, 15]
        max_L = g - 1
        walk_vectors = set()

        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
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
                        walk_vectors.add(tuple(new_counts))

                    stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        # Filter and normalize
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)

        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0:
            return None

        n_constraints = C.shape[0]

        # Precompute which constraint rows are affected by each variable
        var_to_rows = [np.where(C[:, j] != 0)[0] for j in range(15)]

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

        def solve_voltage(m, max_iters=20000):
            """
            Simulated Annealing with vectorized cost deltas.
            Only recomputes affected constraint rows when a variable changes.
            """
            v = np.random.randint(1, m, size=15).astype(np.int64)
            residues = np.mod(C @ v, m).astype(np.int64)
            cost = int(np.sum(residues == 0))

            best_v = v.copy()
            best_cost = cost

            if best_cost == 0:
                return v, 0

            T = 1.0
            cooling = 0.9995
            min_T = 0.01

            for it in range(max_iters):
                if remaining() < 5:
                    break

                j = random.randint(0, 14)
                old_val = v[j]
                new_val = random.randint(1, m - 1)
                if new_val == old_val:
                    continue

                delta_val = new_val - old_val
                rows = var_to_rows[j]

                if len(rows) > 0:
                    old_res = residues[rows]
                    new_res = (old_res + C[rows, j] * delta_val) % m

                    old_zero = (old_res == 0)
                    new_zero = (new_res == 0)

                    delta_cost = int(np.sum(new_zero)) - int(np.sum(old_zero))
                    residues[rows] = new_res
                    v[j] = new_val

                new_cost = cost + delta_cost

                if delta_cost < 0 or (T > min_T and random.random() < np.exp(-delta_cost / T)):
                    cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
                else:
                    # Revert
                    v[j] = old_val
                    if len(rows) > 0:
                        residues[rows] = (residues[rows] - C[rows, j] * delta_val) % m

                T *= cooling

            return best_v, best_cost

        # Try primes in increasing order starting from 53
        m = 53
        best_result = None
        restarts_per_prime = 8

        while m < 500 and remaining() > 20:
            if is_prime(m):
                for restart in range(restarts_per_prime):
                    if remaining() < 10:
                        break

                    v, cost = solve_voltage(m, max_iters=15000)

                    if cost == 0:
                        n = 10 * m
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

                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc >= g:
                            return (n, edges)
                        else:
                            pass

            m += 1

        return None

    res = voltage_lift_petersen()
    if res is not None:
        return res

    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
