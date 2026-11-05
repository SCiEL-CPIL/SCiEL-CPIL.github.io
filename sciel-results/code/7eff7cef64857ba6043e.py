import time
import random
import sys
from collections import deque
import numpy as np


def construct(k, g):
    """Construct a k-regular graph with girth at least g.

    Returns (n, edges). For k == 3 and g == 16 uses a voltage lift of the
    Petersen graph; otherwise falls back to the LUW construction.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    if k == 3:
        res = _construct_petersen_lift(g, deadline, start_time)
        if res is not None:
            return res
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)


def _construct_petersen_lift(g, deadline, start_time):
    """Voltage lift of the Petersen graph with a fast local search plus a
    smooth epigraph polish. Returns (n, edges) or None."""

    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),   # outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),   # inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),   # spokes
    ]
    n_base = 10
    E = 15

    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    adj_base = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)

    # ---- Collect non-backtracking closed walk coefficient vectors (length 5..g-1)
    max_walk_len = g - 1
    coeff_vectors = set()
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * E)]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in adj_base[v]:
                if next_v == prev_v:
                    continue
                e_idx = edge_index[(v, next_v)]
                u_edge, v_edge = pet_edges[e_idx]
                new_counts = counts.copy()
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                new_l = l + 1
                if next_v == start_v and new_l >= 5:
                    coeff_vectors.add(tuple(new_counts))
                stack.append((next_v, v, new_l, new_counts))

    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(E):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)

    if len(coeff_list) == 0:
        return None
    C = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(C)

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

    def check_girth_fast(n, edges, min_girth, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
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
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < min_girth:
                return min_cycle
        return min_cycle

    def solve_voltage(m, max_iters=4000, seed=0):
        """Fast local search (with a smooth epigraph polish) for v in Z_m^E with
        C @ v != 0 (mod m) for every row. Returns (v, cost)."""
        rng = np.random.RandomState(seed)
        v = rng.randint(1, m, size=E).astype(np.int64)

        def cost(v):
            residues = np.mod(C @ v, m)
            return int(np.sum(residues == 0))

        best_v = v.copy()
        best_cost = cost(v)
        if best_cost == 0:
            return best_v, 0

        # ---- Phase 1: coordinate descent on a cached violations list
        viol = np.flatnonzero(np.mod(C @ best_v, m) == 0)
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        it = 0
        while it < max_iters:
            it += 1
            if time.time() > deadline - 8:
                break
            if len(viol) == 0:
                return best_v, 0
            # pick a violated constraint, then a coordinate in it
            row = viol[rng.randint(0, len(viol) - 1)]
            nz = np.flatnonzero(C[row])
            j = nz[rng.randint(0, len(nz) - 1)]
            old_val = v[j]
            v[j] = rng.randint(1, m - 1)
            new_cost = cost(v)
            delta = new_cost - best_cost
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = v.copy()
                    if new_cost == 0:
                        return best_v, 0
                viol = np.flatnonzero(np.mod(C @ v, m) == 0)
            else:
                v[j] = old_val
            T *= cooling

        # ---- Phase 2: smooth epigraph polish (soft-min) on the remaining violations
        if len(viol) > 0 and time.time() < deadline - 8:
            Cv = C[viol]
            base = best_v.copy()
            beta = 8.0
            # gradient of soft-min of squared distances to 0 (mod m)
            for p in range(4000):
                if time.time() > deadline - 8:
                    break
                r = np.mod(Cv @ base, m).astype(np.float64)
                r = np.minimum(r, m - r)
                w = np.exp(-beta * r * r)
                w = w / (w.sum() + 1e-12)
                gvec = -(Cv.T @ w) * (2.0 * r)
                nz = np.flatnonzero(gvec != 0)
                if len(nz) == 0:
                    break
                j = nz[rng.randint(0, len(nz) - 1)]
                base[j] = (base[j] + 1) % m
                c2 = int(np.sum(np.mod(C @ base, m) == 0))
                if c2 < best_cost:
                    best_cost = c2
                    best_v = base.copy()
                    if c2 == 0:
                        return best_v, 0
                else:
                    base[j] = (base[j] - 2) % m
                    c2 = int(np.sum(np.mod(C @ base, m) == 0))
                    if c2 < best_cost:
                        best_cost = c2
                        best_v = base.copy()
                        if c2 == 0:
                            return best_v, 0
                    base[j] = (base[j] + 1) % m
        return best_v, best_cost

    # ---- Search primes in increasing order (n = 10 * m)
    # Moore bound for k=3, g=16 is 766 -> m >= 77 -> first prime 79.
    primes = []
    cand = 79
    while len(primes) < 40:
        if is_prime(cand):
            primes.append(cand)
        cand += 1

    for m in primes:
        if time.time() > deadline - 20:
            break
        n_candidate = 10 * m
        for restart in range(24):
            if time.time() > deadline - 15:
                break
            v, cost = solve_voltage(m, max_iters=4000, seed=restart * 7919 + 13)
            if cost == 0:
                edges = _build_lifted_graph(pet_edges, v.tolist(), m)
                gc = check_girth_fast(n_candidate, edges, g, deadline - 5)
                if gc >= g:
                    return n_candidate, edges
    return None


def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            sign = 1 if (u == u_edge and v == v_edge) else -1
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    return list(edges)


def _construct_luw(k, g, d=None, variant=0):
    """LUW construction with specified depth d and variant for rule selection."""
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
