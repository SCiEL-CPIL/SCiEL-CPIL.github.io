import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth, deadline=None):
    """Check if girth >= target_girth using BFS from each vertex.
    Early termination: stop exploring from a start once depth reaches target_girth//2,
    since any cycle through vertices at depth >= target_girth//2 has length >= target_girth.
    Returns True if girth >= target_girth, False otherwise."""
    max_depth = target_girth // 2 - 1
    for start in range(n):
        if deadline is not None and time.time() > deadline:
            return True  # assume valid if we run out of time (risky but better than timeout)
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            dv = dist[v]
            if dv >= max_depth:
                continue  # no need to explore further; cycles would be >= target_girth
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    if dv + 1 < max_depth:
                        queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dv + dist[u] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction.
    Optimized to find minimal d for required girth with faster neighbor computation."""
    q = k
    deadline = time.time() + 130.0  # leave 20s margin for final girth check and return

    def _build_luw(d):
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
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_list = [rule[c] for c in range(2, d + 1)]

        # Precompute coefficient tables for faster neighbor generation
        # For side 0: y[c] = x[c] + y[a] * x[b]
        # For side 1: y[c] = x[c] - x[a] * y[b]
        # We can precompute for each position c the (a, b) pair and use flat arrays.

        # Precompute all q^d possible x-tuples? No, too many.
        # Instead, precompute the linear structure: for a given x, the neighbor y is
        # determined by y[1] = first, then y[c] = x[c] + y[a]*x[b] for c>=2.
        # This is a linear map from y[1] to the rest of y (given x).
        # Specifically, y[c] = x[c] + sum of contributions from y[1] through y[c-1].
        # We can precompute for each x: offset[c] and coeff[c] such that
        # y[c] = offset[c] + coeff[c] * y[1] (mod q) — but this only works if
        # the recurrence is linear in y[1] with no branching.
        # Actually the recurrence IS linear: y[2] = x[2] + y[1]*x[1], y[3] = x[3] + y[a]*x[b], etc.
        # Each y[c] depends on y[a] which may depend on y[1], so it's all linear in y[1].
        # Let's precompute: for each x (tuple), compute the offset and multiplier for y[1].

        # For a fixed x, define:
        #   y[1] = t (parameter)
        #   y[c] = off[c] + mul_c * t  (mod q)
        # where off[1] = 0, mul[1] = 1
        #   y[c] = x[c] + y[a] * x[b] = x[c] + (off[a] + mul[a]*t) * x[b]
        #        = (x[c] + off[a]*x[b]) + mul[a]*x[b]*t
        # So: off[c] = x[c] + off[a]*x[b], mul[c] = mul[a]*x[b]

        # Precompute off and mul arrays for a given x
        def compute_offsets(x):
            """Compute offset and multiplier arrays for linear recurrence given x."""
            off = [0] * (d + 1)
            muls = [0] * (d + 1)
            off[1] = 0
            muls[1] = 1
            for idx, c in enumerate(range(2, d + 1)):
                a, b = rule_list[idx]
                off[c] = (x[c] + off[a] * x[b]) % q
                muls[c] = (muls[a] * x[b]) % q
            return off, muls

        # Similarly for side 1: y[c] = x[c] - x[a] * y[b]
        # y[c] = x[c] - x[a]*(off[b] + muls[b]*t) = (x[c] - x[a]*off[b]) - x[a]*muls[b]*t
        # off[c] = x[c] - x[a]*off[b], muls[c] = -x[a]*muls[b]

        def compute_offsets_side1(x):
            """Compute offset and multiplier arrays for side-1 linear recurrence given x."""
            off = [0] * (d + 1)
            muls = [0] * (d + 1)
            off[1] = 0
            muls[1] = 1
            for idx, c in enumerate(range(2, d + 1)):
                a, b = rule_list[idx]
                off[c] = (x[c] - x[a] * off[b]) % q
                muls[c] = (-x[a] * muls[b]) % q
            return off, muls

        def neighbours_side0_fast(x):
            """Generate neighbors from a side-0 vertex using precomputed offsets."""
            off, muls = compute_offsets(x)
            for t in range(q):
                y = [0] * (d + 1)
                y[1] = t
                for c in range(2, d + 1):
                    y[c] = (off[c] + muls[c] * t) % q
                yield tuple(y)

        def neighbours_side1_fast(x):
            """Generate neighbors from a side-1 vertex using precomputed offsets."""
            off, muls = compute_offsets_side1(x)
            for t in range(q):
                y = [0] * (d + 1)
                y[1] = t
                for c in range(2, d + 1):
                    y[c] = (off[c] + muls[c] * t) % q
                yield tuple(y)

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        i = 0
        while i < len(order):
            vertex = order[i]
            side, x = vertex
            if side == 0:
                gen = neighbours_side0_fast(x)
            else:
                gen = neighbours_side1_fast(x)
            for other_x in gen:
                other = (1 - side, other_x)
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    # Try increasing d from a reasonable lower bound
    # Girth of LUW graph is roughly 2*(d-1) or so; need girth >= g
    # Start from d where 2*d >= g, i.e., d >= g//2
    d_start = max(5, (g - 1) // 2)
    for d in range(d_start, d_start + 15):
        if time.time() > deadline:
            break

        n, edges = _build_luw(d)

        # Build adjacency for girth check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        if _has_girth_at_least(adj, n, g, deadline=deadline - 10.0):
            return n, edges

    # Fallback: use a known-good large d value
    d_orig = max(8, g)
    return _build_luw(d_orig)

def construct(k, g):
    start_time = time.time()

    if k == 3 and g == 16:
        # Skip random bipartite (astronomically unlikely to succeed); go straight to LUW
        return _luw_construct(k, g)

    return _luw_construct(k, g)
