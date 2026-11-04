"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 6.0


def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False

    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Girth check via non-backtracking BFS from each vertex.
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
    return True


def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if (g - 5) % 2 == 0:
        d = g - 4

    if q == 4:
        table = [[0, 0, 0, 0],
                 [0, 1, 2, 3],
                 [0, 2, 3, 1],
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


class _ShortCycleCounter:
    """
    Fast counter of non-backtracking walks that close into a cycle of length
    < g. Uses a shared per-vertex visited array with a round stamp, so each
    BFS from a start vertex reuses memory and avoids dict allocation overhead.
    """

    def __init__(self, adj, n, g):
        self.adj = adj
        self.n = n
        self.g = g
        self.max_depth = g - 2
        self.visited = [0] * n
        self.round = 0
        self.dq = deque()

    def count(self):
        adj = self.adj
        n = self.n
        g = self.g
        visited = self.visited
        max_depth = self.max_depth
        count = 0
        dq = self.dq

        for s in range(n):
            self.round += 1
            rnd = self.round
            dq.clear()
            for nb in adj[s]:
                if visited[nb] != rnd:
                    visited[nb] = rnd
                    dq.append((nb, s, 1))

            while dq:
                curr, prev, depth = dq.popleft()
                if curr == s:
                    if depth + 1 < g:
                        count += 1
                    continue
                if depth >= max_depth:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if visited[nb] != rnd:
                        visited[nb] = rnd
                        dq.append((nb, curr, depth + 1))

        return count


def _count_short_cycles(adj, n, g):
    """Count cycles of length < g using non-backtracking walks."""
    return _ShortCycleCounter(adj, n, g).count()


def _generate_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices using configuration
    model with rejection and retry.
    """
    rng = random.Random(seed)
    for _ in range(80):
        stubs = []
        for v in range(n):
            stubs.extend((v, v, v, v))
        rng.shuffle(stubs)

        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (u, v) if u < v else (v, u)
            if edge in edges:
                valid = False
                break
            edges.add(edge)

        if not valid:
            continue

        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == 4 for d in deg):
            return list(edges)

    return None


def _improve_girth(n, edges, g, time_limit, rng):
    """
    Simulated annealing with 2-switch operations to increase girth.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v) if u < v else (v, u))

    counter = _ShortCycleCounter(adj, n, g)
    current_cost = counter.count()
    if current_cost == 0:
        return list(edges), 0

    best_edges = list(edges)
    best_cost = current_cost

    t_start = time.time()
    temp = 30.0
    cooling = 0.998
    min_temp = 0.2
    inv_k = 1.0 / 30.0

    while time.time() < t_start + time_limit and temp > min_temp:
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        if e1 == e2:
            continue

        u1, v1 = e1
        u2, v2 = e2

        if len({u1, v1, u2, v2}) < 4:
            continue

        # Choose one of the two non-crossing 2-switches at random.
        if rng.random() < 0.5:
            na, nb = (u1, u2)
            nc, nd = (v1, v2)
        else:
            na, nb = (u1, v2)
            nc, nd = (v1, u2)

        new_e1 = (na, nb) if na < nb else (nb, na)
        new_e2 = (nc, nd) if nc < nd else (nd, nc)

        if new_e1 == new_e2:
            continue
        if new_e1 in edge_set or new_e2 in edge_set:
            continue

        # Apply switch.
        edge_set.discard(e1)
        edge_set.discard(e2)
        edge_set.add(new_e1)
        edge_set.add(new_e2)

        adj[u1].discard(v1); adj[v1].discard(u1)
        adj[u2].discard(v2); adj[v2].discard(u2)
        adj[na].add(nb); adj[nb].add(na)
        adj[nc].add(nd); adj[nd].add(nc)

        edges.remove(e1)
        edges.remove(e2)
        edges.append(new_e1)
        edges.append(new_e2)

        new_cost = counter.count()

        if new_cost < best_cost:
            best_cost = new_cost
            best_edges = list(edges)

        if new_cost <= current_cost or rng.random() < pow(2.718, -(new_cost - current_cost) * inv_k / temp):
            current_cost = new_cost
        else:
            # Revert.
            edge_set.discard(new_e1)
            edge_set.discard(new_e2)
            edge_set.add(e1)
            edge_set.add(e2)

            adj[na].discard(nb); adj[nb].discard(na)
            adj[nc].discard(nd); adj[nd].discard(nc)
            adj[u1].add(v1); adj[v1].add(u1)
            adj[u2].add(v2); adj[v2].add(u2)

            edges.remove(new_e1)
            edges.remove(new_e2)
            edges.append(e1)
            edges.append(e2)

        temp *= cooling

    return best_edges, best_cost


def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN

    # 1. LUW baseline.
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # 2. For (4,10), attempt to beat the baseline with a smaller graph.
    if k == 4 and g == 10:
        rng = random.Random(42)

        # Candidate sizes to attempt. 
        # The LUW graph has n=2048. Any valid graph with n < 2048 improves the score.
        # We try smaller sizes first, but expand to larger sizes if smaller ones fail,
        # as larger graphs have more degrees of freedom for the 2-switch moves.
        candidates = [320, 340, 360, 400, 500, 700, 1000, 1500]
        
        # Filter out sizes >= current best (LUW is 2048, so all are < 2048)
        candidates = [s for s in candidates if s < best_n]
        
        total_time_available = deadline - time.time()
        # Allocate time proportionally to number of candidates, but ensure at least some time for each
        # We'll just iterate and break if time runs out.
        
        for size in candidates:
            if time.time() > deadline:
                break
            
            # If we already found a smaller graph, skip larger ones? 
            # No, we iterate smallest to largest. If we find 320, we stop.
            # If we find 500, we continue to try 700? No, 500 is better than 700.
            # So if we find a valid graph, we should update best and stop if it's the smallest possible?
            # We can't know if a smaller one exists, but we tried 320 first.
            # So if we find a valid graph at size S, we can stop because we tried all smaller sizes.
            
            attempts = 3 # Reduce attempts per size to allow trying more sizes
            for attempt in range(attempts):
                if time.time() > deadline:
                    break

                edges = _generate_4_regular(size, seed=1000 + size * 31 + attempt * 7)
                if edges is None:
                    continue

                # Quick check: count short cycles
                adj = [set() for _ in range(size)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                counter = _ShortCycleCounter(adj, size, g)
                cost = counter.count()
                
                if cost == 0:
                    # Validate fully
                    if _validate_graph(size, edges, k, g):
                        best_n = size
                        best_edges = edges
                        # Since we iterate from small to large, this is the best we found so far.
                        # We can break out of the size loop as well.
                        return best_n, best_edges

                time_left = deadline - time.time()
                if time_left < 4:
                    break

                # Run SA for a portion of the remaining time
                # Allocate 20% of remaining time to this attempt, min 2s
                sa_time = max(2.0, (time_left - 2.0) * 0.2)
                
                improved_edges, cost = _improve_girth(size, edges, g, sa_time, rng)

                if cost == 0:
                    if _validate_graph(size, improved_edges, k, g):
                        best_n = size
                        best_edges = improved_edges
                        return best_n, best_edges

    return best_n, best_edges

# EVOLVE-BLOCK-END
