"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import random
import time
import math
from itertools import combinations

N = 18
FULL_MASK = (1 << N) - 1
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_INDEX = {}
for idx, (u, v) in enumerate(ALL_EDGES):
    EDGE_INDEX[(u, v)] = idx


def _is_k4_free(adj):
    for a in range(N):
        na = adj[a]
        for b in range(a + 1, N):
            if not (na >> b) & 1:
                continue
            common_ab = na & adj[b]
            for c in range(b + 1, N):
                if not (common_ab >> c) & 1:
                    continue
                common_abc = common_ab & adj[c]
                mask_above_c = FULL_MASK ^ ((1 << (c + 1)) - 1)
                if common_abc & mask_above_c:
                    return False
    return True


def _largest_triangle_free(adj):
    # Collect all triangle masks
    tri_masks = []
    for i in range(N):
        ai = adj[i]
        for j in range(i + 1, N):
            if not (ai >> j) & 1:
                continue
            common = ai & adj[j]
            for k in range(j + 1, N):
                if (common >> k) & 1:
                    tri_masks.append((1 << i) | (1 << j) | (1 << k))

    if not tri_masks:
        return N

    # Descending size, with early pruning
    for size in range(N, 0, -1):
        for subset in combinations(range(N), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            has_tri = False
            for tm in tri_masks:
                if (mask & tm) == tm:
                    has_tri = True
                    break
            if not has_tri:
                return size
    return 0


def _bits_to_edges(adj):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges


def _turan(n, r):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if i % r != j % r:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _random_k4_free(rng, p=0.45):
    # Start with a random sparse graph and add edges while keeping K4-free
    adj = [0] * N
    for (u, v) in ALL_EDGES:
        if rng.random() < p:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    # Add random edges while K4-free
    order = ALL_EDGES[:]
    rng.shuffle(order)
    for (u, v) in order:
        if (adj[u] >> v) & 1:
            continue
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        if _is_k4_free(new_adj):
            adj = new_adj
    return adj


def _line_graph_bipartite(rng, p=0.5):
    # Line graph of a random bipartite graph on 9+9 vertices is K4-free.
    # Vertices 0..8 = left, 9..17 = right.
    adj = [0] * N
    left = list(range(9))
    right = list(range(9, 18))
    # Build random bipartite edges
    bip_edges = []
    for u in left:
        for v in right:
            if rng.random() < p:
                bip_edges.append((u, v))
    # Line graph: two edges adjacent iff share an endpoint
    m = len(bip_edges)
    # Map each bipartite edge to a graph vertex 0..17 (need m <= 18)
    if m > 18:
        bip_edges = bip_edges[:18]
        m = 18
    for i in range(m):
        a1, a2 = bip_edges[i]
        for j in range(i + 1, m):
            b1, b2 = bip_edges[j]
            if a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    rng = random.Random(12345)

    # Multiple diverse starts
    starts = []
    starts.append(_turan(N, 3))
    starts.append(_turan(N, 2))
    for _ in range(3):
        starts.append(_random_k4_free(rng, p=0.45))
    for _ in range(2):
        starts.append(_line_graph_bipartite(rng, p=0.5))

    best_adj = None
    best_score = N + 1
    for s in starts:
        if not _is_k4_free(s):
            continue
        sc = _largest_triangle_free(s)
        if sc < best_score:
            best_score = sc
            best_adj = s[:]

    if best_adj is None:
        best_adj = _turan(N, 3)
        best_score = _largest_triangle_free(best_adj)

    if time.time() > deadline:
        return _bits_to_edges(best_adj)

    # Simulated annealing
    def evaluate(adj_bits):
        return _largest_triangle_free(adj_bits)

    current_adj = best_adj[:]
    current_score = best_score

    T = 2.0
    T_min = 0.02
    cooling = 0.9985

    while time.time() < deadline - 15:
        move_type = rng.random()
        if move_type < 0.45:
            non_edges = [(u, v) for (u, v) in ALL_EDGES if not (current_adj[u] >> v) & 1]
            if not non_edges:
                continue
            u, v = rng.choice(non_edges)
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if not _is_k4_free(new_adj):
                continue
            new_score = evaluate(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        elif move_type < 0.8:
            edges_present = [(u, v) for (u, v) in ALL_EDGES if (current_adj[u] >> v) & 1]
            if not edges_present:
                continue
            u, v = rng.choice(edges_present)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_score = evaluate(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        else:
            edges_present = [(u, v) for (u, v) in ALL_EDGES if (current_adj[u] >> v) & 1]
            non_edges = [(u, v) for (u, v) in ALL_EDGES if not (current_adj[u] >> v) & 1]
            if not edges_present or not non_edges:
                continue
            u1, v1 = rng.choice(edges_present)
            u2, v2 = rng.choice(non_edges)
            if {u1, v1} == {u2, v2}:
                continue
            new_adj = current_adj[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            if not _is_k4_free(new_adj):
                continue
            new_score = evaluate(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]

        T = max(T_min, T * cooling)
        if best_score <= 9:
            break

    # Final local polish: try all single-edge add/remove on best
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            adj = best_adj[:]
            for (u, v) in ALL_EDGES:
                if (adj[u] >> v) & 1:
                    continue
                new_adj = adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                if not _is_k4_free(new_adj):
                    continue
                new_score = evaluate(new_adj)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = new_adj[:]
                    improved = True
                    break
            if improved:
                continue
            for (u, v) in ALL_EDGES:
                if not (adj[u] >> v) & 1:
                    continue
                new_adj = adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                new_score = evaluate(new_adj)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = new_adj[:]
                    improved = True
                    break

    return _bits_to_edges(best_adj)
