"""Structural rewrite: dense multipartite core + greedy Q3-safe complement edges."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

N = 16
DEADLINE = 150.0
SAFETY = 12.0

# 12 Q3 edge-paths of length 3 (u - a - b - v), i.e. the 3 non-diagonal 4-cycles of Q3
Q3_PATHS = [
    (0, 1, 3, 7),
    (0, 2, 6, 7),
    (1, 0, 2, 6),
    (1, 3, 2, 6),
    (2, 0, 1, 3),
    (2, 6, 1, 3),
    (3, 1, 0, 2),
    (3, 7, 6, 2),
    (4, 5, 7, 6),
    (4, 6, 7, 5),
    (5, 4, 6, 7),
    (5, 7, 6, 4),
]


def build_initial():
    """Complete multipartite on 16 vertices with part sizes 3,3,3,3,2,2 (53 edges)."""
    adj = [0] * N
    edges = []
    start = 0
    for size in (3, 3, 3, 3, 2, 2):
        end = start + size
        for i in range(start, end):
            for j in range(end, N):
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                edges.append([i, j])
        for i in range(start, end):
            for j in range(start, i):
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                edges.append([i, j])
        start = end
    return adj, edges


def is_q3_free(adj):
    """Return True iff the graph (adjacency bitsets) contains no Q3 subgraph."""
    for (u, a, b, v) in Q3_PATHS:
        if not (adj[u] >> a) & 1:
            continue
        if not (adj[a] >> b) & 1:
            continue
        if not (adj[b] >> v) & 1:
            continue
        # u-a-b-v present; check for a 4th vertex x closing the cube
        xmask = adj[u] & adj[v] & adj[a] & adj[b]
        xmask &= ~((1 << u) | (1 << a) | (1 << b) | (1 << v))
        if xmask:
            return False
    return True


def construct():
    t0 = time.time()
    adj, edges = build_initial()
    best_edges = edges
    best_count = len(edges)

    # candidate complement edges: pairs within the same part
    parts = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13), (14, 15)]
    candidates = []
    for part in parts:
        for i, j in combinations(part, 2):
            candidates.append((i, j))

    def time_left():
        return (DEADLINE - SAFETY) - (time.time() - t0) > 0

    # Pass 1: add as many complement edges as possible without creating a Q3
    for (i, j) in candidates:
        if not time_left():
            break
        adj[i] |= 1 << j
        adj[j] |= 1 << i
        if is_q3_free(adj):
            edges.append([i, j])
        else:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)

    if len(edges) > best_count:
        best_count = len(edges)
        best_edges = edges[:]

    # Pass 2: local search -- drop a complement edge, try to add a different one
    # (only useful if pass 1 was blocked early by the deadline).
    if time_left() and len(candidates) > 0:
        cur_edges = edges[:]
        cur_adj = [x for x in adj]
        for _ in range(2000):
            if not time_left():
                break
            improved = False
            for k in range(len(candidates) - 1, -1, -1):
                (i, j) = candidates[k]
                if (cur_adj[i] >> j) & 1:
                    continue  # already present
                # temporarily remove one random present complement edge to free a slot
                for l in range(len(candidates) - 1, -1, -1):
                    if l == k:
                        continue
                    (p, q) = candidates[l]
                    if not ((cur_adj[p] >> q) & 1):
                        continue
                    # remove (p,q)
                    cur_adj[p] &= ~(1 << q)
                    cur_adj[q] &= ~(1 << p)
                    # try add (i,j)
                    cur_adj[i] |= 1 << j
                    cur_adj[j] |= 1 << i
                    if is_q3_free(cur_adj):
                        # accept: replace (p,q) with (i,j)
                        for e in range(len(cur_edges)):
                            if (cur_edges[e][0] == p and cur_edges[e][1] == q) or \
                               (cur_edges[e][0] == q and cur_edges[e][1] == p):
                                cur_edges[e] = [i, j]
                                break
                        improved = True
                        break
                    else:
                        cur_adj[i] &= ~(1 << j)
                        cur_adj[j] &= ~(1 << i)
                    # revert removal of (p,q)
                    cur_adj[p] |= 1 << q
                    cur_adj[q] |= 1 << p
                if improved:
                    break
            if not improved:
                break
        if len(cur_edges) > best_count:
            best_count = len(cur_edges)
            best_edges = cur_edges[:]

    return best_edges
# EVOLVE-BLOCK-END
