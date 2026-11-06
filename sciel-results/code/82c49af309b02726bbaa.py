"""Q3-free dense graph on 16 vertices via bitset counting + greedy edge removal."""
import time
import random

N = 16
NPAIRS = N * (N - 1) // 2
DEADLINE = time.time() + 135.0  # safety margin under the 150s kill

# Precomputed helpers
NEIGHBOR_PAIRS = [(a, b) for a in range(N) for b in range(a + 1, N)]
PAIR_IDX = [[0] * N for _ in range(N)]
for i, (a, b) in enumerate(NEIGHBOR_PAIRS):
    PAIR_IDX[a][b] = i
    PAIR_IDX[b][a] = i


def _count_q3(adj):
    """Number of Q3 subgraphs = sum over non-edges (a,d) of C(common_neighbors,2)."""
    total = 0
    for a in range(N):
        ra = adj[a]
        for d in range(a + 1, N):
            if (ra >> d) & 1:
                continue
            c = ra & adj[d]
            m = c.bit_count()
            if m >= 2:
                total += m * (m - 1) // 2
    return total


def _greedy_remove(adj, counts):
    """Remove the edge contained in the most Q3s until no Q3 remains."""
    while counts > 0:
        best_i = -1
        best_c = -1
        for i in range(NPAIRS):
            if counts[i] > best_c:
                best_c = counts[i]
                best_i = i
        if best_i < 0:
            break
        a, b = NEIGHBOR_PAIRS[best_i]
        adj[a] &= ~(1 << b)
        adj[b] &= ~(1 << a)
        counts[best_i] = 0
        # update counts for all other edges sharing a or b
        for x in range(N):
            if x == a or x == b or not ((adj[a] >> x) & 1):
                continue
            ia = PAIR_IDX[a][x]
            if ia != best_i:
                counts[ia] -= best_c
        for y in range(N):
            if y == a or y == b or not ((adj[b] >> y) & 1):
                continue
            ib = PAIR_IDX[b][y]
            if ib != best_i and ib != best_i:
                counts[ib] -= best_c
        counts = _count_q3(adj)  # authoritative recount (cheap, bounds drift)
    return counts


def _edges_from_adj(adj):
    return [[a, b] for a in range(N) for b in range(a + 1, N) if (adj[a] >> b) & 1]


def construct():
    rng = random.Random(12345)
    best_edges = 0
    best_adj = None

    for seed in range(200000):
        if time.time() > DEADLINE:
            break
        # random dense-ish starting graph
        adj = [0] * N
        for a in range(N):
            for b in range(a + 1, N):
                if rng.random() < 0.75:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        counts = _count_q3(adj)
        if counts == 0:
            return _edges_from_adj(adj)
        _greedy_remove(adj, counts)
        if _count_q3(adj) != 0:
            continue
        e = sum(adj[i].bit_count() for i in range(N)) // 2
        if e > best_edges:
            best_edges = e
            best_adj = adj[:]
            if best_edges >= 60:
                break
        if best_edges >= 58 and time.time() > time.time() - 60:
            break

    if best_adj is None:
        # guaranteed valid fallback: K3,13 (39 edges)
        return [[a, b] for a in range(3) for b in range(3, 16)]
    return _edges_from_adj(best_adj)
