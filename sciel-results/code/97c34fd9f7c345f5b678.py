"""Fresh design: bitset local search for a graph with no regular induced subgraph on >=6 vertices."""
import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin under the 150s kill

def _check_subset(adj, S, n):
    # S is a bitmask of the candidate subset. Returns True if induced subgraph is regular.
    # Fast path: degree deltas from the lowest vertex in S.
    first = (S & -S).bit_length() - 1
    base = (adj[first] & S).bit_count()
    S_rest = S & ~(1 << first)
    while S_rest:
        b = (S_rest & -S_rest).bit_length() - 1
        if (adj[b] & S).bit_count() != base:
            return False
        S_rest &= S_rest - 1
    return True

def _is_valid(adj, n):
    # True iff no subset of size >= 6 induces a regular subgraph.
    vertices = list(range(n))
    for size in range(6, n + 1):
        for comb in itertools.combinations(vertices, size):
            S = 0
            for v in comb:
                S |= 1 << v
            if _check_subset(adj, S, n):
                return False
    return True

def _total_violation(adj, n):
    vertices = list(range(n))
    total = 0
    for size in range(6, n + 1):
        for comb in itertools.combinations(vertices, size):
            S = 0
            for v in comb:
                S |= 1 << v
            # count how many vertices in S have degree != base (cheap early exit)
            first = (S & -S).bit_length() - 1
            base = (adj[first] & S).bit_count()
            S_rest = S & ~(1 << first)
            cnt = 0
            while S_rest:
                b = (S_rest & -S_rest).bit_length() - 1
                if (adj[b] & S).bit_count() != base:
                    cnt += 1
                    if cnt >= 2:
                        break
                S_rest &= S_rest - 1
            if cnt == 0:
                total += 1
    return total

def _flip(adj, n, u, v):
    adj[u] ^= 1 << v
    adj[v] ^= 1 << u

def _local_search(adj, n, time_budget):
    cur = [a for a in adj]
    cur_v = _total_violation(cur, n)
    if cur_v == 0:
        return cur
    pairs = list(itertools.combinations(range(n), 2))
    rng = random.Random()
    step = 0
    while time.time() < time_budget and cur_v > 0:
        step += 1
        if step % 200 == 0:
            # random kick to escape local minima
            for _ in range(3):
                u, v = rng.choice(pairs)
                _flip(cur, n, u, v)
            cur_v = _total_violation(cur, n)
            continue
        # pick a random edge to try flipping
        u, v = rng.choice(pairs)
        _flip(cur, n, u, v)
        new_v = _total_violation(cur, n)
        if new_v >= cur_v:
            _flip(cur, n, u, v)  # revert
        else:
            cur_v = new_v
    return cur

def _to_edges(adj, n):
    edges = []
    for u in range(n):
        S = adj[u]
        while S:
            b = (S & -S).bit_length() - 1
            if b > u:
                edges.append([u, b])
            S &= S - 1
    return edges

def construct():
    rng = random.Random(12345)
    best_n = 5
    best_adj = [0] * 5  # empty graph on 5 vertices, always valid

    # Try to build a valid 20-vertex graph via local search from random starts.
    target_n = 20
    max_starts = 300
    for start in range(max_starts):
        if time.time() > _DEADLINE - 20:
            break
        adj = [0] * target_n
        for u in range(target_n):
            for v in range(u + 1, target_n):
                if rng.random() < 0.5:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        # quick local search with a per-start budget
        per_start = min(3.0, max(0.3, (_DEADLINE - 20 - time.time()) / max(1, max_starts - start)))
        adj = _local_search(adj, target_n, time.time() + per_start)
        if _is_valid(adj, target_n):
            return {'n': target_n, 'edges': _to_edges(adj, target_n)}

    # Fallback: return the trivially valid small graph.
    return {'n': best_n, 'edges': _to_edges(best_adj, best_n)}
