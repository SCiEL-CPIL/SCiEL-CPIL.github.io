"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']

# Base adjacency (path). It is bipartite: even/odd partition.
_BASE_ADJ = [set() for _ in range(N)]
_BASE_EDGE_SET = set()
for u, v in P['edges']:
    _BASE_ADJ[u].add(v)
    _BASE_ADJ[v].add(u)
    _BASE_EDGE_SET.add((u, v) if u < v else (v, u))

# Candidate edges: only even-odd non-base edges (guarantees bipartite => triangle-free).
_CANDIDATES = []
for u in range(N):
    for v in range(u + 1, N):
        if (u, v) in _BASE_EDGE_SET:
            continue
        if (u & 1) == (v & 1):
            continue  # same parity would create an odd cycle (triangle risk)
        _CANDIDATES.append((u, v))

# Precompute base distances (path) so we know which pairs already satisfy dist<=4.
def _path_dist(a, b):
    return abs(a - b)

# Pairs that are NOT within distance 4 in the base path (these need help).
_HARD_PAIRS = []
for a in range(N):
    for b in range(a + 1, N):
        if abs(a - b) > 4:
            _HARD_PAIRS.append((a, b))

def _bfs_maxdist(adj, limit=4):
    """Return max shortest-path distance, or -1 if disconnected. Stops expanding beyond limit."""
    n = N
    maxd = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du >= limit:
                continue
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = du + 1
                    dq.append(w)
        # check connectivity
        for d in dist:
            if d == -1:
                return -1
        m = max(dist)
        if m > maxd:
            maxd = m
    return maxd

def _is_valid_fast(edge_list):
    """edge_list: list of (u,v) even-odd non-base edges. Returns True if valid."""
    adj = [set(a) for a in _BASE_ADJ]
    seen = set()
    for u, v in edge_list:
        key = (u, v) if u < v else (v, u)
        if key in _BASE_EDGE_SET or key in seen:
            return False
        seen.add(key)
        adj[u].add(v)
        adj[v].add(u)
    # bipartite by construction => triangle-free guaranteed
    d = _bfs_maxdist(adj, limit=4)
    return d >= 0 and d <= 4

def _to_result(edge_list):
    out = []
    for u, v in edge_list:
        out.append([int(u), int(v)])
    return out

# Seed / known-good 10-edge hub solution.
_FALLBACK = [(12, 1), (12, 3), (12, 5), (12, 7), (12, 9),
             (12, 15), (12, 17), (12, 19), (12, 21), (12, 23)]

def _greedy_cover():
    """Set-cover style greedy: start empty, add the edge that fixes the most hard pairs."""
    # We need a fast per-edge "benefit" estimate. Use a sampled subset of hard pairs
    # to keep it cheap; verify at the end.
    # Sample: focus on the most constrained pairs (those involving endpoints and mid).
    sample = _HARD_PAIRS
    chosen = []
    chosen_set = set()
    adj = [set(a) for a in _BASE_ADJ]

    def pair_dist(a, b, adj):
        # BFS from a to b with early stop
        if a == b:
            return 0
        dist = [-1] * N
        dist[a] = 0
        dq = deque([a])
        while dq:
            u = dq.popleft()
            if u == b:
                return dist[u]
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    if dist[w] > 4:
                        continue
                    dq.append(w)
        return dist[b]

    # Repeatedly pick the candidate that reduces the count of hard pairs beyond 4.
    # To bound cost, limit iterations to a small number and sample pairs.
    sample_pairs = [p for p in sample if p[0] in (0, 23) or p[1] in (0, 23) or p[0] in (12, 11, 13) or p[1] in (12, 11, 13)]
    if not sample_pairs:
        sample_pairs = sample
    # Add a fixed spread
    sample_pairs = list(set(sample_pairs + [(0, 23), (0, 22), (1, 23), (12, 0), (12, 23)]))

    for _ in range(12):
        best_edge = None
        best_score = -1
        for e in _CANDIDATES:
            if e in chosen_set:
                continue
            u, v = e
            # tentative add
            adj[u].add(v)
            adj[v].add(u)
            # count how many sample pairs are now <=4
            cnt = 0
            for (a, b) in sample_pairs:
                if pair_dist(a, b, adj) <= 4:
                    cnt += 1
            # remove
            adj[u].discard(v)
            adj[v].discard(u)
            if cnt > best_score:
                best_score = cnt
                best_edge = e
        if best_edge is None:
            break
        chosen.append(best_edge)
        chosen_set.add(best_edge)
        adj[best_edge[0]].add(best_edge[1])
        adj[best_edge[1]].add(best_edge[0])
        # stop if all sample pairs satisfied
        all_ok = True
        for (a, b) in sample_pairs:
            if pair_dist(a, b, adj) > 4:
                all_ok = False
                break
        if all_ok:
            break
    return chosen

def _local_search_remove(edge_list):
    """Try removing each edge; if still valid, drop it."""
    result = list(edge_list)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            cand = result[:i] + result[i + 1:]
            if _is_valid_fast(cand):
                result = cand
                changed = True
                break
    return result

def _swap_polish(edge_list):
    """Try 1-for-1 swaps to enable further removals, then re-remove."""
    result = list(edge_list)
    if not _is_valid_fast(result):
        return result
    result = _local_search_remove(result)
    result_set = set(result)
    improved = True
    while improved:
        improved = False
        for i in range(len(result)):
            rem = result[i]
            rest = result[:i] + result[i + 1:]
            rest_set = set(rest)
            for e in _CANDIDATES:
                if e in rest_set or e == rem:
                    continue
                test = rest + [e]
                if _is_valid_fast(test):
                    # accept if it allows a subsequent removal
                    shrunken = _local_search_remove(test)
                    if len(shrunken) < len(result):
                        result = shrunken
                        result_set = set(result)
                        improved = True
                        break
            if improved:
                break
    return result

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 90s (and 60s eval)

    # 1) Greedy cover to get a compact seed.
    greedy = _greedy_cover()
    if not _is_valid_fast(greedy):
        greedy = list(_FALLBACK)

    # 2) Remove polish.
    cur = _local_search_remove(greedy)
    if not _is_valid_fast(cur):
        cur = list(_FALLBACK)

    best = cur
    if not _is_valid_fast(best):
        best = list(_FALLBACK)

    # 3) Swap polish on the best.
    if time.time() < deadline:
        polished = _swap_polish(best)
        if _is_valid_fast(polished) and len(polished) <= len(best):
            best = polished

    # 4) Randomized restarts of the greedy with different sample emphasis, keep best.
    if time.time() < deadline:
        import random
        rng = random.Random(12345)
        for restart in range(6):
            if time.time() > deadline:
                break
            # shuffle candidate order by sampling a random subset emphasis
            shuffled = list(_CANDIDATES)
            rng.shuffle(shuffled)
            # Build a greedy on shuffled order with a light benefit heuristic
            chosen = []
            chosen_set = set()
            adj = [set(a) for a in _BASE_ADJ]
            for e in shuffled:
                if time.time() > deadline:
                    break
                u, v = e
                # tentatively add, check if it improves the worst pair distance
                adj[u].add(v)
                adj[v].add(u)
                d = _bfs_maxdist(adj, limit=4)
                adj[u].discard(v)
                adj[v].discard(u)
                # only add if current graph is not yet valid and this edge helps
                cur_d = _bfs_maxdist(adj, limit=4)
                if d >= 0 and (cur_d < 0 or d < cur_d):
                    chosen.append(e)
                    chosen_set.add(e)
                    adj[u].add(v)
                    adj[v].add(u)
                    if d <= 4:
                        break
            if _is_valid_fast(chosen):
                shrunken = _local_search_remove(chosen)
                if _is_valid_fast(shrunken) and len(shrunken) < len(best):
                    best = shrunken

    # Final safety: ensure validity; fall back to the 10-edge hub if anything broke.
    if not _is_valid_fast(best):
        best = list(_FALLBACK)
    if not _is_valid_fast(best):
        best = list(_FALLBACK)

    return _to_result(best)
# EVOLVE-BLOCK-END
