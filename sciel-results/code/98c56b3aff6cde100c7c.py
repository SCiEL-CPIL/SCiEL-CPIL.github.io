import time
import sys
from collections import deque

MOD = 101
N_DENOMS = 10
N_SUBSETS = (1 << N_DENOMS) - 1  # 1023

# Precompute at module level
_invs = [0] * (N_DENOMS + 1)
for _x in range(1, N_DENOMS + 1):
    _invs[_x] = pow(_x, MOD - 2, MOD)

_subset_info = []
for _mask in range(1, 1 << N_DENOMS):
    _s = 0
    _size = 0
    _denoms = []
    for _x in range(1, N_DENOMS + 1):
        if _mask & (1 << (_x - 1)):
            _denoms.append(_x)
            _s = (_s + _invs[_x]) % MOD
            _size += 1
    _subset_info.append((_s, _size, _denoms))

# adj_by_res[r] = list of (size, subset_index), sorted by size
_adj_by_res = [[] for _ in range(MOD)]
for _i, (_res, _sz, _denoms) in enumerate(_subset_info):
    _adj_by_res[_res].append((_sz, _i))
for _r in range(MOD):
    _adj_by_res[_r].sort(key=lambda x: x[0])

# Precompute adj lists for each max_k: adj_k[k][r] = list of subset indices
_adj_k = [None] * (N_DENOMS + 1)
for _k in range(1, N_DENOMS + 1):
    _adj_k[_k] = [[] for _ in range(MOD)]
    for _r in range(MOD):
        for _sz, _idx in _adj_by_res[_r]:
            if _sz <= _k:
                _adj_k[_k][_r].append(_idx)
            else:
                break


def _hopcroft_karp(adj, n_left, n_right):
    """
    Hopcroft-Karp bipartite matching.
    adj: list of lists, adj[u] = list of v neighbors for left node u
    Returns (match_l, match_r) or None if no perfect matching.
    """
    INF = float('inf')
    match_l = [-1] * n_left
    match_r = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        # Find augmenting paths via BFS
        dq = deque()
        for u in range(n_left):
            if match_l[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = INF
        found = False
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                mu = match_r[v]
                if mu == -1:
                    found = True
                elif dist[mu] == INF:
                    dist[mu] = dist[u] + 1
                    dq.append(mu)
        return found

    def dfs(u):
        for v in adj[u]:
            mu = match_r[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = INF
        return False

    matching_size = 0
    while bfs():
        for u in range(n_left):
            if match_l[u] == -1:
                if dfs(u):
                    matching_size += 1
    return matching_size, match_l, match_r


def construct():
    start_time = time.time()
    DEADLINE = 85.0
    n_residues = MOD
    n_subsets = N_SUBSETS

    # Try increasing max_k using Hopcroft-Karp
    for max_k in range(1, N_DENOMS + 1):
        if time.time() - start_time > DEADLINE - 10:
            break

        adj = _adj_k[max_k]

        # Check if any residue has no options
        has_options = True
        for r in range(n_residues):
            if not adj[r]:
                has_options = False
                break
        if not has_options:
            continue

        matching_size, match_l, match_r = _hopcroft_karp(adj, n_residues, n_subsets)
        if matching_size == n_residues:
            result = []
            for r in range(n_residues):
                subset_idx = match_l[r]
                result.append(_subset_info[subset_idx][2])
            return result

    # Fallback: Greedy
    result = []
    used_subsets = set()
    for r in range(n_residues):
        best_si = -1
        best_size = N_DENOMS + 1
        for sz, idx in _adj_by_res[r]:
            if idx not in used_subsets and sz < best_size:
                best_size = sz
                best_si = idx
        if best_si != -1:
            result.append(_subset_info[best_si][2])
            used_subsets.add(best_si)
        else:
            found = False
            for sz, idx in _adj_by_res[r]:
                if idx not in used_subsets:
                    result.append(_subset_info[idx][2])
                    used_subsets.add(idx)
                    found = True
                    break
            if not found:
                result.append([1])
    return result
