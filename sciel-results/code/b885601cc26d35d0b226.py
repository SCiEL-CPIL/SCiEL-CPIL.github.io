import time
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
# Precompute all necessary data at module load time
MOD = 101
DENOMS = list(range(1, 11))

# Precompute modular inverses
_inv = [0] * 11
for i in range(1, 11):
    _inv[i] = pow(i, -1, MOD)

# Generate all non-empty subsets with their size and sum
# _subset_info[k] = list of (sum_mod_101, tuple_of_denoms)
_subset_info = {}
_all_subsets = []  # (size, sum_mod_101, tuple_of_denoms)
for k in range(1, 11):
    subsets = []
    for combo in combinations(DENOMS, k):
        s = 0
        for d in combo:
            s = (s + _inv[d]) % MOD
        subsets.append((s, combo))
        _all_subsets.append((k, s, combo))
    _subset_info[k] = subsets

# Hopcroft-Karp algorithm for bipartite matching
# Left side: residues 0..100
# Right side: subsets (indexed 0..1022)
# Edge exists if subset's sum == residue

def hopcroft_karp(left_size, right_size, adj):
    """
    adj: list of lists, adj[i] = list of right nodes connected to left node i
    Returns matching: list where matching[i] = right node matched to left node i, or -1
    """
    INF = float('inf')
    pair_u = [-1] * left_size
    pair_v = [-1] * right_size
    dist = [0] * left_size
    
    def bfs():
        queue = deque()
        for u in range(left_size):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == INF:
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found
    
    def dfs(u):
        for v in adj[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False
    
    matching = 0
    while bfs():
        for u in range(left_size):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    
    return pair_u

# Find minimum K such that a perfect matching exists using only subsets of size <= K
_best_K = 10
_best_matching = None

for K in range(1, 11):
    # Build adjacency: for each residue, which subsets (of size <= K) have that sum
    # We only consider subsets of size exactly <= K
    adj = [[] for _ in range(MOD)]
    subset_indices = []  # map from subset index to (size, combo)
    
    for idx, (size, s, combo) in enumerate(_all_subsets):
        if size <= K:
            adj[s].append(idx)
            subset_indices.append((size, combo))
    
    # Check if every residue has at least one subset
    if any(len(adj[r]) == 0 for r in range(MOD)):
        continue
    
    # Run Hopcroft-Karp
    pair_u = hopcroft_karp(MOD, len(_all_subsets), adj)
    
    # Check if perfect matching
    if all(pair_u[r] != -1 for r in range(MOD)):
        _best_K = K
        _best_matching = pair_u
        break

# Extract the assignment
_best_for_residue = [None] * MOD
if _best_matching is not None:
    for r in range(MOD):
        idx = _best_matching[r]
        if idx >= 0:
            size, s, combo = _all_subsets[idx]
            _best_for_residue[r] = list(combo)

# Fallback: if matching failed, use greedy assignment
if any(lst is None for lst in _best_for_residue):
    _best_for_residue = [None] * MOD
    for k in range(1, 11):
        for s, combo in _subset_info[k]:
            if _best_for_residue[s] is None:
                _best_for_residue[s] = list(combo)
    for r in range(MOD):
        if _best_for_residue[r] is None:
            for k in range(1, 11):
                for s, combo in _subset_info[k]:
                    if s == r:
                        _best_for_residue[r] = list(combo)
                        break
                if _best_for_residue[r] is not None:
                    break

def construct():
    # Return a deep copy to prevent mutation of the global cache
    return [list(lst) for lst in _best_for_residue]
# EVOLVE-BLOCK-END
