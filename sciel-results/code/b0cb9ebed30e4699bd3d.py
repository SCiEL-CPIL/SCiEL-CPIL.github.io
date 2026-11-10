import time
from itertools import combinations

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_TIME_LIMIT = 85.0  # Safety margin before 90s kill

MOD = 101
DENOMS = list(range(1, 11))

# Precompute modular inverses
_inv = [0] * 11
for i in range(1, 11):
    _inv[i] = pow(i, -1, MOD)

# Generate all non-empty subsets with their size and sum
# _subset_info[k] = list of (sum_mod_101, tuple_of_denoms)
_subset_info = {}
for k in range(1, 11):
    subsets = []
    for combo in combinations(DENOMS, k):
        s = 0
        for d in combo:
            s = (s + _inv[d]) % MOD
        subsets.append((s, combo))
    _subset_info[k] = subsets

# Find minimum K such that all residues are covered by subsets of size <= K
_best_K = 10
for K in range(1, 11):
    if time.time() - _START_TIME > _TIME_LIMIT:
        break
    covered = [False] * MOD
    for k in range(1, K + 1):
        for s, _ in _subset_info[k]:
            covered[s] = True
    if all(covered):
        _best_K = K
        break

# For each residue, collect all subsets of size <= _best_K that cover it
# Then pick the one with minimum size (to minimize max row length)
_best_for_residue = [None] * MOD
_min_size_for_residue = [11] * MOD

for k in range(1, _best_K + 1):
    if time.time() - _START_TIME > _TIME_LIMIT:
        break
    for s, combo in _subset_info[k]:
        if _best_for_residue[s] is None or k < _min_size_for_residue[s]:
            _best_for_residue[s] = list(combo)
            _min_size_for_residue[s] = k

# Safety: ensure all residues are covered
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
