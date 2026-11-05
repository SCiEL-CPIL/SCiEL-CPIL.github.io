import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1, 16] as tuples and bitmasks
_ALL_SUBSETS = list(combinations(range(1, 17), 4))
_TOTAL_SETS = len(_ALL_SUBSETS)
_MASKS = [0] * _TOTAL_SETS
for _i, _s in enumerate(_ALL_SUBSETS):
    _m = 0
    for _bit in _s:
        _m |= (1 << (_bit - 1))
    _MASKS[_i] = _m

# Precompute pairwise intersections for fast lookup
# _INTER[i][j] = _MASKS[i] & _MASKS[j]
# 1820 * 1820 ~ 3.3M entries, manageable
_INTER = [[0] * _TOTAL_SETS for _ in range(_TOTAL_SETS)]
for _i in range(_TOTAL_SETS):
    _mi = _MASKS[_i]
    for _j in range(_i + 1, _TOTAL_SETS):
        _v = _mi & _MASKS[_j]
        _INTER[_i][_j] = _v
        _INTER[_j][_i] = _v


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    best_family = []
    best_size = 0

    random.seed(42)

    def greedy_pass(order):
        """Run a single greedy pass with the given order of set indices."""
        family = []
        family_masks = []

        for idx in order:
            if time.time() > deadline - 2.0:
                break
            m = _MASKS[idx]

            # Check if adding this set creates a sunflower with any two existing sets
            ok = True
            for i in range(len(family)):
                mi = family_masks[i]
                inter_mi = m & mi
                for j in range(i + 1, len(family)):
                    mj = family_masks[j]
                    inter_mj = m & mj
                    if inter_mi != inter_mj:
                        continue
                    inter_ij = mi & mj
                    if inter_ij == inter_mi:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                family.append(idx)
                family_masks.append(m)

        return family

    # Phase 1: Multiple random greedy restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        order = list(range(_TOTAL_SETS))
        random.shuffle(order)

        family = greedy_pass(order)

        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]

    # Phase 2: Local search / improvement on the best family
    # Try to add sets that weren't in the greedy order, or swap sets
    if best_size > 0 and time.time() < deadline - 5.0:
        # Create a set of indices already in the family for fast lookup
        in_family = set(best_family)

        # Try adding each set not in the family
        for idx in range(_TOTAL_SETS):
            if time.time() > deadline - 3.0:
                break
            if idx in in_family:
                continue

            m = _MASKS[idx]
            ok = True
            for i in range(len(best_family)):
                mi = _MASKS[best_family[i]]
                inter_mi = m & mi
                for j in range(i + 1, len(best_family)):
                    mj = _MASKS[best_family[j]]
                    inter_mj = m & mj
                    if inter_mi != inter_mj:
                        continue
                    inter_ij = mi & mj
                    if inter_ij == inter_mi:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                best_family.append(idx)
                in_family.add(idx)
                best_size = len(best_family)

    return [sorted(_ALL_SUBSETS[idx]) for idx in best_family]
