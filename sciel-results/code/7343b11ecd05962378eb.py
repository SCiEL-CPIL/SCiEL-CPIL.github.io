"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower(a, b, c):
    """Check if three 4-element sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(family, s):
    """Check if adding s to family creates no sunflower with any two existing sets."""
    n = len(family)
    for i in range(n):
        fi = family[i]
        for j in range(i + 1, n):
            fj = family[j]
            if _is_sunflower(fi, fj, s):
                return False
    return True

def _greedy_build(order):
    """Build a family greedily following the given order of candidate sets."""
    family = []
    for s in order:
        if _can_add(family, s):
            family.append(s)
    return family

def _local_improve(family, candidates, deadline):
    """Try to improve family by adding sets or swapping."""
    family = list(family)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any candidate not in family
        fam_ids = set(id(s) for s in family)
        for s in candidates:
            if id(s) not in fam_ids and _can_add(family, s):
                family.append(s)
                improved = True
                break
        # Try swapping: remove one, try to add a different one
        if not improved:
            for i in range(len(family)):
                if time.time() >= deadline:
                    break
                saved = family[i]
                family.pop(i)
                best_add = None
                for s in candidates:
                    if id(s) == id(saved) or id(s) in set(id(x) for x in family):
                        continue
                    if _can_add(family, s):
                        best_add = s
                        break
                if best_add is not None:
                    family.append(best_add)
                    improved = True
                else:
                    family.insert(i, saved)
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    # Generate all C(16,4) = 1820 four-element subsets
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]

    best_family = []
    best_len = 0

    # Multiple random restarts with greedy + local search
    while time.time() < deadline:
        random_order = all_sets[:]
        random.shuffle(random_order)
        family = _greedy_build(random_order)
        if len(family) > best_len:
            best_len = len(family)
            best_family = family[:]

        # Local search improvement
        if time.time() < deadline - 3:
            family = _local_improve(family, all_sets, deadline)
            if len(family) > best_len:
                best_len = len(family)
                best_family = family[:]

    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
