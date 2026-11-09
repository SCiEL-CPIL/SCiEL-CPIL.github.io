import time
import random
from itertools import combinations
from collections import defaultdict

def _check_conflict(s_obj, family_sets):
    """Check if s_obj forms a sunflower with any two sets in family_sets."""
    groups = defaultdict(list)
    for a in family_sets:
        c = a & s_obj
        groups[c].append(a)
    
    for c, group in groups.items():
        m = len(group)
        if m < 2:
            continue
        # Precompute remainders
        rems = [a - c for a in group]
        for ai in range(m):
            rem_a = rems[ai]
            for bi in range(ai + 1, m):
                if not (rem_a & rems[bi]):
                    return True
    return False

def _greedy(order, deadline):
    family = []
    family_sets = []
    for i in range(len(order)):
        if time.time() > deadline:
            break
        s_obj = set(order[i])
        if not _check_conflict(s_obj, family_sets):
            family.append(order[i])
            family_sets.append(s_obj)
    return family

def construct():
    deadline = time.time() + 55
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)  # 1820

    best = []

    # Pass 1: lexicographic order
    fam = _greedy(all_sets, deadline)
    if len(fam) > len(best):
        best = fam

    # Pass 2: random restarts
    remaining = deadline - time.time()
    if remaining > 5:
        rng = random.Random(12345)
        for trial in range(100):
            if time.time() > deadline - 3:
                break
            order = all_sets[:]
            rng.shuffle(order)
            fam = _greedy(order, deadline)
            if len(fam) > len(best):
                best = fam

    # Pass 3: try to improve by swapping - local search
    remaining = deadline - time.time()
    if remaining > 10 and best:
        best_sets = [set(s) for s in best]
        improved = True
        rng2 = random.Random(999)
        while improved and time.time() < deadline - 3:
            improved = False
            # Try removing one set and adding a better one
            for idx in range(len(best_sets)):
                if time.time() > deadline - 3:
                    break
                candidate_family = best_sets[:idx] + best_sets[idx+1:]
                # Find a set not in family that can be added
                best_add = None
                for s_tuple in all_sets:
                    s_obj = set(s_tuple)
                    if s_obj in candidate_family:
                        continue
                    if not _check_conflict(s_obj, candidate_family):
                        # Check if this is "better" - just take first valid
                        best_add = (s_tuple, s_obj)
                        break
                if best_add:
                    new_family = candidate_family + [best_add[1]]
                    if len(new_family) > len(best_sets):
                        best_sets = new_family
                        best = [tuple(s) for s in new_family]
                        improved = True
                        break
                    # Even if same size, might help future swaps - skip for now

    return [sorted(s) for s in best]
