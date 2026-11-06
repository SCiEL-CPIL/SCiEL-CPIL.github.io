import numpy as np
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start = time.time()
    deadline = 80.0

    n = 16
    all_sets = list(combinations(range(1, n+1), 4))
    m = len(all_sets)
    set_objs = [frozenset(s) for s in all_sets]

    # Bitmasks for each 4-set
    masks = np.zeros(m, dtype=np.uint32)
    for i, s in enumerate(all_sets):
        mask = 0
        for x in s:
            mask |= (1 << (x-1))
        masks[i] = mask

    # Precompute contain[X] for |X|<=2: list of 4-set indices containing X
    contain = defaultdict(list)
    for i, s in enumerate(all_sets):
        s_set = frozenset(s)
        contain[frozenset()].append(i)
        for x in s:
            contain[frozenset([x])].append(i)
        for x, y in combinations(s, 2):
            contain[frozenset([x, y])].append(i)

    # Precompute forbidden_pairs: for each S, the set of pairs (A,B) that form a sunflower with S
    forbidden_pairs = [set() for _ in range(m)]
    for s_idx in range(m):
        S_set = set_objs[s_idx]
        S_list = list(S_set)
        subsets = [frozenset()]
        for x in S_list:
            subsets.append(frozenset([x]))
        for x, y in combinations(S_list, 2):
            subsets.append(frozenset([x, y]))

        for X in subsets:
            complement = S_set - X
            cand_list = contain[X]
            if len(cand_list) < 2:
                continue
            cand_indices = np.array(cand_list, dtype=np.int64)

            comp_mask = 0
            for x in complement:
                comp_mask |= (1 << (x-1))
            if comp_mask != 0:
                cand_masks = masks[cand_indices]
                valid = (cand_masks & comp_mask) == 0
                cand_indices = cand_indices[valid]
            if len(cand_indices) < 2:
                continue

            X_mask = 0
            for x in X:
                X_mask |= (1 << (x-1))
            cand_masks = masks[cand_indices]
            cand_minus_X = cand_masks & ~X_mask

            n_cand = len(cand_indices)
            for i in range(n_cand - 1):
                rest = cand_minus_X[i+1:]
                disjoint_mask = (cand_minus_X[i] & rest) == 0
                valid_j = np.where(disjoint_mask)[0]
                for j_offset in valid_j:
                    j = i + 1 + int(j_offset)
                    forbidden_pairs[s_idx].add((cand_indices[i], cand_indices[j]))

    # Greedy with multiple random orderings to find large sunflower-free families
    best_family = []
    best_size = 0
    rng = random.Random(42)

    for trial in range(2000):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        family = []
        family_set = set()
        family_pairs = set()
        for idx in order:
            if idx in family_set:
                continue
            if not (family_pairs & forbidden_pairs[idx]):
                for f in family:
                    pair = (min(f, idx), max(f, idx))
                    family_pairs.add(pair)
                family.append(idx)
                family_set.add(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

    result = [sorted(all_sets[i]) for i in best_family]
    return result
