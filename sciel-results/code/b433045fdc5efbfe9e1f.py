"""Rewritten program for sunflower-free 4-uniform family on [1,16]."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 16
    k = 4

    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)

    num_subsets = len(all_subsets)

    # Precompute popcount for all possible 16-bit masks
    popcount = np.zeros(1 << n, dtype=np.int32)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i & (i - 1)] + 1

    def popcount_py(x):
        return popcount[x]

    def can_add_fast(mask_new, chosen_masks):
        """Check if mask_new can be added without forming a sunflower.
        
        Three sets A, B, C form a sunflower if A∩B = A∩C = B∩C.
        For new set M and existing sets A, B:
        M forms sunflower with A, B iff ∃ core I such that I ⊆ M∩A∩B and
        |M∩A∩B| >= 2 (since I must be a 2-subset for k=4... actually no).
        
        Wait, for k=4 sets, the core can be of size 0, 1, 2, 3, or 4.
        A, B, M form a sunflower iff A∩B = A∩M = B∩M.
        Let I = A∩M. Then we need B∩M = I and A∩B = I.
        
        Efficient check: For each pair (A, B) in chosen, check if A∩B = A∩M = B∩M.
        This is O(|chosen|^2) per addition.
        
        Better approach: Group by intersection with M.
        For each possible intersection value I (subset of M), collect all chosen sets A
        such that A∩M = I. If two sets A, B in the same group also satisfy A∩B = I,
        then {A, B, M} is a sunflower.
        """
        # Group chosen sets by their intersection with mask_new
        groups = {}
        for m in chosen_masks:
            inter = m & mask_new
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(m)
        
        for inter, members in groups.items():
            if len(members) < 2:
                continue
            L = len(members)
            # Check all pairs in this group
            for i in range(L):
                mi = members[i]
                for j in range(i + 1, L):
                    mj = members[j]
                    if (mi & mj) == inter:
                        return False
        return True

    # Strategy: Use a more efficient approach with precomputed conflict detection
    # For each pair of chosen sets, precompute which new sets would create a sunflower
    # This is expensive, so instead we use the grouping approach above
    
    best_family = []
    best_size = 0

    # Strategy 1: Deterministic Greedy (Lexicographic)
    chosen = []
    for mask in all_subsets:
        if time.time() > deadline:
            break
        if can_add_fast(mask, chosen):
            chosen.append(mask)

    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Local search polish: try to add more sets or swap
    def try_local_search(family, time_limit):
        """Try to improve family by adding or swapping sets."""
        family = family[:]
        chosen_set = set(family)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add any set not in family
            for mask in all_subsets:
                if mask in chosen_set:
                    continue
                if can_add_fast(mask, family):
                    family.append(mask)
                    chosen_set.add(mask)
                    improved = True
            # Try swaps: remove one, add one
            if time.time() < time_limit:
                for i in range(len(family)):
                    if time.time() > time_limit:
                        break
                    saved = family.pop(i)
                    chosen_set.discard(saved)
                    # Try to add two sets to compensate
                    added = []
                    for mask in all_subsets:
                        if mask in chosen_set or mask == saved:
                            continue
                        if can_add_fast(mask, family + added):
                            added.append(mask)
                            chosen_set.add(mask)
                            if len(added) >= 2:
                                break
                    if len(added) >= 1:
                        family.extend(added)
                        improved = True
                    else:
                        family.insert(i, saved)
                        chosen_set.add(saved)
        return family

    # Apply local search to best family
    if time.time() < deadline - 5.0:
        polished = try_local_search(best_family, deadline - 5.0)
        if len(polished) > best_size:
            best_size = len(polished)
            best_family = polished

    # Strategy 2: Randomized Greedy Restarts with local search
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        random.seed(12345 + restart)
        shuffled = all_subsets[:]
        random.shuffle(shuffled)

        chosen = []
        for mask in shuffled:
            if time.time() > deadline - 5.0:
                break
            if can_add_fast(mask, chosen):
                chosen.append(mask)

        if len(chosen) > best_size:
            # Apply local search
            if time.time() < deadline - 5.0:
                polished = try_local_search(chosen, deadline - 5.0)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_family = polished
            else:
                best_size = len(chosen)
                best_family = chosen[:]

            if best_size >= 250:
                break

    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
