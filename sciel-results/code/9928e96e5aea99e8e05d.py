"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 16
    k = 4

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)

    m = len(all_masks)  # C(16,4) = 1820

    # Precompute pairwise intersections for fast lookup
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            val = mi & all_masks[j]
            inter[i][j] = val
            inter[j][i] = val

    def check_violation_idx(family_indices, idx):
        """
        Check if adding index idx to family creates a sunflower violation.
        A violation exists if there are two sets a, b in family such that
        inter[a][b] == inter[a][idx] == inter[b][idx].
        """
        nf = len(family_indices)
        fidx = family_indices
        for ii in range(nf):
            a = fidx[ii]
            an = inter[a][idx]
            for jj in range(ii + 1, nf):
                b = fidx[jj]
                ab = inter[a][b]
                bn = inter[b][idx]
                if ab == an == bn:
                    return True
        return False

    def mask_to_list(mask):
        """Convert bitmask to sorted list of 1..16 elements."""
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result

    best_size = 0
    best_family_idx = []

    # Strategy 1: Greedy in natural order
    family = []
    for idx in range(m):
        if time.time() > deadline:
            break
        if not check_violation_idx(family, idx):
            family.append(idx)

    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles with diverse seeds
    seeds = [42, 12345, 7, 99, 2024, 555, 777, 1234, 4321, 8888, 13579, 31415, 27182, 16180, 98765, 54321, 11111, 22222, 33333, 44444]
    max_attempts = len(seeds)

    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break

        random.seed(seeds[attempt])
        order = list(range(m))
        random.shuffle(order)

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_idx(family, idx):
                family.append(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Strategy 3: Local improvement - try to add sets that were skipped
    in_family = set(best_family_idx)
    candidates = [idx for idx in range(m) if idx not in in_family]
    random.shuffle(candidates)

    improved = best_family_idx[:]
    for idx in candidates:
        if time.time() > deadline - 5:
            break
        if not check_violation_idx(improved, idx):
            improved.append(idx)

    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Strategy 4: Swap-based local search
    # Try to improve by swapping one set out for another
    for swap_round in range(3):
        if time.time() > deadline - 3:
            break

        in_family = set(best_family_idx)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)

        improved = best_family_idx[:]
        improved_set = set(improved)
        swaps_made = 0

        for add_idx in candidates:
            if time.time() > deadline - 2:
                break
            # Try removing each member and adding add_idx
            for pos, remove_idx in enumerate(improved):
                if time.time() > deadline - 2:
                    break
                # Temporarily remove remove_idx
                new_family = improved[:pos] + improved[pos+1:] + [add_idx]
                new_set = improved_set - {remove_idx} | {add_idx}

                # Check if new_family is valid
                valid = True
                nf = len(new_family)
                # Only need to check pairs involving add_idx
                for ii in range(nf):
                    if new_family[ii] == add_idx:
                        continue
                    a = new_family[ii]
                    an = inter[a][add_idx]
                    for jj in range(ii + 1, nf):
                        if new_family[jj] == add_idx:
                            continue
                        b = new_family[jj]
                        ab = inter[a][b]
                        bn = inter[b][add_idx]
                        if ab == an == bn:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    improved = new_family
                    improved_set = new_set
                    swaps_made += 1
                    break  # Move to next candidate

        if len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

        if swaps_made == 0:
            break

    # Strategy 5: Final greedy add pass
    in_family = set(best_family_idx)
    candidates = [idx for idx in range(m) if idx not in in_family]
    random.shuffle(candidates)

    improved = best_family_idx[:]
    for idx in candidates:
        if time.time() > deadline - 1:
            break
        if not check_violation_idx(improved, idx):
            improved.append(idx)

    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    # Convert back to sorted lists of 1..16
    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]

    return result
# EVOLVE-BLOCK-END
