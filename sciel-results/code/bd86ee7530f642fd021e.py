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

    # Precompute pairwise intersections
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            inter[i][j] = mi & all_masks[j]
            inter[j][i] = inter[i][j]

    def check_violation_idx(family_indices, idx):
        """
        Check if adding index idx to family creates a sunflower violation.
        A sunflower of size 3 exists if there are three distinct sets a, b, c
        in the family such that inter[a][b] == inter[a][c] == inter[b][c].
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

    # Precompute a "popularity" score for each set to use in heuristic ordering
    # Lower popularity (fewer/smaller intersections) is generally better for packing
    popularity = [0] * m
    for i in range(m):
        mi = all_masks[i]
        pop = 0
        for j in range(m):
            if i != j:
                pop += bin(mi & all_masks[j]).count('1')
        popularity[i] = pop
    
    # Strategy 1: Greedy in order of increasing popularity (heuristic)
    order_heuristic = sorted(range(m), key=lambda x: popularity[x])
    family = []
    for idx in order_heuristic:
        if time.time() > deadline:
            break
        if not check_violation_idx(family, idx):
            family.append(idx)

    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles with local improvement
    random.seed(42)
    max_attempts = 50 # Increased attempts

    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break

        # Mix random and heuristic order
        if attempt % 2 == 0:
            order = list(range(m))
            random.shuffle(order)
        else:
            # Perturbed heuristic
            order = order_heuristic[:]
            # Randomly swap a few elements
            for _ in range(10):
                i, j = random.randint(0, m-1), random.randint(0, m-1)
                order[i], order[j] = order[j], order[i]

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_idx(family, idx):
                family.append(idx)

        # Local improvement: try to add sets that were skipped
        in_family = set(family)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)

        improved = family[:]
        for idx in candidates:
            if time.time() > deadline - 2:
                break
            if not check_violation_idx(improved, idx):
                improved.append(idx)

        if len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Strategy 3: Local improvement with "Re-insertion" on the best family
    # Try removing each set and re-adding skipped sets
    if best_family_idx:
        current_best = best_family_idx[:]
        improved = current_best[:]
        in_family = set(improved)
        candidates = [idx for idx in range(m) if idx not in in_family]
        
        # Try to improve by swapping
        for i in range(len(improved)):
            if time.time() > deadline - 5:
                break
            # Remove improved[i]
            removed = improved[i]
            improved.pop(i)
            in_family.remove(removed)
            
            # Try to add candidates
            added_any = False
            for idx in candidates:
                if time.time() > deadline - 5:
                    break
                if not check_violation_idx(improved, idx):
                    improved.append(idx)
                    in_family.add(idx)
                    added_any = True
            
            # If we added more than we removed (net gain is 1 if we added 1, 0 if we added 0? No, we removed 1)
            # We removed 1. If we added 1, size is same. If we added >1, size increases.
            # But usually we can't add more than 1 because the removed set was blocking them.
            # Actually, if we removed a set, we might be able to add multiple sets that were blocked by it.
            if len(improved) > best_size:
                best_size = len(improved)
                best_family_idx = improved[:]
            else:
                # Revert if no improvement or worse
                improved.pop() # Remove the last added one? No, this is messy.
                # Let's just revert to previous state if no gain
                if len(improved) <= len(current_best):
                    improved = current_best[:]
                    in_family = set(improved)
                # If improved is same size, we might want to keep it if it allows further improvements?
                # For simplicity, only keep if strictly better size.

    # Final local improvement on the best
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
