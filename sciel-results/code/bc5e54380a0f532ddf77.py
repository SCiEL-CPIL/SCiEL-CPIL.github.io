import time
import random
from itertools import combinations
from collections import defaultdict

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

    # Precompute for each pair (i,j) and intersection value v,
    # the set of all k such that inter[i][k] == v and inter[j][k] == v
    # This lets us quickly check if adding k would create a sunflower with any existing pair

    # For each pair (i,j), group potential third elements by the intersection value
    # sunflower_witness[i][j][v] = set of k != i,j such that inter[i][k]==v and inter[j][k]==v
    # But this is too much memory. Instead, for each pair (i,j), precompute a dict:
    # pair_conflict[i][j] = dict mapping v -> count of k (k!=i,j) with inter[i][k]==v and inter[j][k]==v

    # Actually, let's precompute for each triple (i,j,k) whether it's a sunflower
    # and store for each i,j the set of k that form a sunflower with i,j

    # sunflower_with[i][j] = set of k such that inter[i][j] == inter[i][k] == inter[j][k]
    sunflower_with = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            v = inter[i][j]
            si = sunflower_with[i]
            sj = sunflower_with[j]
            for k in range(m):
                if k == i or k == j:
                    continue
                if inter[i][k] == v and inter[j][k] == v:
                    si.add(k)
                    sj.add(k)

    def check_violation_idx(family_set, idx):
        """
        Check if adding idx to family creates a sunflower violation.
        family_set is a set of indices.
        """
        for a in family_set:
            # Check if there exists b in family_set such that a,b,idx form a sunflower
            # i.e., inter[a][b] == inter[a][idx] == inter[b][idx]
            # We can check: for each a in family, is idx in sunflower_with[a] for some b?
            # Actually: idx forms sunflower with a and b iff idx in sunflower_with[a] AND b in sunflower_with[idx] AND a in sunflower_with[b]
            # Simpler: for each a in family, check if there's b in family such that inter[a][b] == inter[a][idx] == inter[b][idx]
            # Use precomputed: for pair (a, idx), the conflict value is inter[a][idx]. 
            # We need b in family such that inter[a][b] == inter[a][idx] and inter[b][idx] == inter[a][idx]
            # i.e., b in sunflower_with[a] ∩ sunflower_with[idx] ∩ family
            v = inter[a][idx]
            # b must be in family, b != a, b != idx
            # b must satisfy: inter[a][b] == v and inter[b][idx] == v
            # This is equivalent to: b in sunflower_with[a] and b in sunflower_with[idx]
            # But we can just iterate over family and check
            for b in family_set:
                if b == a:
                    continue
                if inter[a][b] == v and inter[b][idx] == v:
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
    family_set = set()
    for idx in range(m):
        if time.time() > deadline:
            break
        if not check_violation_idx(family_set, idx):
            family.append(idx)
            family_set.add(idx)

    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles with greedy
    random.seed(12345)
    max_attempts = 50

    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break

        order = list(range(m))
        random.shuffle(order)

        family = []
        family_set = set()
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_idx(family_set, idx):
                family.append(idx)
                family_set.add(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Strategy 3: Local improvement with additions and swaps
    def local_improve(family_indices, time_budget):
        """Try to improve family by additions and swaps."""
        improved = family_indices[:]
        in_family = set(improved)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)
        
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            
            # Try additions
            new_candidates = []
            for idx in candidates:
                if time.time() > time_budget:
                    break
                if not check_violation_idx(in_family, idx):
                    improved.append(idx)
                    in_family.add(idx)
                    changed = True
                else:
                    new_candidates.append(idx)
            candidates = new_candidates
            
            # Try swaps: remove one, add one
            if time.time() > time_budget - 1:
                break
            for i in range(len(improved)):
                if time.time() > time_budget:
                    break
                removed = improved[i]
                for idx in candidates:
                    if time.time() > time_budget:
                        break
                    # Check if removing improved[i] and adding idx is valid
                    # Build new family without removed, with idx
                    new_family_set = in_family - {removed} | {idx}
                    # Check if idx can be added to (in_family - {removed})
                    if not check_violation_idx(new_family_set - {idx}, idx):
                        improved[i] = idx
                        in_family.remove(removed)
                        in_family.add(idx)
                        candidates.remove(idx)
                        changed = True
                        break  # Restart after successful swap

        return improved

    # Apply local improvement to best so far
    time_budget = deadline - 10
    improved = local_improve(best_family_idx, time_budget)
    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Strategy 4: More random restarts with different seeds
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579, 24680, 135791]:
        if time.time() > deadline - 12:
            break
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)

        family = []
        family_set = set()
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_idx(family_set, idx):
                family.append(idx)
                family_set.add(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Final local improvement on the best
    if time.time() < deadline - 3:
        improved = local_improve(best_family_idx, deadline - 1)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    # Convert back to sorted lists of 1..16
    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]

    return result
