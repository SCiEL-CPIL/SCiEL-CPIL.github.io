import time
import sys

sys.setrecursionlimit(10000)

MAX_N = 128

# Generate smooth numbers up to MAX_N
smooth = []
for x in range(1, MAX_N + 1):
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    if y == 1:
        smooth.append(x)

smooth_set = set(smooth)
smooth_list = sorted(smooth)
n_smooth = len(smooth_list)
smooth_to_idx = {v: i for i, v in enumerate(smooth_list)}

# Precompute divisibility: div_mask[i] is a bitmask of indices j where smooth[i] divides smooth[j] or smooth[j] divides smooth[i]
# Actually, for antichain: we need no pair where one divides the other.
# Precompute: for each i, the set of j != i such that smooth[i] | smooth[j] or smooth[j] | smooth[i]
incompatible = [0] * n_smooth  # bitmask
for i in range(n_smooth):
    for j in range(i + 1, n_smooth):
        a, b = smooth_list[i], smooth_list[j]
        if b % a == 0 or a % b == 0:
            incompatible[i] |= (1 << j)
            incompatible[j] |= (1 << i)

# For each smooth number index, precompute its "conflict mask"
# conflict_mask[i] = incompatible[i] (bitmask of indices that conflict with i)

best = [None] * (MAX_N + 1)
start_time = time.time()
TIME_LIMIT = 75.0


def check_antichain_fast(indices):
    """Check if the given indices form an antichain using precomputed masks."""
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            a, b = indices[i], indices[j]
            if incompatible[a] & (1 << b):
                return False
    return True


def dfs_for_k(n, k, max_idx, current, current_sum, forbidden_mask):
    """
    DFS to find k smooth numbers summing to n, forming an antichain.
    current: list of chosen indices
    current_sum: sum of chosen values
    forbidden_mask: bitmask of indices that cannot be chosen (conflict with any chosen)
    Returns list of values or None.
    """
    if len(current) == k:
        if current_sum == n:
            return [smooth_list[i] for i in current]
        return None
    
    remaining = k - len(current)
    needed = n - current_sum
    
    # Pruning: if even taking the largest remaining values can't reach needed, or
    # the smallest remaining values already exceed needed
    for idx in range(max_idx, n_smooth):
        v = smooth_list[idx]
        if v > needed:
            break
        # Check if this index is forbidden
        if forbidden_mask & (1 << idx):
            continue
        # Pruning: can we still reach the target with 'remaining' more terms?
        # The maximum we can add with 'remaining' terms (all distinct, > current idx)
        # Simple bound: if v + sum of next (remaining-1) largest available > needed, skip
        # Better: if even the largest possible sum from remaining slots < needed, break
        # The smallest we can add for the remaining (remaining-1) terms
        if idx + (remaining - 1) >= n_smooth:
            # Not enough indices left
            continue
        
        # Try this index
        new_forbidden = forbidden_mask | incompatible[idx]
        # Also need to exclude already-chosen indices (they're already in forbidden via their own masks? No)
        # Actually forbidden_mask already contains conflicts. We just need to make sure we don't pick duplicates.
        # Since we iterate idx in increasing order and use max_idx, no duplicates.
        
        result = dfs_for_k(n, k, idx + 1, current + [idx], current_sum + v, new_forbidden)
        if result is not None:
            return result
    
    return None


def find_minimal_representation(n):
    """Find the minimal-k antichain representation of n using smooth numbers."""
    if time.time() - start_time > TIME_LIMIT:
        return None
    
    # k=1
    if n in smooth_set:
        return [n]
    
    # k=2
    for i in range(n_smooth):
        a = smooth_list[i]
        if a >= n:
            break
        b = n - a
        if b <= a:
            continue
        if b in smooth_set:
            j = smooth_to_idx[b]
            if not (incompatible[i] & (1 << j)):
                return [a, b]
    
    # k=3
    for i in range(n_smooth):
        a = smooth_list[i]
        if a >= n:
            break
        for j in range(i + 1, n_smooth):
            b = smooth_list[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b:
                continue
            if c not in smooth_set:
                continue
            k_idx = smooth_to_idx[c]
            # Check antichain: no pair divides another
            if (incompatible[i] & (1 << j)) or (incompatible[i] & (1 << k_idx)) or (incompatible[j] & (1 << k_idx)):
                continue
            return [a, b, c]
    
    # k=4
    for i in range(n_smooth):
        a = smooth_list[i]
        if a >= n:
            break
        for j in range(i + 1, n_smooth):
            b = smooth_list[j]
            if a + b >= n:
                break
            for k_idx in range(j + 1, n_smooth):
                c = smooth_list[k_idx]
                if a + b + c >= n:
                    break
                d = n - a - b - c
                if d <= c:
                    continue
                if d not in smooth_set:
                    continue
                l_idx = smooth_to_idx[d]
                # Check antichain for all 6 pairs
                if (incompatible[i] & (1 << j)) or (incompatible[i] & (1 << k_idx)) or (incompatible[i] & (1 << l_idx)):
                    continue
                if (incompatible[j] & (1 << k_idx)) or (incompatible[j] & (1 << l_idx)):
                    continue
                if (incompatible[k_idx] & (1 << l_idx)):
                    continue
                return [a, b, c, d]
    
    # k=5 and above: use DFS
    for k in range(5, 31):
        if time.time() - start_time > TIME_LIMIT:
            break
        result = dfs_for_k(n, k, 0, [], 0, 0)
        if result is not None:
            return result
    
    return None


# Main search
for n in range(1, MAX_N + 1):
    if time.time() - start_time > TIME_LIMIT:
        break
    rep = find_minimal_representation(n)
    if rep is not None:
        best[n] = rep

# Safety: fill any missing with a slower brute force
for n in range(1, MAX_N + 1):
    if best[n] is None:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        # Slower fallback
        from itertools import combinations
        candidates = [x for x in smooth_list if x <= n]
        found = False
        for k in range(1, 31):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            for combo in combinations(candidates, k):
                if sum(combo) == n:
                    # Check antichain
                    ok = True
                    for i in range(len(combo)):
                        for j in range(i + 1, len(combo)):
                            a, b = combo[i], combo[j]
                            if b % a == 0 or a % b == 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        best[n] = list(combo)
                        found = True
                        break
            if found:
                break


def construct():
    return [best[i] for i in range(1, MAX_N + 1)]
