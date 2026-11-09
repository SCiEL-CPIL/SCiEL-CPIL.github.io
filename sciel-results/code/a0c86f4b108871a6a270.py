import time
from itertools import combinations

N = 24
ALL = list(range(1, 25))
L = 5354228880  # lcm(1..24)
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i

# For each element i, precompute the set of bitmasks over [1..24]\{i}
# such that the sum of reciprocals of the elements in the mask equals 1/i.
# We store as a list of (mask, size) pairs, sorted by size ascending,
# so that during validity check we can short-circuit on small subsets first.
# To keep memory reasonable, we only store subsets up to size 7.
# If a valid larger subset exists, it contains a smaller one? No, not necessarily.
# But for the reciprocal subsum property, if 1/i = sum of 1/k_j, the number of terms
# is bounded. In practice for i in [1,24], the maximal subset size that sums to 1/i
# using distinct elements from [1,24]\{i} is small (typically <= 5 or 6).
# Let's be safe and go up to size 8.

MAX_SUBSET_SIZE = 8

def _gen_subsets_by_sum(arr, max_val, max_size):
    """
    arr: list of (weight, element_index)
    Returns dict: sum -> list of bitmasks (over element indices in arr)
    Only keeps subsets up to max_size.
    """
    # result[sum] = list of (mask, size)
    result = {0: [(0, 0)]}
    for idx, (w, elem_idx) in enumerate(arr):
        if w > max_val:
            continue
        new_entries = []
        for s, entries in result.items():
            ns = s + w
            if ns > max_val:
                continue
            for mask, size in entries:
                if size >= max_size:
                    continue
                new_mask = mask | (1 << idx)
                new_entries.append((ns, new_mask, size + 1))
        # Merge new_entries into result
        for ns, new_mask, new_size in new_entries:
            if ns not in result:
                result[ns] = []
            result[ns].append((new_mask, new_size))
    return result

# DANGEROUS[i] = list of (bitmask_over_1..24, size) for subsets of [1..24]\{i}
# whose reciprocals sum to 1/i
DANGEROUS = [[] for _ in range(25)]

def _precompute_dangerous():
    for i in range(1, 25):
        target = RECIP[i]
        others = [k for k in range(1, 25) if k != i]
        
        # Build arr of (weight, local_index) where local_index is 0..len(others)-1
        arr = [(RECIP[k], k) for k in others]
        
        # Use meet-in-the-middle for efficiency
        n = len(others)
        mid = n // 2
        left_elems = others[:mid]
        right_elems = others[mid:]
        
        left_arr = [(RECIP[k], k) for k in left_elems]
        right_arr = [(RECIP[k], k) for k in right_elems]
        
        # Generate all subset sums for left and right
        left_sums = {0: [(0, 0)]}  # sum -> list of (local_mask, size)
        for idx, k in enumerate(left_elems):
            w = RECIP[k]
            if w > target:
                continue
            new_entries = []
            for s, entries in left_sums.items():
                ns = s + w
                if ns > target:
                    continue
                for lmask, lsize in entries:
                    if lsize >= MAX_SUBSET_SIZE:
                        continue
                    new_entries.append((ns, lmask | (1 << idx), lsize + 1))
            for ns, nm, nsz in new_entries:
                if ns not in left_sums:
                    left_sums[ns] = []
                left_sums[ns].append((nm, nsz))
        
        right_sums = {0: [(0, 0)]}
        for idx, k in enumerate(right_elems):
            w = RECIP[k]
            if w > target:
                continue
            new_entries = []
            for s, entries in right_sums.items():
                ns = s + w
                if ns > target:
                    continue
                for rmask, rsize in entries:
                    if rsize >= MAX_SUBSET_SIZE:
                        continue
                    new_entries.append((ns, rmask | (1 << idx), rsize + 1))
            for ns, nm, nsz in new_entries:
                if ns not in right_sums:
                    right_sums[ns] = []
                right_sums[ns].append((nm, nsz))
        
        found = {}  # global_mask -> size
        for ls, left_entries in left_sums.items():
            rs_needed = target - ls
            if rs_needed < 0 or rs_needed not in right_sums:
                continue
            for lmask, lsize in left_entries:
                for rmask, rsize in right_sums[rs_needed]:
                    total_size = lsize + rsize
                    if total_size > MAX_SUBSET_SIZE:
                        continue
                    if total_size == 0:
                        continue
                    # Convert local masks to global bitmask
                    gmask = 0
                    for idx in range(len(left_elems)):
                        if lmask & (1 << idx):
                            gmask |= (1 << (left_elems[idx] - 1))
                    for idx in range(len(right_elems)):
                        if rmask & (1 << idx):
                            gmask |= (1 << (right_elems[idx] - 1))
                    if gmask not in found or total_size < found[gmask]:
                        found[gmask] = total_size
        
        DANGEROUS[i] = sorted([(m, s) for m, s in found.items()], key=lambda x: x[1])

_precompute_dangerous()

def is_valid_mask(mask):
    """Check if the set represented by mask is valid (no reciprocal subsum)."""
    for i in range(1, 25):
        if mask & (1 << (i - 1)):
            for dm, dsz in DANGEROUS[i]:
                if (dm & mask) == dm:
                    return False
    return True

def mask_to_list(mask):
    return [i for i in range(1, 25) if mask & (1 << (i - 1))]

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s kill
    
    def now():
        return time.time()
    
    # Initial seed: upper half {13..24}
    seed_mask = 0
    for i in range(13, 25):
        seed_mask |= (1 << (i - 1))
    
    best_mask = seed_mask if is_valid_mask(seed_mask) else (1 << 23)
    best_size = bin(best_mask).count('1')
    
    # Greedy add in various orders
    def greedy_add(order, start_mask=None):
        mask = start_mask if start_mask is not None else 0
        for x in order:
            if now() > deadline:
                break
            bit = 1 << (x - 1)
            if mask & bit:
                continue
            trial = mask | bit
            if is_valid_mask(trial):
                mask = trial
        return mask
    
    # Multiple deterministic orderings
    orderings = [
        ALL[:],
        ALL[::-1],
        sorted(ALL, key=lambda x: RECIP[x]),
        sorted(ALL, key=lambda x: RECIP[x], reverse=True),
        sorted(ALL, key=lambda x: x % 7),
        sorted(ALL, key=lambda x: x % 5),
        sorted(ALL, key=lambda x: x % 11),
    ]
    
    for o in orderings:
        if now() > deadline:
            break
        res = greedy_add(o)
        sz = bin(res).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = res
    
    # Local search: try adding missing elements, and swaps
    def local_search(mask, max_iters=300):
        iters = 0
        improved = True
        while improved and iters < max_iters and now() < deadline:
            iters += 1
            improved = False
            
            # Try additions
            missing = [x for x in ALL if not (mask & (1 << (x - 1)))]
            for x in missing:
                if now() > deadline:
                    break
                trial = mask | (1 << (x - 1))
                if is_valid_mask(trial):
                    mask = trial
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove a, add b
            in_set = [x for x in ALL if mask & (1 << (x - 1))]
            if not in_set or not missing:
                break
            for a in in_set:
                if now() > deadline:
                    break
                for b in missing:
                    trial = (mask & ~(1 << (a - 1))) | (1 << (b - 1))
                    if is_valid_mask(trial):
                        mask = trial
                        improved = True
                        break
                if improved:
                    break
        
        return mask
    
    # Apply local search to best
    if best_mask:
        ls = local_search(best_mask, max_iters=200)
        sz = bin(ls).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = ls
    
    # Multi-start with random perturbations
    import random
    rng = random.Random(12345)
    
    for _ in range(50):
        if now() > deadline:
            break
        # Start from a random subset of size around best_size
        target_size = max(2, best_size - 3 + rng.randint(0, 3))
        target_size = min(target_size, 24)
        rand_set = rng.sample(ALL, target_size)
        rand_mask = 0
        for x in rand_set:
            rand_mask |= (1 << (x - 1))
        
        # Repair: remove violators
        changed = True
        while changed and now() < deadline:
            changed = False
            for i in range(1, 25):
                if rand_mask & (1 << (i - 1)):
                    for dm, dsz in DANGEROUS[i]:
                        if (dm & rand_mask) == dm:
                            # Remove one element from the dangerous subset
                            # Pick the largest element in the dangerous subset
                            elems = [k for k in range(1, 25) if dm & (1 << (k - 1))]
                            if elems:
                                rem = max(elems)
                                rand_mask &= ~(1 << (rem - 1))
                                changed = True
                                break
                if changed:
                    break
        
        # Greedy add
        order = ALL[:]
        rng.shuffle(order)
        rand_mask = greedy_add(order, start_mask=rand_mask)
        
        # Local search
        rand_mask = local_search(rand_mask, max_iters=100)
        
        sz = bin(rand_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = rand_mask
    
    # Final polish: try all single swaps from best
    if best_mask and now() < deadline:
        in_set = [x for x in ALL if best_mask & (1 << (x - 1))]
        missing = [x for x in ALL if not (best_mask & (1 << (x - 1)))]
        for a in in_set:
            if now() > deadline:
                break
            for b in missing:
                trial = (best_mask & ~(1 << (a - 1))) | (1 << (b - 1))
                if is_valid_mask(trial):
                    sz = bin(trial).count('1')
                    if sz > best_size:
                        best_size = sz
                        best_mask = trial
                        in_set = [x for x in ALL if best_mask & (1 << (x - 1))]
                        missing = [x for x in ALL if not (best_mask & (1 << (x - 1)))]
    
    # Final validation
    if not is_valid_mask(best_mask):
        best_mask = 1 << 23  # {24}
    
    return mask_to_list(best_mask)

if __name__ == "__main__":
    print(construct())
