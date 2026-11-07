import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 78.0

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
P_INDEX = {p: i for i, p in enumerate(PRIMES)}

def get_vector(n):
    v = [0] * 12
    for p in PRIMES:
        while n % p == 0:
            v[P_INDEX[p]] += 1
            n //= p
    if n > 1:
        pass
    return tuple(v)

# Pre-compute vectors for 2..40
VECTORS = {}
for i in range(2, 41):
    VECTORS[i] = get_vector(i)

# Maximum exponent per prime in any subset sum of up to 18 elements from 2..40
# For prime 2: max exponent in a single number is 5 (32), so max in 18 numbers is 90
# But we only need to track up to what's possible. Let's compute max possible exponents.
# Actually, for the bitset approach, we need to know the max possible exponent sum.
# Max exponent of 2 in any number 2..40: 32 = 2^5, so 5
# Max exponent of 3: 27 = 3^3, so 3
# Max exponent of 5: 25 = 5^2, so 2
# Max exponent of 7: 7^1, so 1
# For primes >= 11: max exponent is 1 (since 11^2 = 121 > 40)
# So for a set of k elements, max exponent of prime p is k * max_exp_p
# For k=18: max_exp_2 = 90, max_exp_3 = 54, max_exp_5 = 36, max_exp_7 = 18, others = 18

# We'll use a flat index scheme. Let's define the dimensions:
# dim[0] = 91 (exponents 0..90 for prime 2)
# dim[1] = 55 (exponents 0..54 for prime 3)
# dim[2] = 37 (exponents 0..36 for prime 5)
# dim[3] = 19 (exponents 0..18 for prime 7)
# dim[4..11] = 19 each (exponents 0..18 for primes 11..37)

DIMS = [91, 55, 37, 19, 19, 19, 19, 19, 19, 19, 19, 19]
TOTAL_DIM = 1
for d in DIMS:
    TOTAL_DIM *= d

# Precompute strides for flat indexing
STRIDES = [1] * 12
for i in range(11, -1, -1):
    STRIDES[i] = STRIDES[i+1] * DIMS[i+1] if i < 11 else 1

def vector_to_flat(v):
    idx = 0
    for i in range(12):
        idx += v[i] * STRIDES[i]
    return idx

def flat_to_vector(idx):
    v = [0] * 12
    for i in range(12):
        v[i] = (idx // STRIDES[i]) % DIMS[i]
    return tuple(v)

# Precompute flat indices for all vectors 2..40
VECTORS_FLAT = {}
for i in range(2, 41):
    VECTORS_FLAT[i] = vector_to_flat(VECTORS[i])

# Precompute all possible flat indices and their vectors
# This is too large: 91*55*37*19^8 is huge. We won't precompute all.

def _subset_sums_set(nums):
    """Return the set of all subset sums as flat indices."""
    sums = {0}  # 0 is the flat index for all-zero vector
    for x in nums:
        v = VECTORS_FLAT[x]
        new_sums = set()
        for s in sums:
            new_s = s + v
            new_sums.add(new_s)
        sums = sums | new_sums
    return sums

def _can_add(sums, x):
    """True iff adding x keeps all subset sums distinct."""
    v = VECTORS_FLAT[x]
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def _try_add(selected, sums, x):
    """Try to add x to selected; returns (success, new_selected, new_sums)."""
    if not _can_add(sums, x):
        return False, selected, sums
    v = VECTORS_FLAT[x]
    new_sums = set(sums)
    for s in sums:
        new_sums.add(s + v)
    new_selected = selected + [x]
    return True, new_selected, new_sums

def _construct_initial_set():
    """
    Construct a high-quality initial set.
    Primes are always good.
    Add composites that are 'far' from others.
    """
    # Start with primes
    selected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    sums = _subset_sums_set(selected)
    
    # Try adding composites greedily
    candidates = [i for i in range(2, 41) if i not in selected]
    for x in candidates:
        if len(selected) >= 18:
            break
        ok, selected, sums = _try_add(selected, sums, x)
        if ok:
            pass
    return selected, sums

def _local_search(selected, sums, deadline, max_rounds=20):
    """Local search: try swapping elements to improve the set size."""
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        
        # 1. Try adding any candidate
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sel, new_sums = _try_add(selected, sums, x)
            if ok:
                selected = new_sel
                sums = new_sums
                improved = True
                used = set(selected)
                candidates = [c for c in range(2, 41) if c not in used]
                if len(selected) >= 18:
                    break
        
        if not improved:
            # 2. Try single swaps: remove one, add one different element
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_sums = _subset_sums_set(remaining)
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    ok1, sel1, sums1 = _try_add(remaining, rem_sums, x)
                    if ok1 and x != removed:
                        selected = sel1
                        sums = sums1
                        improved = True
                        break
                if improved:
                    break
            
        if not improved:
            # 3. Try 2-swaps with a limit
            if time.time() < deadline - 5.0:
                n = len(selected)
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, n):
                        if time.time() > deadline:
                            break
                        remaining = [selected[k] for k in range(n) if k != i and k != j]
                        rem_sums = _subset_sums_set(remaining)
                        for x in candidates:
                            if time.time() > deadline:
                                break
                            if x == selected[i] or x == selected[j]:
                                continue
                            ok1, sel1, sums1 = _try_add(remaining, rem_sums, x)
                            if ok1:
                                for y in candidates:
                                    if time.time() > deadline:
                                        break
                                    if y == x or y == selected[i] or y == selected[j]:
                                        continue
                                    ok2, sel2, sums2 = _try_add(sel1, sums1, y)
                                    if ok2:
                                        selected = sel2
                                        sums = sums2
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
        if not improved:
            break
            
    return selected, sums

def construct():
    if time.time() > _DEADLINE:
        return [2]

    best_selected = None
    best_len = 0

    # Phase 1: Strong heuristic set + local search
    selected, sums = _construct_initial_set()
    if len(selected) < 18 and time.time() < _DEADLINE:
        selected, sums = _local_search(selected, sums, _DEADLINE)

    # Final greedy fill
    if time.time() < _DEADLINE and len(selected) < 18:
        used = set(selected)
        for x in range(2, 41):
            if x in used:
                continue
            if len(selected) >= 18:
                break
            ok, selected, sums = _try_add(selected, sums, x)
            if ok:
                used.add(x)

    if len(selected) > best_len:
        best_len = len(selected)
        best_selected = list(selected)

    # Phase 2: Random restarts
    rng = random.Random(42)
    remaining_time = _DEADLINE - time.time()
    num_restarts = max(1, int(remaining_time / 2.0))
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 2.0:
            break
        
        # Random initial set
        rsel = []
        rsums = {0}
        candidates = list(range(2, 41))
        rng.shuffle(candidates)
        for x in candidates:
            if len(rsel) >= 18:
                break
            if _can_add(rsums, x):
                rsel.append(x)
                v = VECTORS_FLAT[x]
                new_sums = set(rsums)
                for s in rsums:
                    new_sums.add(s + v)
                rsums = new_sums
        
        # Local search on this random start
        if time.time() < _DEADLINE - 1.0:
            rsel, rsums = _local_search(rsel, rsums, _DEADLINE - 1.0, max_rounds=5)
        
        # Greedy fill
        if time.time() < _DEADLINE - 0.5 and len(rsel) < 18:
            used = set(rsel)
            for x in range(2, 41):
                if x in used:
                    continue
                if len(rsel) >= 18:
                    break
                ok, rsel, rsums = _try_add(rsel, rsums, x)
                if ok:
                    used.add(x)
        
        if len(rsel) > best_len:
            best_len = len(rsel)
            best_selected = list(rsel)
            if best_len >= 18:
                break

    # Ensure validity
    selected = best_selected
    if not selected:
        selected = [2]
    
    # Final check
    final_sums = _subset_sums_set(selected)
    if len(final_sums) != 2 ** len(selected):
        selected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return selected
