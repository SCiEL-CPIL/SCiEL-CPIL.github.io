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
        # Should not happen for n <= 40
        pass
    return tuple(v)

# Pre-compute vectors for 2..40
VECTORS = {}
for i in range(2, 41):
    VECTORS[i] = get_vector(i)

def _subset_sums_set(nums):
    """Return the set of all subset sums of vectors of nums."""
    sums = {tuple([0]*12)}
    for x in nums:
        v = VECTORS[x]
        add = set()
        for s in sums:
            # Add v to s
            new_s = tuple(s[i] + v[i] for i in range(12))
            add.add(new_s)
        sums = sums | add
    return sums

def _can_add(sums, x):
    """True iff adding x keeps all subset sums distinct."""
    v = VECTORS[x]
    for s in sums:
        new_s = tuple(s[i] + v[i] for i in range(12))
        if new_s in sums:
            return False
    return True

def _try_add(selected, sums, x):
    """Try to add x to selected; returns (success, new_selected, new_sums)."""
    if not _can_add(sums, x):
        return False, selected, sums
    v = VECTORS[x]
    new_sums = set(sums)
    for s in sums:
        new_s = tuple(s[i] + v[i] for i in range(12))
        new_sums.add(new_s)
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
    # Sort by 'magnitude' or something?
    # Just try all
    for x in candidates:
        if len(selected) >= 18:
            break
        ok, selected, sums = _try_add(selected, sums, x)
        if ok:
            pass # Added
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
            # 3. Try 2-swaps: remove two, add two
            # This is expensive: O(n^2 * m^2)
            # n=12, m=27. 144 * 729 ~ 100k checks.
            # Each check is O(2^12) ~ 4k ops. Total 400M ops. Too slow in Python.
            # We need to be smarter or limit this.
            
            # Alternative: Random restarts might be better than heavy local search.
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
        rsums = {tuple([0]*12)}
        candidates = list(range(2, 41))
        rng.shuffle(candidates)
        for x in candidates:
            if len(rsel) >= 18:
                break
            if _can_add(rsums, x):
                rsel.append(x)
                v = VECTORS[x]
                new_sums = set(rsums)
                for s in rsums:
                    new_s = tuple(s[i] + v[i] for i in range(12))
                    new_sums.add(new_s)
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
