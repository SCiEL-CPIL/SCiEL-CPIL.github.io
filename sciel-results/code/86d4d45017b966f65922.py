"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = time.time() + 80  # 80s safety margin before 90s kill

def _is_sidon(subset):
    """Check if subset has all unordered pair sums unique including diagonals (2*x)."""
    sums = set()
    for i, a in enumerate(subset):
        for j in range(i, len(subset)):
            s = a + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(arr):
    """Find largest Sidon subset size of arr (arr is a sorted list of ints)."""
    n = len(arr)
    # Try sizes from n down to 1
    for size in range(n, 0, -1):
        for combo in combinations(range(n), size):
            sub = [arr[i] for i in combo]
            if _is_sidon(sub):
                return size
    return 0

def _max_sidon_size_fast(arr, upper_hint=None):
    """Faster: check from upper_hint down, with early termination."""
    n = len(arr)
    start = min(n, upper_hint) if upper_hint else n
    for size in range(start, 0, -1):
        found = False
        for combo in combinations(range(n), size):
            sub = [arr[i] for i in combo]
            if _is_sidon(sub):
                found = True
                break
        if found:
            return size
    return 0

def _collision_score(arr):
    """Higher score = more sum collisions = harder to extract large Sidon subset."""
    # Count frequency of each pair sum (including diagonals)
    from collections import Counter
    sums = Counter()
    for i, a in enumerate(arr):
        for j in range(i, len(arr)):
            sums[a + arr[j]] += 1
    # Score: sum of (freq-1) for each sum that appears more than once
    # This counts the number of "excess" collisions
    score = sum(max(0, f - 1) for f in sums.values())
    return score

def _greedy_build(base, target_n=12, lo=0, hi=128):
    """Greedily extend base set to target_n, maximizing collision score."""
    current = list(base)
    current_set = set(current)
    for _ in range(target_n - len(current)):
        best_elem = None
        best_score = -1
        for x in range(lo, hi + 1):
            if x in current_set:
                continue
            new_arr = current + [x]
            new_arr.sort()
            s = _collision_score(new_arr)
            if s > best_score:
                best_score = s
                best_elem = x
        if best_elem is None:
            break
        current.append(best_elem)
        current_set.add(best_elem)
        current.sort()
    return current

def _local_search(arr, max_iter=5000):
    """Local search: swap elements to reduce max Sidon size."""
    n = len(arr)
    arr = list(arr)
    arr.sort()
    current_ms = _max_sidon_size(arr)
    best_arr = list(arr)
    best_ms = current_ms
    
    rng = random.Random(42)
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        # Try swapping one element
        i = rng.randrange(n)
        old_val = arr[i]
        # Try replacing with a random value
        new_val = rng.randint(0, 128)
        if new_val == old_val:
            continue
        if new_val in arr:
            continue
        arr[i] = new_val
        arr.sort()
        ms = _max_sidon_size(arr)
        if ms < current_ms:
            current_ms = ms
            best_arr = list(arr)
            best_ms = ms
            if ms <= 4:
                break
        else:
            # Revert
            arr.remove(new_val)
            arr.append(old_val)
            arr.sort()
    
    return best_arr, best_ms

def construct():
    start_time = time.time()
    
    # Initial base: use a known small Sidon set as seed for collision building
    # Start with a spread-out set to cover the range
    base = [0, 16, 32, 48, 64, 80, 96, 112, 120, 124, 126, 128]
    
    # Greedy build from a few different seeds
    candidates = []
    
    # Seed 1: arithmetic progression-like
    c1 = _greedy_build([0, 20, 40, 60, 80, 100], target_n=12)
    candidates.append(c1)
    
    # Seed 2: clustered
    c2 = _greedy_build([0, 1, 2, 3, 4, 5], target_n=12)
    candidates.append(c2)
    
    # Seed 3: mixed
    c3 = _greedy_build([0, 5, 10, 50, 100, 128], target_n=12)
    candidates.append(c3)
    
    # Seed 4: dense in middle
    c4 = _greedy_build([30, 35, 40, 45, 50, 55], target_n=12)
    candidates.append(c4)
    
    best_candidate = None
    best_ms = 999
    
    for cand in candidates:
        if len(cand) != 12:
            continue
        ms = _max_sidon_size(cand)
        if ms < best_ms:
            best_ms = ms
            best_candidate = cand
    
    # Local search on best candidate
    if best_candidate and time.time() < DEADLINE - 10:
        refined, ms_ref = _local_search(best_candidate, max_iter=2000)
        if ms_ref < best_ms:
            best_candidate = refined
            best_ms = ms_ref
    
    # If still > 5, try more aggressive construction
    if best_ms > 5 and time.time() < DEADLINE - 15:
        # Try: force many pairs to share sums
        # Use structure: pairs (a_i, b_i) where a_i + b_i = constant for many i
        # E.g., symmetric around 64: (0,128),(1,127),(2,126),... each pair sums to 128
        # But we need 12 elements, so 6 pairs each summing to 128
        sym = [0, 128, 1, 127, 2, 126, 3, 125, 4, 124, 5, 123]
        sym.sort()
        ms_sym = _max_sidon_size(sym)
        if ms_sym < best_ms:
            best_candidate = sym
            best_ms = ms_sym
        
        # Local search on symmetric set
        if time.time() < DEADLINE - 10:
            refined2, ms_r2 = _local_search(sym, max_iter=3000)
            if ms_r2 < best_ms:
                best_candidate = refined2
                best_ms = ms_r2
    
    # Final: ensure exactly 12 distinct ints in [0,128]
    result = sorted(set(best_candidate))
    if len(result) != 12:
        # Fallback
        result = list(range(12))
    
    return result
# EVOLVE-BLOCK-END
