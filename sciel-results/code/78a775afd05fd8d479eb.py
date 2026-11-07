"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

P = {'n': 256, 'maximum_size': 60}

def _check_valid_fast(s):
    """Fast check using NumPy uint64 bitsets. Returns True if valid."""
    n = len(s)
    if n == 0:
        return True
    max_sum = sum(s)
    words = (max_sum + 64) // 64
    # dp[k] is a list of uint64 words representing sums achievable with exactly k elements
    dp = [np.zeros(words, dtype=np.uint64) for _ in range(n + 1)]
    dp[0][0] = 1  # sum 0 with 0 elements
    
    for x in s:
        for k in range(n, 0, -1):
            # Shift dp[k-1] left by x bits
            src = dp[k - 1]
            dst = dp[k]
            word_shift = x // 64
            bit_shift = x % 64
            
            if bit_shift == 0:
                # Pure word shift
                for w in range(words - 1, word_shift - 1, -1):
                    dst[w] |= src[w - word_shift]
            else:
                for w in range(words - 1, word_shift - 1, -1):
                    val = src[w - word_shift] << bit_shift
                    if w - word_shift - 1 >= 0:
                        val |= src[w - word_shift - 1] >> (64 - bit_shift)
                    dst[w] |= val
    
    # Check cross-cardinality: no sum appears in two different cardinality sets
    # For each word position, check if any bit is set in two different dp[k]
    seen_mask = np.zeros(words, dtype=np.uint64)
    for k in range(n + 1):
        overlap = seen_mask & dp[k]
        if np.any(overlap != 0):
            return False
        seen_mask |= dp[k]
    return True

def _greedy_build(max_val=256, target_size=60, deadline=None):
    """Greedy: add elements one by one, preferring larger values first."""
    current = []
    used = set()
    candidates = list(range(max_val, 0, -1))
    random.shuffle(candidates)
    
    for cand in candidates:
        if len(current) >= target_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        if cand in used:
            continue
        test = current + [cand]
        test.sort()
        if _check_valid_fast(test):
            current = test
            used.add(cand)
    return current

def _try_extend(current, deadline):
    """Try to extend current set with additional elements."""
    current = list(current)
    used = set(current)
    # Try adding elements from smallest to largest, then largest to smallest
    for extra in list(range(1, 257)) + list(range(256, 0, -1)):
        if time.time() > deadline:
            break
        if extra in used:
            continue
        test = current + [extra]
        test.sort()
        if _check_valid_fast(test):
            current = test
            used.add(extra)
    return current

def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90s limit
    
    best = []
    
    # Try multiple random seeds
    for seed in range(500):
        if time.time() > deadline:
            break
        random.seed(seed)
        result = _greedy_build(max_val=256, target_size=60, deadline=deadline)
        if len(result) > len(best):
            best = result
        
        # Try to extend with smaller values
        if time.time() < deadline:
            extended = _try_extend(best, deadline)
            if len(extended) > len(best):
                best = extended
        
        if len(best) >= 30:
            # If we're doing well, try more aggressive extension
            if time.time() < deadline:
                extended = _try_extend(best, deadline)
                if len(extended) > len(best):
                    best = extended
    
    # Final validation
    if best and _check_valid_fast(best):
        return list(best)
    return [1, 2, 4, 8, 16, 32, 64, 128]  # fallback: powers of 2
# EVOLVE-BLOCK-END
