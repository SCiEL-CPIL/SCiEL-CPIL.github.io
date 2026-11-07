"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _is_valid(els):
    """Check if all non-empty subset sums are distinct and pairwise incomparable under divisibility."""
    n = len(els)
    sums = set()
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            s = sum(els[i] for i in combo)
            if s in sums:
                return False
            sums.add(s)
    sum_list = sorted(sums)
    m = len(sum_list)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = sum_list[i], sum_list[j]
            if b % a == 0:
                return False
    return True

def _find_valid_set(max_val=256, target_size=8, timeout=55):
    start = time.time()
    random.seed(12345)
    
    # Try decreasing sizes
    for n in range(target_size, 2, -1):
        if time.time() - start > timeout - 5:
            break
        max_attempts = 20000 if n <= 6 else 5000
        for attempt in range(max_attempts):
            if time.time() - start > timeout - 5:
                break
            # Bias towards larger numbers to reduce divisibility
            cand = random.sample(range(50, max_val + 1), n)
            if _is_valid(cand):
                return cand
    # Try even smaller
    for n in range(6, 1, -1):
        if time.time() - start > timeout - 2:
            break
        for attempt in range(10000):
            if time.time() - start > timeout - 2:
                break
            cand = random.sample(range(1, max_val + 1), n)
            if _is_valid(cand):
                return cand
    return [31, 32, 34]

def construct():
    result = _find_valid_set()
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
