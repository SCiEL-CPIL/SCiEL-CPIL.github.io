import itertools
import math
import time
import random

def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximizes the cardinality of A+B.
    """
    t_start = time.time()
    TIME_LIMIT = 85.0  # Safety margin

    def is_pairwise_coprime(sums_set):
        sums_list = list(sums_set)
        n = len(sums_list)
        for i in range(n):
            si = sums_list[i]
            if si == 1:
                continue
            for j in range(i + 1, n):
                sj = sums_list[j]
                if math.gcd(si, sj) != 1:
                    return False
        return True

    best_A = None
    best_B = None
    best_size = 0

    # Strategy 1: Targeted Random Search for Size 9
    # Focus on parity cases that allow 9 distinct sums without immediate even-even conflicts.
    # Case 1: A all odd, B all even (or vice versa). Sums all odd.
    # Case 2: Mixed parity, but ensure at most 1 even sum.
    
    odd_nums = [i for i in range(1, 129) if i % 2 != 0]
    even_nums = [i for i in range(1, 129) if i % 2 == 0]
    
    # Pre-compute some random sets to try
    random.seed(42)
    
    # Try specific structured sets first (fast)
    # These are guesses based on prime sums
    candidates = [
        ([1, 3, 11], [2, 6, 14]),
        ([1, 3, 13], [2, 6, 14]),
        ([1, 5, 11], [2, 6, 14]),
        ([1, 3, 17], [2, 4, 16]),
        ([1, 5, 17], [2, 4, 16]),
        ([1, 3, 7], [2, 4, 10]),
        ([1, 3, 7], [2, 8, 14]),
        ([1, 5, 9], [2, 4, 10]),
        ([1, 3, 11], [4, 8, 16]),
        ([1, 3, 11], [2, 10, 18]),
    ]
    
    for A, B in candidates:
        sums = {a + b for a in A for b in B}
        if is_pairwise_coprime(sums):
            if len(sums) > best_size:
                best_size = len(sums)
                best_A = A
                best_B = B
                if best_size == 9:
                    return [best_A, best_B]

    # Random Search
    # We will mix random sampling from full range and parity-restricted ranges.
    for i in range(200000):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
            
        # Alternate between parity-restricted and general
        if i % 2 == 0:
            # A odd, B even
            A = list(random.sample(odd_nums, 3))
            B = list(random.sample(even_nums, 3))
        else:
            # General
            A = list(random.sample(range(1, 129), 3))
            B = list(random.sample(range(1, 129), 3))
            
        sums = {a + b for a in A for b in B}
        
        # Quick prune: if >1 even sum, it's likely invalid unless one is 2 and others are... 
        # Actually, just check coprime. It's fast enough for k<=9.
        if is_pairwise_coprime(sums):
            if len(sums) > best_size:
                best_size = len(sums)
                best_A = A
                best_B = B
                if best_size == 9:
                    return [best_A, best_B]

    # If we didn't find 9, try a small exhaustive search for size 8/9 in small numbers
    # This catches cases where random might miss a tight cluster of small numbers.
    small_range = list(range(1, 31))
    # To keep it fast, we only iterate A, and for each A, a subset of B
    # Or just full small range if time permits
    if time.time() - t_start < TIME_LIMIT - 10:
        for A in itertools.combinations(small_range, 3):
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for B in itertools.combinations(small_range, 3):
                sums = {a + b for a in A for b in B}
                if is_pairwise_coprime(sums):
                    if len(sums) > best_size:
                        best_size = len(sums)
                        best_A = list(A)
                        best_B = list(B)
                        if best_size == 9:
                            return [best_A, best_B]

    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]
