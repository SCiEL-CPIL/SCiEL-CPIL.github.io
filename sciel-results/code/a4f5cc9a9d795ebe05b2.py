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

    odd_nums = [i for i in range(1, 129) if i % 2 != 0]
    even_nums = [i for i in range(1, 129) if i % 2 == 0]
    
    random.seed(42)
    
    # Try specific structured sets first (fast)
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

    # Strategy: Exhaustive search with aggressive pruning
    # Key insight: For 9 distinct sums that are pairwise coprime,
    # at most one can be even (and if so, it must be 2).
    # So either all 9 sums are odd, or exactly one is 2 and 8 are odd.
    
    # Phase 1: Search for all-odd sums (A all odd, B all even OR A all even, B all odd)
    # This gives 9 odd sums, all must be pairwise coprime.
    
    def try_all_odd_sums():
        nonlocal best_A, best_B, best_size
        # A odd, B even -> all sums odd
        for a1 in odd_nums:
            if time.time() - t_start > TIME_LIMIT - 10:
                return
            for a2 in odd_nums:
                if a2 <= a1:
                    continue
                for a3 in odd_nums:
                    if a3 <= a2:
                        continue
                    A = (a1, a2, a3)
                    for b1 in even_nums:
                        if time.time() - t_start > TIME_LIMIT - 10:
                            return
                        for b2 in even_nums:
                            if b2 <= b1:
                                continue
                            for b3 in even_nums:
                                if b3 <= b2:
                                    continue
                                B = (b1, b2, b3)
                                sums = {A[0]+B[0], A[0]+B[1], A[0]+B[2],
                                       A[1]+B[0], A[1]+B[1], A[1]+B[2],
                                       A[2]+B[0], A[2]+B[1], A[2]+B[2]}
                                if len(sums) < best_size:
                                    continue
                                if is_pairwise_coprime(sums):
                                    if len(sums) > best_size:
                                        best_size = len(sums)
                                        best_A = list(A)
                                        best_B = list(B)
                                        if best_size == 9:
                                            return
    try_all_odd_sums()
    
    if best_size == 9:
        return [best_A, best_B]

    # Phase 2: Random search with parity constraints
    for i in range(500000):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
            
        if i % 2 == 0:
            # A odd, B even
            A = list(random.sample(odd_nums, 3))
            B = list(random.sample(even_nums, 3))
        else:
            # A even, B odd
            A = list(random.sample(even_nums, 3))
            B = list(random.sample(odd_nums, 3))
            
        sums = {a + b for a in A for b in B}
        if len(sums) < best_size:
            continue
        if is_pairwise_coprime(sums):
            if len(sums) > best_size:
                best_size = len(sums)
                best_A = A
                best_B = B
                if best_size == 9:
                    return [best_A, best_B]

    # Phase 3: Focused exhaustive search in small range for any parity pattern
    small_range = list(range(1, 25))
    if time.time() - t_start < TIME_LIMIT - 10:
        for A in itertools.combinations(small_range, 3):
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for B in itertools.combinations(small_range, 3):
                sums = {a + b for a in A for b in B}
                if len(sums) < best_size:
                    continue
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
