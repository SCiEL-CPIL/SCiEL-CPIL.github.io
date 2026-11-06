"""Search for optimal pairwise-coprime sumsets."""
import math
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def pairwise_coprime(values):
        """Check if all distinct values in list are pairwise coprime."""
        vals = list(set(values))
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        """Return (valid, num_distinct) for the sumset."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if pairwise_coprime(distinct):
            return True, len(distinct)
        return False, 0
    
    # Start with the known solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # |A+B| = 5
    
    # Brute force search: try all combinations of 3 distinct integers from [1,128]
    # That's C(128,3)^2 which is ~1.1e12 — too large for full brute force.
    # Strategy: 
    # 1. First, try to find 9 distinct pairwise-coprime sums
    # 2. Then 8, then 7, etc.
    # 
    # Key insight: For 9 distinct sums, we need a 3x3 grid of pairwise coprime values.
    # 
    # Let's use a smarter approach:
    # - Generate candidate primes (since coprime numbers are often primes)
    # - Try to find 3x3 grids where all 9 entries are distinct primes (or pairwise coprime)
    
    # First, let's check if we can get 9 distinct pairwise coprime sums
    # using a structured search
    
    # Generate primes up to 256 (max sum is 128+128=256)
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, 257) if is_prime(p)]
    
    # Also include 1 (coprime to everything)
    # And prime powers (p^k) are coprime to q^m for distinct primes p,q
    # But for simplicity, let's focus on primes first
    
    # Strategy: For a 3x3 sum grid A+B, we have:
    # a1+b1, a1+b2, a1+b3
    # a2+b1, a2+b2, a2+b3
    # a3+b1, a3+b2, a3+b3
    # 
    # If we fix A = [a1, a2, a3] and B = [b1, b2, b3],
    # the sums form a structured set.
    # 
    # Key property: (a1+b2) - (a1+b1) = b2-b1, so differences within rows are constant.
    # Similarly for columns.
    # 
    # Let's try: fix B, then for each B, try all A and check.
    # But C(128,3) = 341,376 for each set, so C(128,3)^2 is too large.
    # 
    # Better: iterate over smaller ranges or use heuristics.
    # 
    # Let's try: for B in a reasonable range, for A in a reasonable range, check.
    # Start with small values and expand.
    
    # Phase 1: Try to find 9 distinct pairwise coprime sums
    # Use a limited search space first
    
    search_range = 128
    found_9 = False
    
    # Try structured approach: use primes for the sums
    # If all 9 sums are distinct primes, they're automatically pairwise coprime.
    # 
    # We need a1+b1, a1+b2, a1+b3, a2+b1, ... all to be distinct primes.
    # 
    # Let's parameterize: let d1 = a2-a1, d2 = a3-a1, e1 = b2-b1, e2 = b3-b1
    # Then sums are:
    # s11 = a1+b1
    # s12 = s11 + e1
    # s13 = s11 + e2
    # s21 = s11 + d1
    # s22 = s11 + d1 + e1
    # s23 = s11 + d1 + e2
    # s31 = s11 + d2
    # s32 = s11 + d2 + e1
    # s33 = s11 + d2 + e2
    # 
    # All 9 must be distinct primes (or pairwise coprime).
    # a1 >= 1, b1 >= 1, a3 = a1+d2 <= 128, b3 = b1+e2 <= 128
    # a2 = a1+d1, b2 = b1+e1 must be in [1,128] and distinct from a1,a3 and b1,b3 respectively.
    
    # Let's search over s11 (base sum), d1, d2, e1, e2
    # s11 = a1+b1, with a1,b1 >= 1, so s11 >= 2
    # a1 <= 128, b1 <= 128, so s11 <= 256
    # d1, d2 > 0 (assuming a1 < a2 < a3), e1, e2 > 0
    # a1+d1 <= 128, a1+d2 <= 128, b1+e1 <= 128, b1+e2 <= 128
    
    # For each s11, a1 can range from max(1, s11-128) to min(128, s11-1)
    # Then b1 = s11 - a1
    
    # This is still complex. Let's just do a bounded search.
    
    # Try small search space first for 9 distinct primes
    max_val = 128
    best_found = (best_score, best_A[:], best_B[:])
    
    # Try all A from a limited set, then for each A, try all B
    # To keep it tractable, let's sample
    
    # Actually, let's be smarter. Let's try specific structured candidates.
    
    # Known: [[1,7,13],[4,10,16]] gives 5 distinct coprime sums.
    # Let's try to improve by search.
    
    # Phase 1: Exhaustive search over smaller ranges
    # Try A and B from [1, 50] first
    limit = 50
    range_vals = list(range(1, limit + 1))
    
    # Precompute pairwise coprimality for all pairs in [2, 100]
    # Actually, let's just compute on the fly but cache
    
    # For efficiency, let's iterate over A combinations and for each, iterate over B
    # But that's C(50,3)^2 = 19600^2 ≈ 3.8e8 — still too large.
    
    # Better: iterate over A, and for each A, find best B.
    # For a fixed A, the sums are {a+b : a in A, b in B}
    # We want these to be pairwise coprime and maximize distinct count.
    
    # Let's try a different approach: random search with local improvement
    
    import random
    random.seed(42)
    
    def random_search(iterations=500000):
        nonlocal best_score, best_A, best_B
        count = 0
        while count < iterations and time.time() < deadline:
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            valid, nd = evaluate(A, B)
            if valid and nd > best_score:
                best_score = nd
                best_A = A[:]
                best_B = B[:]
            count += 1
            if count % 100000 == 0:
                break  # check time
    
    # But random search may not find 9. Let's think more carefully.
    # 
    # For 9 distinct sums to be pairwise coprime, a good strategy is to make them all primes.
    # Let's search for prime 3x3 grids.
    
    # Let's try: fix the differences and search for base values.
    # Let d1, d2 be differences in A, e1, e2 in B.
    # Small differences give more room for primes.
    
    # Try small differences
    best_9_found = False
    for d1 in range(1, 20):
        for d2 in range(d1 + 1, 40):
            for e1 in range(1, 20):
                for e2 in range(e1 + 1, 40):
                    # Check if a1+d1, a1+d2, b1+e1, b1+e2 are all <= 128
                    # and a1 >= 1, b1 >= 1
                    # a1 <= 128 - d2, b1 <= 128 - e2
                    # s11 = a1 + b1
                    # We need all 9 sums to be primes (or pairwise coprime)
                    # Let's try to find a1, b1 such that all 9 are primes
                    
                    max_a1 = 128 - d2
                    max_b1 = 128 - e2
                    if max_a1 < 1 or max_b1 < 1:
                        continue
                    
                    # Try a range of a1, b1
                    # s11 = a1 + b1, and we need:
                    # s11, s11+e1, s11+e2, s11+d1, s11+d1+e1, s11+d1+e2,
                    # s11+d2, s11+d2+e1, s11+d2+e2 all prime
                    
                    # s11 ranges from 2 to 256
                    # For each s11, a1 can be max(1, s11-max_b1) to min(max_a1, s11-1)
                    # b1 = s11 - a1
                    
                    for s11 in range(2, 257):
                        a1_min = max(1, s11 - max_b1)
                        a1_max = min(max_a1, s11 - 1)
                        if a1_min > a1_max:
                            continue
                        # Check if all 9 sums are prime
                        sums = [s11, s11+e1, s11+e2, s11+d1, s11+d1+e1, 
                               s11+d1+e2, s11+d2, s11+d2+e1, s11+d2+e2]
                        if all(2 <= s <= 256 and is_prime(s) for s in sums):
                            # All 9 are distinct primes (since differences are nonzero)
                            # Find valid a1, b1
                            for a1 in range(a1_min, a1_max + 1):
                                b1 = s11 - a1
                                if b1 < 1 or b1 > 128:
                                    continue
                                A = [a1, a1 + d1, a1 + d2]
                                B = [b1, b1 + e1, b1 + e2]
                                # Verify all in [1,128] and distinct
                                if all(1 <= x <= 128 for x in A + B) and \
                                   len(set(A)) == 3 and len(set(B)) == 3:
                                    best_A = A
                                    best_B = B
                                    best_score = 9
                                    best_9_found = True
                                    break
                        if best_9_found:
                            break
                    if best_9_found:
                        break
                if best_9_found:
                    break
            if best_9_found:
                break
        if best_9_found:
            break
    
    if best_9_found:
        return [best_A, best_B]
    
    # If 9 not found, try 8 distinct pairwise coprime sums
    # Do a more targeted search for 8
    
    # Let's do a random search with local refinement for 8 or 7
    random.seed(123)
    for _ in range(200000):
        if time.time() > deadline - 5:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        valid, nd = evaluate(A, B)
        if valid and nd > best_score:
            best_score = nd
            best_A = A[:]
            best_B = B[:]
            if best_score >= 8:
                break
    
    # Local improvement: try to swap elements to increase distinct coprime count
    def local_improve(A, B, max_iters=10000):
        nonlocal best_score, best_A, best_B
        A = A[:]
        B = B[:]
        cur_valid, cur_nd = evaluate(A, B)
        if not cur_valid:
            return
        cur_score = cur_nd
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try swapping an element in A
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in A:
                        continue
                    A_new = A[:]
                    A_new[i] = new_val
                    valid, nd = evaluate(A_new, B)
                    if valid and nd > cur_score:
                        A = A_new
                        cur_score = nd
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try swapping an element in B
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in B:
                        continue
                    B_new = B[:]
                    B_new[i] = new_val
                    valid, nd = evaluate(A, B_new)
                    if valid and nd > cur_score:
                        B = B_new
                        cur_score = nd
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if cur_score > best_score:
            best_score = cur_score
            best_A = A[:]
            best_B = B[:]
    
    if best_score < 8:
        local_improve(best_A, best_B)
    
    return [best_A, best_B]
