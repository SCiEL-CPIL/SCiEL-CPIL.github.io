"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # All integers in [1, 128]
    # We need A and B each with 3 distinct integers in [1, 128]
    # All DISTINCT values in A+B must be pairwise coprime
    # Maximize the number of distinct values in A+B
    
    # Key insight: if all distinct sums are pairwise coprime, and they're all > 1,
    # then no sum can divide another. Also, if two sums share a common factor, that's bad.
    # 
    # Strategy: Try to make all 9 sums distinct and pairwise coprime.
    # If we can get 9 distinct pairwise-coprime sums, that's the maximum possible.
    
    # Let's think about this more carefully.
    # If A = {a1, a2, a3} and B = {b1, b2, b3}, then A+B has up to 9 elements.
    # For all 9 to be distinct, we need all ai + bj to be different.
    # For all 9 to be pairwise coprime, each pair must have gcd 1.
    
    # One approach: use primes. If all 9 sums are distinct primes, they're automatically pairwise coprime.
    # So we need: find A={a1,a2,a3}, B={b1,b2,b3} in [1,128] such that all 9 values ai+bj are distinct primes.
    
    # Let's try a more systematic search.
    # First, generate all primes up to 256 (max sum is 128+128=256)
    
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
    prime_set = set(primes)
    
    # Check if all 9 sums can be distinct primes
    # We need A and B such that {ai + bj for i,j in 0,1,2} are all in prime_set and all distinct
    
    # Let's try: fix A and search for B, or use a smarter approach.
    
    # Actually, let's think about the structure. If a1 < a2 < a3 and b1 < b2 < b3,
    # then a1+b1 < a1+b2 < a1+b3, and a2+b1 < a2+b2 < a2+b3, etc.
    # But cross-terms can interleave.
    
    # Let me try a direct search with pruning.
    # Since 128 choose 3 is about 340k, and for each A we'd need to check B options,
    # that's too slow naively. Let's be smarter.
    
    # Alternative: pick 9 primes that can be arranged as a 3x3 "sum matrix".
    # That is, find primes p11, p12, p13, p21, p22, p23, p31, p32, p33 such that
    # p_ij = ai + bj for some a1,a2,a3,b1,b2,b3.
    # This means: p12 - p11 = b2 - b1, p13 - p11 = b3 - b1, p21 - p11 = a2 - a1, etc.
    # And consistency: p12 + p21 = p11 + p22, etc. (the "rectangle" condition)
    
    # Let's try to find such configurations.
    
    # For efficiency, let's iterate over possible differences.
    # Let da = a2 - a1, db = b2 - b1, dc = a3 - a1, dd = b3 - b1.
    # Then:
    # p11 = a1 + b1
    # p12 = a1 + b1 + db
    # p13 = a1 + b1 + dd
    # p21 = a1 + da + b1
    # p22 = a1 + da + b1 + db
    # p23 = a1 + da + b1 + dd
    # p31 = a1 + dc + b1
    # p32 = a1 + dc + b1 + db
    # p33 = a1 + dc + b1 + dd
    
    # All must be primes, and a1, a2, a3, b1, b2, b3 in [1, 128].
    
    # So: p11, p11+db, p11+dd, p11+da, p11+da+db, p11+da+dd, p11+dc, p11+dc+db, p11+dc+dd all primes.
    # And a1 >= 1, a1+da <= 128, a1+dc <= 128, b1 >= 1, b1+db <= 128, b1+dd <= 128.
    # So: 1 <= a1 <= 128 - dc, 1 <= b1 <= 128 - dd, da >= 1, dc >= da (if a3 > a2), db >= 1, dd >= db.
    
    # Actually a2 and a3 just need to be distinct from a1 and each other, not necessarily ordered.
    # Let's just say da, dc are positive integers with da != dc, and similarly db != dd.
    
    # Let's search over small differences to keep things manageable.
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Try to find 9 distinct prime sums
    # Iterate over possible p11 values and differences
    
    # To limit search, note that p11 >= 2 (smallest prime) and p11 + dc + dd <= 256
    # So dc + dd <= 256 - p11
    
    # Let's try a focused search
    max_diff = 50  # limit differences for speed
    
    found = False
    for p11 in primes:
        if time.time() > deadline - 5:
            break
        # p11 + dc + dd <= 256, so dc + dd <= 256 - p11
        max_sum_diff = 256 - p11
        if max_sum_diff < 2:
            break
        
        for da in range(1, min(max_diff, max_sum_diff - 1)):
            for dc in range(1, min(max_diff, max_sum_diff - da)):
                if dc == da:
                    continue
                for db in range(1, min(max_diff, max_sum_diff - 1)):
                    for dd in range(1, min(max_diff, max_sum_diff - da - db + 1)):
                        if dd == db:
                            continue
                        if da + db + dd > max_sum_diff:
                            continue
                        if dc + dd > max_sum_diff:
                            continue
                        
                        # Check all 9 are primes
                        p12 = p11 + db
                        p13 = p11 + dd
                        p21 = p11 + da
                        p22 = p11 + da + db
                        p23 = p11 + da + dd
                        p31 = p11 + dc
                        p32 = p11 + dc + db
                        p33 = p11 + dc + dd
                        
                        if (p11 in prime_set and p12 in prime_set and p13 in prime_set and
                            p21 in prime_set and p22 in prime_set and p23 in prime_set and
                            p31 in prime_set and p32 in prime_set and p33 in prime_set):
                            
                            # All 9 are primes. Now check bounds.
                            # a1 = a1, a2 = a1 + da, a3 = a1 + dc
                            # b1 = b1, b2 = b1 + db, b3 = b1 + dd
                            # p11 = a1 + b1
                            # Need: 1 <= a1, a1+da, a1+dc <= 128
                            #       1 <= b1, b1+db, b1+dd <= 128
                            # a1 >= 1, b1 >= 1, a1 + b1 = p11
                            # a1 <= 128 - max(da, dc), b1 <= 128 - max(db, dd)
                            # a1 >= 1, b1 = p11 - a1 >= 1 => a1 <= p11 - 1
                            # b1 <= 128 - max(db, dd) => p11 - a1 <= 128 - max(db,dd) => a1 >= p11 - 128 + max(db,dd)
                            # a1 <= 128 - max(da, dc)
                            # a1 >= 1
                            
                            a1_min = max(1, p11 - 128 + max(db, dd))
                            a1_max = min(p11 - 1, 128 - max(da, dc))
                            
                            if a1_min <= a1_max:
                                a1 = a1_min
                                b1 = p11 - a1
                                A = [a1, a1 + da, a1 + dc]
                                B = [b1, b1 + db, b1 + dd]
                                
                                # Verify all in [1, 128]
                                if all(1 <= x <= 128 for x in A) and all(1 <= x <= 128 for x in B):
                                    # Verify distinct
                                    if len(set(A)) == 3 and len(set(B)) == 3:
                                        # Verify sums
                                        sums = [a + b for a in A for b in B]
                                        distinct_sums = set(sums)
                                        if len(distinct_sums) == 9:
                                            # All pairwise coprime (since all are distinct primes)
                                            best_count = 9
                                            best_A = A
                                            best_B = B
                                            found = True
                                            return [A, B]
    
    if found:
        return [best_A, best_B]
    
    # If we couldn't find 9, try to find the best we can
    # Fall back to a simpler search for 8 or 7 distinct pairwise-coprime sums
    
    # Let's try a more direct search for good solutions
    # Use the parity insight: one set all odd, other all even, so all sums are odd
    # Then sums are odd, and if they're primes, they're pairwise coprime
    
    # Try to find A all odd, B all even, with many distinct prime sums
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Sample some combinations
    import random
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline - 10 and attempts < 200000:
        attempts += 1
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        
        # Check pairwise coprime
        valid = True
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(distinct) > best_count:
            best_count = len(distinct)
            best_A = A
            best_B = B
            if best_count == 9:
                return [A, B]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
