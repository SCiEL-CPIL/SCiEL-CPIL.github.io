import time
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin under 90s wall-clock
    
    # Precompute primes up to 256 (max sum 128+128)
    # Sieve of Eratosthenes
    limit = 256
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_set = set(primes)
    
    def is_pairwise_coprime(vals):
        # vals is a list of distinct integers
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True

    def score_of(A, B):
        s = {a + b for a in A for b in B}
        if len(s) < 2:
            return -1
        # Optimization: if all are primes and distinct, they are pairwise coprime
        # (Note: 2 is the only even prime. If 2 is in the set, all others must be odd.
        # Primes > 2 are odd. So distinct primes are always pairwise coprime.)
        if all(x in prime_set for x in s):
            return len(s)
        if not is_pairwise_coprime(list(s)):
            return -1
        return len(s)

    # Start with a known valid solution (score 5 or 6 usually)
    # Example: A=[1, 7, 13], B=[4, 10, 16]
    # Sums: 5, 11, 17, 11, 17, 23, 17, 23, 29 -> {5, 11, 17, 23, 29} (5 distinct primes)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # If we already have 9, return immediately
    if best_score == 9:
        return best_A, best_B

    # Strategy: Search for Arithmetic Progressions
    # A = [a1, a1+da, a1+2*da]
    # B = [b1, b1+db, b1+2*db]
    # Sums are:
    # a1+b1
    # a1+b1+db
    # a1+b1+2*db
    # a1+b1+da
    # a1+b1+da+db
    # a1+b1+da+2*db
    # a1+b1+2*da
    # a1+b1+2*da+db
    # a1+b1+2*da+2*db
    
    # Let p0 = a1+b1.
    # The 9 sums are p0 + i*da + j*db for i,j in {0,1,2}.
    # We want all 9 to be distinct primes.
    
    # Iterate over possible differences da, db.
    # da, db must be such that a1+2*da <= 128 and b1+2*db <= 128.
    # Max da, db is roughly 63.
    
    # To optimize, we can iterate over p0 (the smallest sum) and check if the
    # 8 other numbers (p0 + offsets) are primes.
    # Offsets: da, db, 2*da, 2*db, da+db, da+2*db, 2*da+db, 2*da+2*db
    
    found_solution = False
    
    # Limit search space to keep it fast
    # We can iterate da, db first, then find valid p0
    # Or iterate p0, da, db.
    # Let's iterate da, db, and then find p0.
    
    # Precompute offsets for speed? No, da, db vary.
    
    # Optimization: 
    # For a fixed da, db, the condition is that for some p0,
    # p0, p0+da, p0+db, p0+da+db, p0+2*da, p0+2*da+db, p0+2*db, p0+da+2*db, p0+2*da+2*db
    # are all primes.
    
    # Let's iterate da from 1 to 60, db from 1 to 60.
    # For each pair (da, db), we check if there exists a p0 such that all 9 values are prime.
    # p0 must be such that:
    # 1 <= a1 <= 128, 1 <= a1+2*da <= 128 => 1 <= a1 <= 128-2*da
    # 1 <= b1 <= 128, 1 <= b1+2*db <= 128 => 1 <= b1 <= 128-2*db
    # a1 + b1 = p0
    # So p0_min = 2 (1+1)
    # p0_max = (128-2*da) + (128-2*db) = 256 - 2*da - 2*db
    # Also a1 >= 1, b1 >= 1 => p0 >= 2.
    # Also a1 = p0 - b1.
    # Constraints on a1, b1:
    # 1 <= a1 <= 128 - 2*da
    # 1 <= b1 <= 128 - 2*db
    # a1 + b1 = p0
    
    # This implies:
    # p0 - (128 - 2*db) <= a1 <= p0 - 1
    # 1 <= a1 <= 128 - 2*da
    # So max(1, p0 - (128 - 2*db)) <= a1 <= min(128 - 2*da, p0 - 1)
    # If this interval is non-empty, there is a valid (a1, b1).
    
    # We can just check if any p0 in the valid range works.
    # To speed up, we can iterate p0 and check primes.
    
    # Let's swap loops: iterate p0?
    # p0 is between 2 and 256.
    # For each p0, iterate da, db?
    # da, db up to 60. 60*60*256 ~ 900,000 checks.
    # Each check is 9 prime lookups. ~8 million operations. Very fast.
    
    for p0 in range(2, 257):
        if time.time() > deadline - 5:
            break
        if not is_prime[p0]:
            continue
            
        # da must be such that p0 + 2*da <= 256 (max sum)
        # And a1+2*da <= 128.
        # Let's just iterate da, db and check if a valid a1, b1 exists.
        
        for da in range(1, 64):
            if time.time() > deadline - 3:
                break
            # Check if p0 + 2*da is prime? Not necessarily, it's just one of the sums.
            # The sums are p0 + i*da + j*db.
            # We need all 9 to be prime.
            
            # Max db such that p0 + 2*da + 2*db <= 256?
            # Actually, we just check if the values are in [2, 256] and prime.
            
            for db in range(1, 64):
                if time.time() > deadline - 2:
                    break
                
                # Calculate the 8 offsets
                # Offsets: da, db, 2*da, 2*db, da+db, da+2*db, 2*da+db, 2*da+2*db
                # Check if p0 + offset is prime for all offsets.
                # Also check if they are distinct.
                
                sums = [
                    p0,
                    p0 + da,
                    p0 + db,
                    p0 + da + db,
                    p0 + 2*da,
                    p0 + 2*da + db,
                    p0 + 2*db,
                    p0 + da + 2*db,
                    p0 + 2*da + 2*db
                ]
                
                # Quick check: all distinct?
                if len(set(sums)) != 9:
                    continue
                    
                # Quick check: all in range and prime?
                valid = True
                for s in sums:
                    if s > 256 or not is_prime[s]:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Now check if there exists a1, b1
                # Constraints:
                # 1 <= a1 <= 128 - 2*da
                # 1 <= b1 <= 128 - 2*db
                # a1 + b1 = p0
                
                a1_min = 1
                a1_max = 128 - 2*da
                b1_min = 1
                b1_max = 128 - 2*db
                
                # a1 = p0 - b1
                # b1_min <= p0 - a1 <= b1_max  =>  p0 - b1_max <= a1 <= p0 - b1_min
                a1_min_constrained = max(a1_min, p0 - b1_max)
                a1_max_constrained = min(a1_max, p0 - b1_min)
                
                if a1_min_constrained <= a1_max_constrained:
                    # Valid solution found
                    a1 = a1_min_constrained
                    b1 = p0 - a1
                    A = [a1, a1 + da, a1 + 2*da]
                    B = [b1, b1 + db, b1 + 2*db]
                    
                    # Verify
                    if all(1 <= v <= 128 for v in A) and all(1 <= v <= 128 for v in B):
                        if len(set(A)) == 3 and len(set(B)) == 3:
                            sc = score_of(A, B)
                            if sc == 9:
                                return A, B
                            elif sc > best_score:
                                best_score = sc
                                best_A = A
                                best_B = B
                    # If we found a score 9, we return.
                    # If we found a better score, we update.
                    # Since we are looking for primes, sc should be 9.
                    # We can break early if we found 9.
                    if best_score == 9:
                        return best_A, best_B

    # Fallback: Random Search
    # If the systematic search didn't find 9 (unlikely given the density of primes),
    # try random search for non-AP structures or non-prime coprime sets.
    import random
    random.seed(42)
    
    # Bias towards sets that produce odd sums (to avoid factor 2 conflicts if 2 is present, 
    # though 2 is coprime to odd numbers, so it's fine. But even sums > 2 are composite.
    # If we want primes, we need odd sums.
    # Sum is odd if one addend is even and the other is odd.
    # So A should be all even or all odd?
    # If A is all odd, B must be all even for all sums to be odd.
    # If A is all even, B must be all odd.
    # Let's try this structure in random search.
    
    while time.time() < deadline - 2:
        # Choose parity strategy
        if random.random() < 0.5:
            # A odd, B even
            A = sorted(random.sample(range(1, 129, 2), 3))
            B = sorted(random.sample(range(2, 129, 2), 3))
        else:
            # A even, B odd
            A = sorted(random.sample(range(2, 129, 2), 3))
            B = sorted(random.sample(range(1, 129, 2), 3))
            
        sc = score_of(A, B)
        if sc > best_score:
            best_score = sc
            best_A = A
            best_B = B
            if best_score == 9:
                break

    return best_A, best_B
