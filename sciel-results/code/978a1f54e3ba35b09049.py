"""Rewritten program using structured number-theoretic search for pairwise-coprime sumsets."""
import math
import random
import time
from math import gcd

# EVOLVE-BLOCK-START

def _is_pairwise_coprime(vals):
    """Check if all values in vals are pairwise coprime."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return (distinct_count, is_valid). Higher distinct_count with valid is better."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct = list(sums)
    if not _is_pairwise_coprime(distinct):
        return (len(distinct), False)
    return (len(distinct), True)

def _try_improve(A, B, deadline, rng):
    """Hill-climbing with targeted perturbations."""
    best_A, best_B = list(A), list(B)
    best_count, best_valid = _score(best_A, best_B)
    
    # Precompute current sums for quick comparison
    current_sums = set()
    for a in best_A:
        for b in best_B:
            current_sums.add(a + b)
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        # Try perturbing one element
        if rng.random() < 0.5:
            idx = rng.randrange(3)
            old_val = best_A[idx]
            new_val = rng.randint(1, 128)
            if new_val == old_val:
                continue
            trial_A = list(best_A)
            trial_A[idx] = new_val
            if len(set(trial_A)) != 3:
                continue
            count, valid = _score(trial_A, best_B)
            if valid and count > best_count:
                best_A, best_B = trial_A, list(best_B)
                best_count, best_valid = count, True
            elif valid and count == best_count and best_valid and rng.random() < 0.2:
                best_A = trial_A
            elif not best_valid and count > best_count:
                best_A, best_B = trial_A, list(best_B)
                best_count, best_valid = count, False
        else:
            idx = rng.randrange(3)
            old_val = best_B[idx]
            new_val = rng.randint(1, 128)
            if new_val == old_val:
                continue
            trial_B = list(best_B)
            trial_B[idx] = new_val
            if len(set(trial_B)) != 3:
                continue
            count, valid = _score(best_A, trial_B)
            if valid and count > best_count:
                best_A, best_B = list(best_A), trial_B
                best_count, best_valid = count, True
            elif valid and count == best_count and best_valid and rng.random() < 0.2:
                best_B = trial_B
            elif not best_valid and count > best_count:
                best_A, best_B = list(best_A), trial_B
                best_count, best_valid = count, False
    
    return best_A, best_B, best_count, best_valid

def _generate_prime_based_start(rng):
    """Generate starting points biased toward primes and numbers with few small factors."""
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # Mix of primes and other numbers
    if rng.random() < 0.7:
        A = sorted(rng.sample(primes, 3))
        B = sorted(rng.sample(primes, 3))
    else:
        # Use numbers that are 1 or 2 mod small primes
        candidates = [x for x in range(1, 129) if x not in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]]
        A = sorted(rng.sample(candidates, 3))
        B = sorted(rng.sample(candidates, 3))
    
    return A, B

def _prime_grid_search(deadline):
    """
    Direct search for prime grids: find a_i, b_j such that all 9 sums a_i + b_j 
    are distinct primes (hence pairwise coprime).
    
    Key insight: If a1+a1=p11, a1+a2=p12, a1+a3=p13, a2+a1=p21, a2+a2=p22, a2+a3=p23,
    a3+a1=p31, a3+a2=p32, a3+a3=p33,
    then we need:
    - p12 - p11 = a2 - a1
    - p13 - p11 = a3 - a1  
    - p21 - p11 = b2 - b1
    - p22 - p11 = (a2-a1) + (b2-b1)
    - etc.
    
    So we can pick 3 primes for the first row, 3 for the first column, and check consistency.
    """
    # Generate primes up to 256 (max sum is 128+128=256)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes_up_to_256 = sieve(256)
    prime_set = set(primes_up_to_256)
    
    best_A = None
    best_B = None
    best_count = 0
    
    # Try all combinations of 3 primes for first row sums
    # p11, p12, p13 = a1+b1, a1+b2, a1+b3
    # p21, p22, p23 = a2+b1, a2+b2, a2+b3
    # p31, p32, p33 = a3+b1, a3+b2, a3+b3
    
    # We need: a2-a1 = p21-p11 = p22-p12 = p23-p13
    #           a3-a1 = p31-p11 = p32-p12 = p33-p13
    #           b2-b1 = p12-p11 = p22-p21 = p32-p31
    #           b3-b1 = p13-p11 = p23-p21 = p33-p31
    
    # Also: a1 = p11 - b1, and b1 >= 1, a1 >= 1, so b1 <= p11-1
    #       a2 = a1 + (p21-p11), must be in [1,128]
    #       a3 = a1 + (p31-p11), must be in [1,128]
    #       b2 = b1 + (p12-p11), must be in [1,128]
    #       b3 = b1 + (p13-p11), must be in [1,128]
    
    # Strategy: iterate over possible (p11, p12, p13) triples for first row
    # and (p21, p31) for first column, check consistency
    
    # To keep it fast, limit the search space
    # Primes between 3 and 250 (sums must be >= 1+1=2, but realistically >= 3 for distinctness)
    small_primes = [p for p in primes_up_to_256 if 3 <= p <= 250]
    
    # Limit to first 100 primes to keep search manageable
    search_primes = small_primes[:100]
    n_primes = len(search_primes)
    
    # Iterate over p11, p12, p13 (first row)
    for i1 in range(n_primes):
        p11 = search_primes[i1]
        if time.time() > deadline:
            break
        
        for i2 in range(i1 + 1, n_primes):
            p12 = search_primes[i2]
            b2_minus_b1 = p12 - p11
            if b2_minus_b1 <= 0 or b2_minus_b1 > 127:
                continue
            
            for i3 in range(i2 + 1, n_primes):
                p13 = search_primes[i3]
                b3_minus_b1 = p13 - p11
                if b3_minus_b1 <= 0 or b3_minus_b1 > 127:
                    continue
                
                # Now try p21, p31 for first column
                for j2 in range(n_primes):
                    p21 = search_primes[j2]
                    a2_minus_a1 = p21 - p11
                    if a2_minus_a1 <= 0 or a2_minus_a1 > 127:
                        continue
                    
                    # Check p22 = p11 + a2_minus_a1 + b2_minus_b1
                    p22 = p11 + a2_minus_a1 + b2_minus_b1
                    if p22 > 256 or p22 not in prime_set:
                        continue
                    
                    for j3 in range(n_primes):
                        p31 = search_primes[j3]
                        a3_minus_a1 = p31 - p11
                        if a3_minus_a1 <= 0 or a3_minus_a1 > 127:
                            continue
                        
                        # Check p23 = p11 + a2_minus_a1 + b3_minus_b1
                        p23 = p11 + a2_minus_a1 + b3_minus_b1
                        if p23 > 256 or p23 not in prime_set:
                            continue
                        
                        # Check p32 = p11 + a3_minus_a1 + b2_minus_b1
                        p32 = p11 + a3_minus_a1 + b2_minus_b1
                        if p32 > 256 or p32 not in prime_set:
                            continue
                        
                        # Check p33 = p11 + a3_minus_a1 + b3_minus_b1
                        p33 = p11 + a3_minus_a1 + b3_minus_b1
                        if p33 > 256 or p33 not in prime_set:
                            continue
                        
                        # All 9 sums are primes! Now find b1 such that all a_i, b_j in [1,128]
                        # a1 = p11 - b1, a2 = a1 + a2_minus_a1, a3 = a1 + a3_minus_a1
                        # b2 = b1 + b2_minus_b1, b3 = b1 + b3_minus_b1
                        
                        # Constraints:
                        # 1 <= b1 <= 128
                        # 1 <= a1 = p11 - b1 <= 128  =>  p11-128 <= b1 <= p11-1
                        # 1 <= a2 = p11 - b1 + a2_minus_a1 <= 128
                        # 1 <= a3 = p11 - b1 + a3_minus_a1 <= 128
                        # 1 <= b2 = b1 + b2_minus_b1 <= 128
                        # 1 <= b3 = b1 + b3_minus_b1 <= 128
                        
                        b1_min = max(1, p11 - 128, p11 - a2_minus_a1 - 128 + 1, p11 - a3_minus_a1 - 128 + 1, 1 - b2_minus_b1, 1 - b3_minus_b1)
                        b1_max = min(128, p11 - 1, p11 - a2_minus_a1 - 1 + 128, p11 - a3_minus_a1 - 1 + 128, 128 - b2_minus_b1, 128 - b3_minus_b1)
                        
                        if b1_min > b1_max:
                            continue
                        
                        # Try each valid b1
                        for b1 in range(b1_min, b1_max + 1):
                            a1 = p11 - b1
                            a2 = a1 + a2_minus_a1
                            a3 = a1 + a3_minus_a1
                            b2 = b1 + b2_minus_b1
                            b3 = b1 + b3_minus_b1
                            
                            # Verify all in [1, 128]
                            if not (1 <= a1 <= 128 and 1 <= a2 <= 128 and 1 <= a3 <= 128):
                                continue
                            if not (1 <= b1 <= 128 and 1 <= b2 <= 128 and 1 <= b3 <= 128):
                                continue
                            
                            # Check distinctness
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            if len(set(A)) != 3 or len(set(B)) != 3:
                                continue
                            
                            # Compute all sums and verify they're the expected primes
                            sums = set()
                            for a in A:
                                for b in B:
                                    sums.add(a + b)
                            
                            # All sums should be distinct primes
                            if len(sums) == 9 and all(s in prime_set for s in sums):
                                # This is a valid solution with 9 distinct sums!
                                return [sorted(A), sorted(B)], 9, True
                            
                            # Even if not all 9 are distinct, check if valid
                            if len(sums) > best_count:
                                count, valid = _score(A, B)
                                if valid and count > best_count:
                                    best_A = sorted(A)
                                    best_B = sorted(B)
                                    best_count = count
    
    if best_A is not None:
        return best_A, best_B, best_count, True
    return None, None, 0, False

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    rng = random.Random(42)
    
    # Start with a known valid solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count, best_valid = _score(best_A, best_B)
    
    # If we already have 9 (maximum), return immediately
    if best_count == 9 and best_valid:
        return [best_A, best_B]
    
    # First, try the direct prime grid search
    if time.time() < deadline - 10:
        pg_A, pg_B, pg_count, pg_valid = _prime_grid_search(deadline - 10)
        if pg_A is not None:
            if pg_valid and pg_count > best_count:
                best_A, best_B = pg_A, pg_B
                best_count, best_valid = pg_count, True
                if best_count == 9:
                    return [best_A, best_B]
            elif not pg_valid and pg_count > best_count:
                best_A, best_B = pg_A, pg_B
                best_count, best_valid = pg_count, False
    
    # Try multiple restarts with different strategies
    for seed in range(150):
        if time.time() > deadline:
            break
        
        # Use different initialization strategies
        if seed % 3 == 0:
            A, B = _generate_prime_based_start(rng)
        elif seed % 3 == 1:
            A = sorted(rng.sample(range(1, 129), 3))
            B = sorted(rng.sample(range(1, 129), 3))
        else:
            # Bias toward small numbers to keep sums in a reasonable range
            small_range = range(1, 65)
            A = sorted(rng.sample(small_range, 3))
            B = sorted(rng.sample(small_range, 3))
        
        # Local improvement
        A2, B2, cnt, valid = _try_improve(A, B, deadline, rng)
        
        if valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, True
            if best_count == 9:
                break
        elif not best_valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, False
    
    # Final refinement from the best found
    if time.time() < deadline - 2 and best_count < 9:
        A2, B2, cnt, valid = _try_improve(best_A, best_B, deadline, rng)
        if valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, True
        elif not best_valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, False
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
