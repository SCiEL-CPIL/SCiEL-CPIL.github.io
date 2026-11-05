import math
import time
import random
from itertools import combinations

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def get_sums(A, B):
    return {a + b for a in A for b in B}

def is_pairwise_coprime(sums):
    sums_list = list(sums)
    for i in range(len(sums_list)):
        for j in range(i + 1, len(sums_list)):
            if math.gcd(sums_list[i], sums_list[j]) != 1:
                return False
    return True

def score(A, B):
    """
    Returns the score based on the task contract:
    1. A, B must be valid (3 distinct ints in [1, 128]).
    2. All DISTINCT values in A+B must be pairwise coprime.
    3. Score is the cardinality of A+B (number of distinct sums).
    If condition 2 is not met, score is 0.
    """
    if len(A) != 3 or len(B) != 3:
        return 0
    if len(set(A)) != 3 or len(set(B)) != 3:
        return 0
    for x in A:
        if not (1 <= x <= 128):
            return 0
    for x in B:
        if not (1 <= x <= 128):
            return 0
    
    sums = get_sums(A, B)
    if not is_pairwise_coprime(sums):
        return 0
    return len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 80 # Safety margin for 90s limit
    
    # Precompute primes up to 256
    primes = [i for i in range(2, 257) if is_prime(i)]
    
    best_A = None
    best_B = None
    best_score = 0
    
    # Strategy 1: Try to find a solution with 9 distinct prime sums.
    # This guarantees pairwise coprime and max cardinality.
    # We iterate over possible B sets and try to find A.
    # Or just random search with heavy bias towards primes.
    
    # Since 128 is small, we can do a smarter random search.
    random.seed(42)
    
    # Known good starting points?
    # Let's try some structured sets first.
    # e.g. A = [1, 2, 4], B = [1, 2, 4] -> Sums: 2,3,5, 3,4,6, 5,6,8. Distinct: 2,3,4,5,6,8.
    # 2 and 4 not coprime. Score 0 under strict rule.
    
    # Let's use local search.
    # Initialize with a random valid set.
    A = random.sample(range(1, 129), 3)
    B = random.sample(range(1, 129), 3)
    
    current_score = score(A, B)
    if current_score > best_score:
        best_score = current_score
        best_A = A[:]
        best_B = B[:]
        
    # If we haven't found a valid set, we might be stuck at 0.
    # We need to find ANY set where distinct sums are pairwise coprime.
    
    # Heuristic: Sums should be primes.
    # Let's try to construct A and B such that sums are primes.
    
    improved = True
    while time.time() < deadline and improved:
        improved = False
        # Local search: swap one element in A or B
        for _ in range(100): # Limit inner loop steps per outer iteration to prevent hanging on one branch
             if time.time() > deadline: break
             # Pick a random position to swap
             side = random.choice(['A', 'B'])
             idx = random.randint(0, 2)
             new_val = random.randint(1, 128)
             
             old_val = A[idx] if side == 'A' else B[idx]
             if new_val == old_val:
                 continue
             if side == 'A' and new_val in A:
                 continue
             if side == 'B' and new_val in B:
                 continue
                 
             if side == 'A':
                 A[idx] = new_val
             else:
                 B[idx] = new_val
                 
             new_score = score(A, B)
             
             if new_score > current_score:
                 current_score = new_score
                 improved = True
                 if current_score > best_score:
                     best_score = current_score
                     best_A = A[:]
                     best_B = B[:]
                     if best_score == 9:
                         return [best_A, best_B]
             else:
                 # Revert
                 if side == 'A':
                     A[idx] = old_val
                 else:
                     B[idx] = old_val
             
             # If we hit 9, we are done
             if current_score == 9:
                 return [A, B]
        
        # If stuck, do a random restart?
        # Only if current score is low.
        if current_score < 5 and random.random() < 0.1:
             A = random.sample(range(1, 129), 3)
             B = random.sample(range(1, 129), 3)
             current_score = score(A, B)
             if current_score > best_score:
                 best_score = current_score
                 best_A = A[:]
                 best_B = B[:]
             
    return [best_A, best_B]

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    import random
    import math
    from itertools import combinations

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def get_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(sums):
        sums_list = list(sums)
        for i in range(len(sums_list)):
            for j in range(i + 1, len(sums_list)):
                if math.gcd(sums_list[i], sums_list[j]) != 1:
                    return False
        return True

    def score(A, B):
        if len(A) != 3 or len(B) != 3:
            return 0
        if len(set(A)) != 3 or len(set(B)) != 3:
            return 0
        for x in A:
            if not (1 <= x <= 128):
                return 0
        for x in B:
            if not (1 <= x <= 128):
                return 0
        sums = get_sums(A, B)
        if not is_pairwise_coprime(sums):
            return 0
        return len(sums)

    start_time = time.time()
    deadline = start_time + 80
    
    # Initial best (from seed knowledge or default)
    # Let's start with a known valid pair if possible, or just 0.
    # Seed had [1, 7, 13] and [4, 10, 16] -> sums: 5, 11, 17, 8, 14, 20, 14, 20, 26.
    # Distinct: 5, 8, 11, 14, 17, 20, 26.
    # gcd(8, 14) = 2. Invalid.
    # Let's just start with 0.
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]
    best_score = score(best_A, best_B)
    
    # Precompute primes
    primes = [i for i in range(2, 257) if is_prime(i)]
    
    # Strategy: 
    # 1. Exploit the parity constraint for 9 distinct prime sums.
    #    If all 9 sums are distinct primes, they must be pairwise coprime.
    #    For 9 sums to be odd primes (all primes > 2 are odd), 
    #    a+b must be odd for all a in A, b in B.
    #    This implies one set must be all odd and the other all even.
    #    
    #    Let A = {a1, a2, a3} be odd, B = {b1, b2, b3} be even.
    #    Sums S = {ai + bj}.
    #    We want S to contain 9 distinct primes.
    #    
    #    We can iterate over possible even B sets (small number) and try to find odd A sets.
    #    Or iterate over odd A sets and try to find even B sets.
    #    
    #    Even numbers in [1, 128]: 1, 3, ..., 127 (128 values? No, 128 is even. 2, 4, ..., 128 -> 64 evens)
    #    Odd numbers in [1, 128]: 1, 3, ..., 127 (64 odds)
    
    evens = [x for x in range(1, 129) if x % 2 == 0]
    odds = [x for x in range(1, 129) if x % 2 == 1]
    
    # There are C(64, 3) ~ 41664 combinations for each set.
    # Total pairs ~ 1.7 billion. Too many to brute force directly.
    # But we can prune heavily.
    
    # For a fixed B (even), the sums are a_i + b_j.
    # We need all 9 sums to be prime and distinct.
    
    # Let's try a smarter search.
    # Iterate over B (even) combinations.
    # For each B, we want to find A (odd) such that A+B are 9 distinct primes.
    
    # We can use a "candidate filter".
    # For a fixed B = {b1, b2, b3}, and a candidate a, 
    # the sums are {a+b1, a+b2, a+b3}.
    # For A = {a1, a2, a3}, we need all 9 sums to be prime and distinct.
    
    # Let's precompute for each odd a, which even b's make a+b prime?
    # Actually, let's just iterate over B combinations, and for each B, 
    # check if there exist 3 distinct odd numbers a1, a2, a3 such that 
    # all 9 sums are prime and distinct.
    
    # To speed up, we can precompute a list of "valid a's" for a given B.
    # An odd number a is "valid" for B if a+b1, a+b2, a+b3 are all prime.
    # Then we need to pick 3 such a's such that the 9 sums are distinct.
    
    # Let's try this.
    
    # Precompute primality for sums up to 256
    is_prime_arr = [False] * 257
    for p in primes:
        is_prime_arr[p] = True
    
    # Iterate over B combinations
    # To avoid duplicate work, we can fix b1 < b2 < b3.
    
    found = False
    best_score_current = best_score
    
    # Limit the search space to increase chance of finding a solution quickly
    # We can sample B sets or just iterate all if fast enough.
    # C(64,3) = 41664. For each, we filter A.
    # Filtering A: 64 odds. For each a, check 3 sums.
    # So 41664 * 64 * 3 ops ~ 8 million ops. Very fast in Python? Maybe 1-2 seconds.
    # Then for the valid A's, we need to check distinctness of 9 sums.
    # If we have k valid A's, we need to check C(k, 3) triples.
    # k is likely small (maybe 5-10). C(10,3) = 120.
    # So total ops ~ 8M + 41664 * 120 ~ 8M + 5M = 13M. Should be fine.
    
    # Let's implement this.
    
    # Optimization: Sort evens and odds.
    
    # We can break early if we find a solution with score 9.
    
    # Let's also keep a backup random search in case this takes too long or fails.
    
    search_start = time.time()
    
    # To ensure we don't miss the deadline, we'll add a timeout to this loop.
    # If it takes > 30s, we abort and fall back to random search.
    
    for b1_idx in range(len(evens)):
        if time.time() - search_start > 30:
            break
        b1 = evens[b1_idx]
        for b2_idx in range(b1_idx + 1, len(evens)):
            b2 = evens[b2_idx]
            for b3_idx in range(b2_idx + 1, len(evens)):
                b3 = evens[b3_idx]
                
                # Find valid a's
                valid_as = []
                for a in odds:
                    s1 = a + b1
                    s2 = a + b2
                    s3 = a + b3
                    if s1 <= 256 and s2 <= 256 and s3 <= 256:
                        if is_prime_arr[s1] and is_prime_arr[s2] and is_prime_arr[s3]:
                            valid_as.append(a)
                
                if len(valid_as) < 3:
                    continue
                
                # Check triples of valid_as
                # We need 9 distinct sums.
                # Sums are {a_i + b_j}.
                # Since b1, b2, b3 are distinct, and a_i are distinct,
                # a_i + b_j can collide if a_i + b_j = a_k + b_l.
                # This happens if a_i - a_k = b_l - b_j.
                
                # We just need to check the distinctness of the 9 sums.
                
                # To speed up, we can just pick the first 3 valid_as and check.
                # If that fails, try other combinations.
                
                # Let's try a few random triples from valid_as first, then exhaustive if small.
                
                if len(valid_as) <= 10:
                    for a1_idx in range(len(valid_as)):
                        a1 = valid_as[a1_idx]
                        for a2_idx in range(a1_idx + 1, len(valid_as)):
                            a2 = valid_as[a2_idx]
                            for a3_idx in range(a2_idx + 1, len(valid_as)):
                                a3 = valid_as[a3_idx]
                                
                                A = [a1, a2, a3]
                                B = [b1, b2, b3]
                                
                                sums = get_sums(A, B)
                                if len(sums) == 9 and is_pairwise_coprime(sums):
                                    return [A, B]
                else:
                    # Random sample
                    for _ in range(20):
                        A = random.sample(valid_as, 3)
                        B = [b1, b2, b3]
                        sums = get_sums(A, B)
                        if len(sums) == 9 and is_pairwise_coprime(sums):
                            return [A, B]
    
    # If we didn't find a 9-sum solution, fall back to random search for high score
    # The seed program did this. We can reuse the logic but with better initialization.
    
    # Let's try to find a solution with 8 or more sums.
    # We can use the random search from the seed, but with a time limit.
    
    random.seed(42)
    
    # Initialize with the best found so far or random
    A = best_A[:]
    B = best_B[:]
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        for _ in range(100):
            if time.time() > deadline: break
            side = random.choice(['A', 'B'])
            idx = random.randint(0, 2)
            new_val = random.randint(1, 128)
            
            if side == 'A':
                if new_val in A: continue
                old = A[idx]
                A[idx] = new_val
            else:
                if new_val in B: continue
                old = B[idx]
                B[idx] = new_val
                
            new_score = score(A, B)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return [best_A, best_B]
            else:
                if side == 'A':
                    A[idx] = old
                else:
                    B[idx] = old
                    
            if current_score == 9:
                return [A, B]
                
        if not improved:
            # Random restart
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            current_score = score(A, B)
            if current_score > best_score:
                best_score = current_score
                best_A = A[:]
                best_B = B[:]
                
    return [best_A, best_B]

# EVOLVE-BLOCK-END
