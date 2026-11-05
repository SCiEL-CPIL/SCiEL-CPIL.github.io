import time
import math
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 85  # safety margin before 90s kill

    # Precompute primality for all possible sums (2..256)
    def sieve(max_n):
        is_p = [True] * (max_n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(max_n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, max_n+1, i):
                    is_p[j] = False
        return is_p

    is_prime = sieve(256)

    def is_coprime_set(nums):
        """Check if all pairs in nums are coprime."""
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def score_pair(A, B):
        """Count distinct pairwise-coprime values in A+B."""
        if len(A) != 3 or len(B) != 3:
            return 0
        if len(set(A)) != 3 or len(set(B)) != 3:
            return 0
        for x in A:
            if x < 1 or x > 128:
                return 0
        for x in B:
            if x < 1 or x > 128:
                return 0
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        # Greedy: pick sums in sorted order, each must be coprime to all previously chosen
        count = 0
        chosen = []
        for s in sorted(sums):
            ok = True
            for t in chosen:
                if math.gcd(s, t) != 1:
                    ok = False
                    break
            if ok:
                chosen.append(s)
                count += 1
        return count

    def all_nine_distinct_coprime(A, B):
        """Check if all 9 sums are distinct and pairwise coprime."""
        sums = []
        for a in A:
            for b in B:
                s = a + b
                if s in sums:
                    return False
                sums.append(s)
        return is_coprime_set(sums)

    # Strategy: Find A, B such that all 9 sums are distinct primes (hence pairwise coprime)
    # This gives score 9, which is the maximum possible (3x3=9 sums).
    
    # Precompute all valid triples
    all_triples = list(combinations(range(1, 129), 3))
    
    # Build index: for each sum value, which triples contain it as an element
    # Actually, let's use a more direct approach.
    
    # For a given A = {a1, a2, a3}, the sums with B = {b1, b2, b3} are:
    # a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    # We need all 9 to be distinct primes.
    
    # Key insight: fix A, then for each pair (b_i, b_j) in B, we need:
    # a_i + b_i, a_i + b_j, a_j + b_i, a_j + b_j to be primes (for the sub-problem)
    # But more efficiently: for fixed A, find B such that all a_i + b_j are primes.
    
    # For fixed A = {a1, a2, a3}, we need B = {b1, b2, b3} such that:
    # All 9 values a_i + b_j are primes.
    # This means b_j must be such that a1+b_j, a2+b_j, a3+b_j are all prime.
    # So b_j must be in the intersection: {x : a1+x, a2+x, a3+x all prime}
    
    # Let's precompute for each A, the set of valid b values.
    
    # Actually, let's just do a direct search with pruning.
    
    rng = random.Random(42)
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]
    
    # Phase 1: Try to find a 9-prime solution via structured search
    # For each A, compute valid b's, then check if we can pick 3 distinct valid b's
    
    # Precompute: for each a in [1,128], for each b in [1,128], is a+b prime?
    # prime_sum[a][b] = is_prime[a+b]
    prime_sum = [[False]*129 for _ in range(129)]
    for a in range(1, 129):
        for b in range(1, 129):
            prime_sum[a][b] = is_prime[a + b]
    
    # For a given A = (a1, a2, a3), valid b values are those where:
    # prime_sum[a1][b] and prime_sum[a2][b] and prime_sum[a3][b]
    
    found = False
    
    # Iterate over A triples in a smart order
    # Try smaller A first since they have more room for b
    
    # Limit search time
    search_time_limit = DEADLINE - 10
    
    # Try all A triples but with early termination
    # There are C(128,3) = 341,376 triples. For each, we check up to 128 b values.
    # That's ~44 million checks. Might be too slow. Let's optimize.
    
    # Better: for each A, compute valid_b set, then check if |valid_b| >= 3
    
    # But we also need the 9 sums to be DISTINCT, not just prime.
    # If a1+b1 = a2+b2, that's a collision.
    
    # Let's do a hybrid: random sampling of A, then check valid B's
    
    attempts = 0
    while time.time() < search_time_limit and not found:
        A = sorted(rng.sample(range(1, 129), 3))
        a1, a2, a3 = A
        
        # Find all b in [1,128] such that a1+b, a2+b, a3+b are all prime
        valid_bs = []
        for b in range(1, 129):
            if prime_sum[a1][b] and prime_sum[a2][b] and prime_sum[a3][b]:
                valid_bs.append(b)
        
        if len(valid_bs) < 3:
            attempts += 1
            continue
        
        # Try to pick 3 distinct b's from valid_bs such that all 9 sums are distinct
        # The 9 sums are a_i + b_j for i,j in {1,2,3}
        # Distinctness: a_i + b_j != a_k + b_l unless (i,j)=(k,l)
        # This is equivalent to: the 3x3 matrix of sums has all distinct entries
        
        # Try all combinations of 3 from valid_bs (if not too many)
        if len(valid_bs) <= 20:
            for B in combinations(valid_bs, 3):
                B = list(B)
                if all_nine_distinct_coprime(A, B):
                    best_A, best_B = A, B
                    best_score = 9
                    found = True
                    break
        else:
            # Sample some
            for _ in range(100):
                B = list(rng.sample(valid_bs, 3))
                if all_nine_distinct_coprime(A, B):
                    best_A, best_B = A, B
                    best_score = 9
                    found = True
                    break
        
        attempts += 1
        if attempts % 10000 == 0:
            pass  # could add progress tracking
    
    if found:
        return [best_A, best_B]
    
    # Phase 2: If 9 not found, do hill climbing to maximize score
    # But first, let's try a more systematic search with smaller ranges
    
    # Try A, B from [1, 30]
    small_range = list(range(1, 31))
    small_sets = list(combinations(small_range, 3))
    
    for A in small_sets:
        if time.time() >= DEADLINE - 5:
            break
        A = list(A)
        a1, a2, a3 = A
        for B in small_sets:
            B = list(B)
            s = score_pair(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]
            if s >= 9:
                return [best_A, best_B]
    
    # Phase 3: Hill climbing with random restarts
    def hill_climb(A, B, time_limit):
        t = time.time()
        A_cur = A[:]
        B_cur = B[:]
        s = score_pair(A_cur, B_cur)
        best_local_s = s
        best_local_A = A_cur[:]
        best_local_B = B_cur[:]
        
        while time.time() < time_limit and time.time() < DEADLINE:
            improved = False
            
            # Try changing one element of A
            for i in range(3):
                old = A_cur[i]
                for v in range(1, 129):
                    if v == old or v in A_cur:
                        continue
                    A2 = A_cur[:]
                    A2[i] = v
                    s2 = score_pair(A2, B_cur)
                    if s2 > s:
                        s = s2
                        A_cur = A2
                        improved = True
                        if s > best_local_s:
                            best_local_s = s
                            best_local_A = A_cur[:]
                            best_local_B = B_cur[:]
                        if s >= 9:
                            return s, A_cur, B_cur
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try changing one element of B
            for j in range(3):
                old = B_cur[j]
                for v in range(1, 129):
                    if v == old or v in B_cur:
                        continue
                    B2 = B_cur[:]
                    B2[j] = v
                    s2 = score_pair(A_cur, B2)
                    if s2 > s:
                        s = s2
                        B_cur = B2
                        improved = True
                        if s > best_local_s:
                            best_local_s = s
                            best_local_A = A_cur[:]
                            best_local_B = B_cur[:]
                        if s >= 9:
                            return s, A_cur, B_cur
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Stuck: perturb
            if rng.random() < 0.5:
                i = rng.randrange(3)
                while True:
                    v = rng.randint(1, 128)
                    if v not in A_cur:
                        break
                A_cur[i] = v
            else:
                j = rng.randrange(3)
                while True:
                    v = rng.randint(1, 128)
                    if v not in B_cur:
                        break
                B_cur[j] = v
            s = score_pair(A_cur, B_cur)
            if s > best_local_s:
                best_local_s = s
                best_local_A = A_cur[:]
                best_local_B = B_cur[:]
                if s >= 9:
                    return s, A_cur, B_cur
        
        return best_local_s, best_local_A, best_local_B
    
    restarts = 0
    while time.time() < DEADLINE - 2:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = score_pair(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
        if s >= 9:
            break
        per_climb = min(2.0, (DEADLINE - time.time()) * 0.2)
        s2, A2, B2 = hill_climb(A, B, time.time() + per_climb)
        if s2 > best_score:
            best_score = s2
            best_A, best_B = A2[:], B2[:]
        if best_score >= 9:
            break
        restarts += 1
    
    return [best_A, best_B]
