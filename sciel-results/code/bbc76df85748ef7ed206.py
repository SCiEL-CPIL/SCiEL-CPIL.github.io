import time
import random
import math
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 80  # 10s safety margin before 90s kill

    # Precompute primes up to 256 (max sum is 128+128=256)
    _is_prime_arr = [True] * 257
    _is_prime_arr[0] = False
    _is_prime_arr[1] = False
    for i in range(2, int(256**0.5) + 1):
        if _is_prime_arr[i]:
            for j in range(i*i, 257, i):
                _is_prime_arr[j] = False

    # Precompute gcd table for numbers up to 256
    # gcd_table[i][j] = gcd(i, j)
    # Actually, we can just use math.gcd which is fast enough for small numbers
    # But let's precompute pairwise coprime check for the 9 sums
    # We'll just use math.gcd directly in valid_pair for clarity

    def valid_pair(A, B):
        """Return the maximum number of distinct pairwise-coprime values in A+B."""
        if len(A) != 3 or len(B) != 3:
            return 0
        if len(set(A)) != 3 or len(set(B)) != 3:
            return 0
        for x in A + B:
            if x < 1 or x > 128:
                return 0
        
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        sum_list = list(sums)
        n = len(sum_list)
        
        if n == 0:
            return 0
        
        # Precompute conflicts
        conflict = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(sum_list[i], sum_list[j]) != 1:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        # Find maximum independent set by checking all subsets
        max_count = 0
        for mask in range(1 << n):
            is_independent = True
            for i in range(n):
                if mask & (1 << i):
                    if mask & conflict[i]:
                        is_independent = False
                        break
            if is_independent:
                count = bin(mask).count('1')
                if count > max_count:
                    max_count = count
        
        return max_count

    def all_distinct_primes(A, B):
        """Check if all 9 sums are distinct primes."""
        sums = []
        for a in A:
            for b in B:
                s = a + b
                if s > 256 or not _is_prime_arr[s]:
                    return False
                if s in sums:
                    return False
                sums.append(s)
        return True

    # --- Phase 1: Deterministic search for 9 distinct primes ---
    # Strategy: Use the Diophantine structure.
    # If a1+b1=p1, a1+b2=p2, a1+b3=p3, then b1=p1-a1, b2=p2-a1, b3=p3-a1.
    # Then a2+b1=p4 => a2 = p4-b1 = p4-p1+a1.
    # a2+b2=p5 => a2 = p5-b2 = p5-p2+a1.
    # So p4-p1 = p5-p2 => p4-p5 = p1-p2.
    # Similarly, a3 = p6-b1 = p6-p1+a1, and a3+b2=p7 => a3=p7-p2+a1 => p6-p1=p7-p2.
    # And a1+b1=p1, a2+b1=p4, a3+b1=p6.
    # a2+b3=p8 => a2=p8-b3=p8-p3+a1. So p4-p1=p8-p3.
    # a3+b3=p9 => a3=p9-p3+a1. So p6-p1=p9-p3.
    
    # So the primes must satisfy:
    # p4 - p5 = p1 - p2
    # p6 - p7 = p1 - p2  (wait, p6-p1=p7-p2 => p6-p7=p1-p2)
    # p4 - p8 = p1 - p3
    # p6 - p9 = p1 - p3
    
    # Let's verify:
    # a1 = x
    # b1 = p1 - x
    # b2 = p2 - x
    # b3 = p3 - x
    # a2 = p4 - b1 = p4 - p1 + x
    # a3 = p6 - b1 = p6 - p1 + x
    # Check a2 + b2 = (p4 - p1 + x) + (p2 - x) = p4 - p1 + p2. This must equal p5.
    # So p5 = p4 - p1 + p2 => p4 - p5 = p1 - p2. Correct.
    # Check a2 + b3 = (p4 - p1 + x) + (p3 - x) = p4 - p1 + p3. This must equal p8.
    # So p8 = p4 - p1 + p3 => p4 - p8 = p1 - p3. Correct.
    # Check a3 + b2 = (p6 - p1 + x) + (p2 - x) = p6 - p1 + p2. This must equal p7.
    # So p7 = p6 - p1 + p2 => p6 - p7 = p1 - p2. Correct.
    # Check a3 + b3 = (p6 - p1 + x) + (p3 - x) = p6 - p1 + p3. This must equal p9.
    # So p9 = p6 - p1 + p3 => p6 - p9 = p1 - p3. Correct.
    
    # So we can iterate over p1, p2, p3, p4, p6 (5 primes) and derive p5, p7, p8, p9.
    # Then check if p5, p7, p8, p9 are primes.
    # Then check if x (which is a1) can be chosen such that all a_i, b_j are in [1, 128] and distinct.
    # a1 = x
    # a2 = p4 - p1 + x
    # a3 = p6 - p1 + x
    # b1 = p1 - x
    # b2 = p2 - x
    # b3 = p3 - x
    
    # Constraints:
    # 1 <= x <= 128
    # 1 <= p4 - p1 + x <= 128
    # 1 <= p6 - p1 + x <= 128
    # 1 <= p1 - x <= 128
    # 1 <= p2 - x <= 128
    # 1 <= p3 - x <= 128
    # All 6 values distinct.
    
    # This is a very efficient search!
    # Primes up to 256.
    primes = [i for i in range(2, 257) if _is_prime_arr[i]]
    
    found = False
    best_A = None
    best_B = None
    
    # To reduce search space, we can fix p1, p2, p3, p4, p6 from the primes list.
    # But 5 nested loops over ~50 primes is 300M iterations. Too slow.
    # We can prune.
    # p5 = p4 - p1 + p2 must be prime.
    # p7 = p6 - p1 + p2 must be prime.
    # p8 = p4 - p1 + p3 must be prime.
    # p9 = p6 - p1 + p3 must be prime.
    
    # Let's try a smaller range of primes first, or use random sampling.
    # Actually, let's just do a random search over 5 primes, check the 4 derived, then check bounds.
    # This is much faster.
    
    rng = random.Random(42)
    
    # Try random combinations of 5 primes
    for _ in range(100000):
        if time.time() >= DEADLINE - 10:
            break
        p1, p2, p3, p4, p6 = rng.sample(primes, 5)
        p5 = p4 - p1 + p2
        p7 = p6 - p1 + p2
        p8 = p4 - p1 + p3
        p9 = p6 - p1 + p3
        
        if p5 < 2 or p5 > 256 or p7 < 2 or p7 > 256 or p8 < 2 or p8 > 256 or p9 < 2 or p9 > 256:
            continue
        if not _is_prime_arr[p5] or not _is_prime_arr[p7] or not _is_prime_arr[p8] or not _is_prime_arr[p9]:
            continue
        
        # All 9 are primes. Now find x.
        # x must satisfy:
        # max(1, 1-(p4-p1), 1-(p6-p1), p1-128) <= x <= min(128, 128-(p4-p1), 128-(p6-p1), p1-1)
        # Also x must make all 6 values distinct.
        
        low = max(1, 1 - (p4 - p1), 1 - (p6 - p1), p1 - 128)
        high = min(128, 128 - (p4 - p1), 128 - (p6 - p1), p1 - 1)
        
        if low > high:
            continue
        
        # Try a few x values in [low, high]
        # To be fast, just pick a few random ones or scan a small subset
        # Since the range is small, we can just scan all if range is small
        if high - low < 100:
            for x in range(low, high + 1):
                a1 = x
                a2 = p4 - p1 + x
                a3 = p6 - p1 + x
                b1 = p1 - x
                b2 = p2 - x
                b3 = p3 - x
                
                vals = [a1, a2, a3, b1, b2, b3]
                if len(set(vals)) == 6:
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    # Verify
                    if valid_pair(A, B) == 9:
                        return [A, B]
        else:
            # Sample a few x
            for _ in range(20):
                x = rng.randint(low, high)
                a1 = x
                a2 = p4 - p1 + x
                a3 = p6 - p1 + x
                b1 = p1 - x
                b2 = p2 - x
                b3 = p3 - x
                
                vals = [a1, a2, a3, b1, b2, b3]
                if len(set(vals)) == 6:
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    if valid_pair(A, B) == 9:
                        return [A, B]

    # --- Phase 2: Fallback to local search if 9 primes not found ---
    def best_move_A(A, B):
        """Try all single-element swaps in A; return best (score, A_new)."""
        base = valid_pair(A, B)
        best_s = base
        best_a = A[:]
        for i in range(3):
            for v in range(1, 129):
                if v in A:
                    continue
                A2 = A[:]
                A2[i] = v
                s = valid_pair(A2, B)
                if s > best_s:
                    best_s = s
                    best_a = A2
        return best_s, best_a

    def best_move_B(A, B):
        """Try all single-element swaps in B; return best (score, B_new)."""
        base = valid_pair(A, B)
        best_s = base
        best_b = B[:]
        for j in range(3):
            for v in range(1, 129):
                if v in B:
                    continue
                B2 = B[:]
                B2[j] = v
                s = valid_pair(A, B2)
                if s > best_s:
                    best_s = s
                    best_b = B2
        return best_s, best_b

    def hill_climb(A, B, time_limit):
        """Iterated local search. Returns improved (score, A, B)."""
        A_cur = A[:]
        B_cur = B[:]
        s = valid_pair(A_cur, B_cur)
        best_local_s = s
        best_local_A = A_cur[:]
        best_local_B = B_cur[:]

        while time.time() < time_limit and time.time() < DEADLINE:
            s_new, A_new = best_move_A(A_cur, B_cur)
            if s_new > s:
                s = s_new
                A_cur = A_new
                if s > best_local_s:
                    best_local_s = s
                    best_local_A = A_cur[:]
                    best_local_B = B_cur[:]
                    if s >= 9:
                        return s, A_cur, B_cur
                continue
            s_new, B_new = best_move_B(A_cur, B_cur)
            if s_new > s:
                s = s_new
                B_cur = B_new
                if s > best_local_s:
                    best_local_s = s
                    best_local_A = A_cur[:]
                    best_local_B = B_cur[:]
                    if s >= 9:
                        return s, A_cur, B_cur
                continue
            # Perturb
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
            s = valid_pair(A_cur, B_cur)
            if s > best_local_s:
                best_local_s = s
                best_local_A = A_cur[:]
                best_local_B = B_cur[:]
                if s >= 9:
                    return s, A_cur, B_cur

        return best_local_s, best_local_A, best_local_B

    best_score = 0
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]

    while time.time() < DEADLINE - 2:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = valid_pair(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
        if s >= 9:
            break
        per_climb = min(1.0, (DEADLINE - time.time()) * 0.1)
        s2, A2, B2 = hill_climb(A, B, time.time() + per_climb)
        if s2 > best_score:
            best_score = s2
            best_A, best_B = A2[:], B2[:]
        if best_score >= 9:
            break

    return [best_A, best_B]
