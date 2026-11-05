import time
import random
import math

def construct():
    start = time.time()
    DEADLINE = start + 80  # 10s safety margin before 90s kill

    # Precompute primes up to 256 (max sum is 128+128=256)
    is_prime = [False] * 257
    is_prime[2] = True
    is_prime[3] = True
    for i in range(5, 257, 2):
        is_prime[i] = True
    for i in range(5, 257):
        if is_prime[i]:
            for j in range(i * i, 257, i):
                is_prime[j] = False
    # Fix: 1 is not prime
    is_prime[0] = False
    is_prime[1] = False

    rng = random.Random(12345)

    # --- Fast scoring: number of distinct pairwise-coprime sums in A+B ---
    # Greedy: sort sums, pick each if coprime to all previously chosen
    def score_pair(A, B):
        sums = []
        for a in A:
            for b in B:
                s = a + b
                sums.append(s)
        # Get distinct sums
        seen = set()
        distinct = []
        for s in sums:
            if s not in seen:
                seen.add(s)
                distinct.append(s)
        distinct.sort()
        chosen = []
        count = 0
        for s in distinct:
            ok = True
            for t in chosen:
                if math.gcd(s, t) != 1:
                    ok = False
                    break
            if ok:
                chosen.append(s)
                count += 1
        return count

    # --- Strategy 1: Find A, B where all 9 sums are distinct primes ---
    # For a fixed A, the 9 sums are {a_i + b_j : i,j in {0,1,2}}.
    # Quick filter: b must be such that all a_i + b are primes > 2 (odd primes, so odd).
    # Since all primes > 2 are odd, we need a_i + b odd for all i.
    # This means: if any a_i is even, b must be odd; if all a_i are odd, b must be odd too.
    # Actually: a_i + b odd means a_i and b have opposite parity.
    # So all a_i must have the same parity (all odd or all even), and b must have the opposite.
    
    def quick_b_filter(A):
        """Return list of b in [1,128] where all a+b are odd primes."""
        # Check parity constraint
        parities = [a % 2 for a in A]
        if len(set(parities)) != 1:
            return []  # Can't have all sums odd
        # All a have same parity p; b must have parity 1-p
        target_parity = 1 - parities[0]
        candidates = []
        for b in range(1, 129):
            if b % 2 != target_parity:
                continue
            # Check all a+b are primes
            valid = True
            for a in A:
                s = a + b
                if s > 256 or not is_prime[s]:
                    valid = False
                    break
            if valid:
                candidates.append(b)
        return candidates

    def check_nine(A, B):
        """Check if all 9 sums a+b are distinct primes."""
        seen = set()
        for a in A:
            for b in B:
                s = a + b
                if s in seen:
                    return False
                seen.add(s)
        return True

    found = False
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]

    # Exhaustive-ish search for 9-prime solution
    # Try all A with all-odd or all-even elements, then check B
    all_odds = list(range(1, 129, 2))   # 64 values
    all_evens = list(range(2, 129, 2))  # 64 values
    
    # Phase 1: Try structured A (arithmetic progressions, small values)
    t1 = time.time()
    
    # Try small A values first (more likely to have primes in sums)
    small_odds = list(range(1, 50, 2))
    small_evens = list(range(2, 50, 2))
    
    for A in [sorted(x) for x in combinations(small_odds, 3)]:
        if time.time() > DEADLINE - 20:
            break
        B_candidates = quick_b_filter(A)
        if len(B_candidates) < 3:
            continue
        for B in [sorted(x) for x in combinations(B_candidates, 3)]:
            if check_nine(A, B):
                return [A, B]
    
    # Phase 2: Broader search with random A
    t2 = time.time()
    attempts = 0
    while time.time() < DEADLINE - 15:
        # Random A with all-odd or all-even
        if rng.random() < 0.5:
            A = sorted(rng.sample(all_odds, 3))
        else:
            A = sorted(rng.sample(all_evens, 3))
        
        B_candidates = quick_b_filter(A)
        if len(B_candidates) >= 3:
            # Try a few random B from candidates
            for _ in range(min(5, len(B_candidates))):
                B = sorted(rng.sample(B_candidates, 3))
                if check_nine(A, B):
                    return [A, B]
        
        attempts += 1
        if attempts % 10000 == 0:
            # Update best from random sampling
            pass

    # --- Strategy 2: General hill-climb for max pairwise-coprime distinct sums ---
    # Fast local search
    def hill_climb(A, B, time_limit):
        A_cur = A[:]
        B_cur = B[:]
        s = score_pair(A_cur, B_cur)
        best_local_s = s
        best_local_A = A_cur[:]
        best_local_B = B_cur[:]

        while time.time() < time_limit and time.time() < DEADLINE - 5:
            improved = False
            # Try improving A
            for i in range(3):
                for v in range(1, 129):
                    if v in A_cur:
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
                        break
                if improved:
                    break
            if not improved:
                # Try improving B
                for j in range(3):
                    for v in range(1, 129):
                        if v in B_cur:
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
                            break
                    if improved:
                        break
            if not improved:
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
                s = score_pair(A_cur, B_cur)
                if s > best_local_s:
                    best_local_s = s
                    best_local_A = A_cur[:]
                    best_local_B = B_cur[:]
                if s >= 9:
                    return s, A_cur, B_cur

        return best_local_s, best_local_A, best_local_B

    # Random restarts with hill climbing
    restarts = 0
    while time.time() < DEADLINE - 10:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = score_pair(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
        if s >= 9:
            break
        per_climb = min(1.5, (DEADLINE - 10 - time.time()) * 0.2)
        s2, A2, B2 = hill_climb(A, B, time.time() + per_climb)
        if s2 > best_score:
            best_score = s2
            best_A, best_B = A2[:], B2[:]
        if best_score >= 9:
            break
        restarts += 1

    return [best_A, best_B]
