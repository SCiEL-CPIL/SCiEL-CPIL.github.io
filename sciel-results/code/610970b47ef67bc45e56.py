"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill

    def pairwise_coprime_check(sums_set):
        """Check if all elements in sums_set are pairwise coprime."""
        slist = list(sums_set)
        for i in range(len(slist)):
            for j in range(i + 1, len(slist)):
                if math.gcd(slist[i], slist[j]) != 1:
                    return False
        return True

    def eval_sums(A, B):
        """Return (num_distinct, is_valid) for A+B."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) <= 1:
            return len(sums), True
        return len(sums), pairwise_coprime_check(sums)

    # Parity constraint: use all-odd A and all-even B (or vice versa)
    # This ensures all sums are odd, eliminating factor 2 from coprimality concerns
    odd_vals = [v for v in range(1, 129) if v % 2 == 1]
    even_vals = [v for v in range(1, 129) if v % 2 == 0]

    # Precompute: for each odd a and even b, the sum s = a+b is odd
    # We need all 9 sums to be distinct and pairwise coprime (all odd)

    # Strategy: backtracking CSP on difference structure
    # For A = {a1, a2, a3} and B = {b1, b2, b3}:
    # Sums form a 3x3 grid: s[i][j] = a_i + b_j
    # s[i][j] - s[i][j'] = b_j - b_j' (row differences depend only on B)
    # s[i][j] - s[i'][j] = a_i - a_i' (column differences depend only on A)

    # Precompute valid odd sums in [2, 256] that are prime powers or primes
    # For pairwise coprimality among odd numbers, we want numbers with distinct prime factors

    # Build list of "good" odd numbers in [2,256] (odd, prime power or prime)
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for p in range(3, int(math.isqrt(n)) + 1, 2):
            if n % p == 0:
                return False
        return True

    def is_odd_prime_power(n):
        if n <= 1 or n % 2 == 0:
            return False
        if is_prime(n):
            return True
        # Check if n = p^k for odd prime p
        for p in range(3, int(math.isqrt(n)) + 1, 2):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
                return False
        return False

    good_odd_sums = set()
    for n in range(3, 257, 2):
        if is_odd_prime_power(n):
            good_odd_sums.add(n)

    # Direct search: try combinations of 3 odd values for A and 3 even values for B
    # Use heuristic: prefer values that produce prime-power sums

    # Score odd values by how many even values give them a good odd sum
    odd_scores = {}
    for a in odd_vals:
        cnt = 0
        for b in even_vals:
            if (a + b) in good_odd_sums:
                cnt += 1
        odd_scores[a] = cnt

    # Score even values similarly
    even_scores = {}
    for b in even_vals:
        cnt = 0
        for a in odd_vals:
            if (a + b) in good_odd_sums:
                cnt += 1
        even_scores[b] = cnt

    sorted_odds = sorted(odd_vals, key=lambda x: odd_scores[x], reverse=True)
    sorted_evens = sorted(even_vals, key=lambda x: even_scores[x], reverse=True)

    # Use top candidates for fast search
    top_odds = sorted_odds[:30]
    top_evens = sorted_evens[:30]

    best_score = 0
    best_A = None
    best_B = None

    # Phase 1: Systematic search over top candidates
    # Try all C(30,3) x C(30,3) is too many, so use guided search
    # Instead, try structured combinations

    # Phase 1a: Try small ranges first (likely to work)
    for a1 in top_odds[:15]:
        if time.time() > deadline - 10:
            break
        for a2 in top_odds[:15]:
            if a2 <= a1:
                continue
            if time.time() > deadline - 10:
                break
            for a3 in top_odds[:15]:
                if a3 <= a2:
                    continue
                if time.time() > deadline - 10:
                    break
                A = [a1, a2, a3]
                # For this A, find best B
                # Compute which sums each b would create
                b_candidates = []
                for b in top_evens[:20]:
                    sums_for_b = [a + b for a in A]
                    # Check if these 3 sums are all good and coprime to each other
                    valid_b = True
                    for s in sums_for_b:
                        if s not in good_odd_sums:
                            valid_b = False
                            break
                    if valid_b:
                        for i in range(3):
                            for j in range(i + 1, 3):
                                if math.gcd(sums_for_b[i], sums_for_b[j]) != 1:
                                    valid_b = False
                                    break
                            if not valid_b:
                                break
                    if valid_b:
                        b_candidates.append(b)

                if len(b_candidates) >= 3:
                    # Try combinations of 3 from b_candidates
                    for b1, b2, b3 in combinations(b_candidates[:15], 3):
                        B = [b1, b2, b3]
                        sums = set()
                        for a in A:
                            for b in B:
                                sums.add(a + b)
                        if len(sums) == 9 and pairwise_coprime_check(sums):
                            best_score = 9
                            best_A = A[:]
                            best_B = B[:]
                            break
                    if best_score >= 9:
                        break
            if best_score >= 9:
                break
        if best_score >= 9:
            break

    if best_score < 9:
        # Phase 1b: Wider search
        for a1 in top_odds[:20]:
            if time.time() > deadline - 10:
                break
            for a2 in top_odds[:20]:
                if a2 <= a1:
                    continue
                if time.time() > deadline - 10:
                    break
                for a3 in top_odds[:20]:
                    if a3 <= a2:
                        continue
                    if time.time() > deadline - 10:
                        break
                    A = [a1, a2, a3]
                    b_candidates = []
                    for b in top_evens[:25]:
                        sums_for_b = [a + b for a in A]
                        valid_b = True
                        for s in sums_for_b:
                            if s not in good_odd_sums:
                                valid_b = False
                                break
                        if valid_b:
                            for i in range(3):
                                for j in range(i + 1, 3):
                                    if math.gcd(sums_for_b[i], sums_for_b[j]) != 1:
                                        valid_b = False
                                        break
                                if not valid_b:
                                    break
                        if valid_b:
                            b_candidates.append(b)

                    if len(b_candidates) >= 3:
                        for b1, b2, b3 in combinations(b_candidates[:20], 3):
                            B = [b1, b2, b3]
                            sums = set()
                            for a in A:
                                for b in B:
                                    sums.add(a + b)
                            if len(sums) == 9 and pairwise_coprime_check(sums):
                                best_score = 9
                                best_A = A[:]
                                best_B = B[:]
                                break
                        if best_score >= 9:
                            break
                    if best_score >= 9:
                        break
                if best_score >= 9:
                    break
            if best_score >= 9:
                break

    if best_score < 9:
        # Phase 2: Random search with parity constraint
        import random
        random.seed(123)
        for _ in range(50000):
            if time.time() > deadline - 10:
                break
            A = random.sample(top_odds[:40], 3)
            B = random.sample(top_evens[:40], 3)
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9 and pairwise_coprime_check(sums):
                best_score = 9
                best_A = A[:]
                best_B = B[:]
                break

    if best_score < 9:
        # Phase 3: Local search from seed
        import random
        random.seed(42)
        A = [1, 7, 13]
        B = [4, 10, 16]
        # Ensure parity: make A odd, B even
        for _ in range(100000):
            if time.time() > deadline - 10:
                break
            # Randomly mutate
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                A[idx] = random.choice(odd_vals)
                if len(set(A)) != 3:
                    continue
            else:
                idx = random.randint(0, 2)
                B[idx] = random.choice(even_vals)
                if len(set(B)) != 3:
                    continue

            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9 and pairwise_coprime_check(sums):
                best_score = 9
                best_A = A[:]
                best_B = B[:]
                break

    if best_A is not None and best_B is not None:
        return [best_A, best_B]
    else:
        # Fallback: return seed
        return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
