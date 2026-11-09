import math
import time
import random
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _is_pairwise_coprime(vals):
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if _is_pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin before 90s kill

    best_score = 0
    best_A = None
    best_B = None

    def update(A, B):
        nonlocal best_score, best_A, best_B
        s = _score(A, B)
        if s > best_score:
            best_score = s
            best_A = list(A)
            best_B = list(B)
            if best_score == 9:
                return True
        return False

    # Parity insight: for 9 distinct pairwise-coprime sums, all must be odd primes
    # (2 can't appear with 9 distinct sums since it forces a contradiction).
    # So A must be all-odd and B all-even (or vice versa).

    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(2, 129) if x % 2 == 0]

    # Pre-filter: only keep numbers that can participate in prime sums
    # For odd a, even b: a+b must be prime
    # For even a, odd b: a+b must be prime

    # Try to find a solution via structured search
    # Strategy: pick 3 odd numbers for A, 3 even for B, check if all 9 sums are distinct primes

    # First, try a smarter approach:
    # We need 9 distinct primes in A+B.
    # Let's enumerate small candidates.

    # Quick exhaustive-ish search with pruning
    # For each triple of odds, for each triple of evens, check
    # But that's C(64,3) * C(63,3) ≈ 41664 * 39711 ≈ 1.65B - too many.

    # Better: local search starting from good seeds.

    # Generate initial candidates
    random.seed(42)

    # Seed 1: known good structure
    # A = [1, 7, 13], B = [4, 10, 16]
    # Sums: 5,8,10,11,13,14,16,17,19 - wait, 8,10,14,16 are even, not coprime to each other
    # Actually gcd(8,10)=2, so this is invalid!
    # Let me recheck: the seed program says score 1.8, which means 9*0.2=1.8? No.
    # Score is value/seed. If seed score is the baseline...
    # Actually looking at the task: "The evaluator measures the raw value (higher is better).
    # The search score is value/seed"
    # So if current program scores 1.8, and it finds 9 sums, then seed must be 5.
    # But 9 distinct pairwise coprime sums would be the max possible.

    # Let me just search properly.

    # Try random pairs with parity constraint
    for _ in range(200000):
        if time.time() > deadline:
            break

        A = random.sample(odds, 3)
        B = random.sample(evens, 3)

        sums = set()
        valid = True
        for a in A:
            for b in B:
                s = a + b
                if not _is_prime(s):
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break

        if valid and len(sums) == 9:
            # All 9 sums are distinct primes, hence pairwise coprime
            if update(A, B):
                return [best_A, best_B]

    # If we didn't find 9, do local search to improve
    if best_A is None:
        # Start with a reasonable initial guess
        best_A = [1, 3, 5]
        best_B = [2, 4, 6]
        best_score = _score(best_A, best_B)

    # Local search: mutate one element at a time
    current_A = list(best_A)
    current_B = list(best_B)
    current_score = best_score

    # Determine parity constraint
    # If all A are odd, B should be even, and vice versa
    a_parity = current_A[0] % 2
    b_parity = current_B[0] % 2

    # Ensure consistent parity
    if a_parity == b_parity:
        # This won't give all-odd sums, so try to fix
        # Just proceed with whatever
        pass

    for iteration in range(50000):
        if time.time() > deadline:
            break

        if random.random() < 0.02:
            # Random restart
            if random.random() < 0.5:
                current_A = random.sample(odds, 3)
                current_B = random.sample(evens, 3)
            else:
                current_A = random.sample(evens, 3)
                current_B = random.sample(odds, 3)
            current_score = _score(current_A, current_B)
            if current_score > best_score:
                best_score = current_score
                best_A = list(current_A)
                best_B = list(current_B)
                if best_score == 9:
                    return [best_A, best_B]
            continue

        improved = False

        # Try mutating A
        for i in range(3):
            if time.time() > deadline:
                return [best_A, best_B]
            for _ in range(20):
                new_a = random.choice(odds if current_A[0] % 2 == 1 else evens)
                if new_a in current_A:
                    continue
                test_A = list(current_A)
                test_A[i] = new_a
                test_score = _score(test_A, current_B)
                if test_score > current_score:
                    current_A = test_A
                    current_score = test_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_A = list(current_A)
                        best_B = list(current_B)
                        if best_score == 9:
                            return [best_A, best_B]
                    break

        # Try mutating B
        if not improved:
            for i in range(3):
                if time.time() > deadline:
                    return [best_A, best_B]
                for _ in range(20):
                    new_b = random.choice(evens if current_B[0] % 2 == 0 else odds)
                    if new_b in current_B:
                        continue
                    test_B = list(current_B)
                    test_B[i] = new_b
                    test_score = _score(current_A, test_B)
                    if test_score > current_score:
                        current_B = test_B
                        current_score = test_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_A = list(current_A)
                            best_B = list(current_B)
                            if best_score == 9:
                                return [best_A, best_B]
                        break

    return [best_A, best_B]
