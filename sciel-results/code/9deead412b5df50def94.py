import time
import random
import math
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 70  # 20s safety margin before 90s kill

    def valid_pair(A, B):
        """Return the number of distinct pairwise-coprime values in A+B."""
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

    # --- Direct construction strategy ---
    # The sums a+b for a in A, b in B. We want 9 distinct sums, all pairwise coprime.
    # A powerful strategy: pick A and B such that all 9 sums are distinct primes,
    # or at least a large set of pairwise-coprime integers.

    # Strategy 1: Exhaustive search over structured families
    # Let A = {x, x+d, x+2d} (arithmetic progression) and B similarly.
    # Sums range from 2x+2y to 2x+2y+4d... not necessarily distinct.

    # Strategy 2: Random restarts with strong local search
    rng = random.Random(42)
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]

    def hill_climb(A, B, time_limit):
        """Iterated local search. Returns improved (score, A, B)."""
        t = time.time()
        A_cur = A[:]
        B_cur = B[:]
        s = valid_pair(A_cur, B_cur)
        best_local_s = s
        best_local_A = A_cur[:]
        best_local_B = B_cur[:]

        while time.time() < time_limit and time.time() < DEADLINE:
            # Try to improve A
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
            # Try to improve B
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
            # Stuck. Perturb.
            # Pick a random element in A or B and change it
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

    # Strategy 3: Targeted search for 9 pairwise-coprime sums
    # All 9 sums distinct and pairwise coprime.
    # If all 9 sums are primes > 2, they are pairwise coprime.
    # So: find A, B with 3 distinct elements in [1,128] such that all 9 a+b are distinct primes.
    # This is a strong constraint but might be satisfiable.

    def all_distinct_primes(A, B):
        sums = []
        for a in A:
            for b in B:
                s = a + b
                if s <= 2 or not _is_prime(s):
                    return False
                if s in sums:
                    return False
                sums.append(s)
        return True

    _prime_cache = {}
    def _is_prime(n):
        if n in _prime_cache:
            return _prime_cache[n]
        if n < 2:
            _prime_cache[n] = False
            return False
        if n < 4:
            _prime_cache[n] = True
            return True
        if n % 2 == 0 or n % 3 == 0:
            _prime_cache[n] = False
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                _prime_cache[n] = False
                return False
            i += 6
        _prime_cache[n] = True
        return True

    # Try to find A, B where all 9 sums are distinct primes.
    # Phase 1: deterministic scan over small anchor sets (elements <= 12).
    # This is ~36k evaluations and likely finds a 9-prime solution quickly.
    found_prime_9 = False
    small_range = list(range(1, 13))
    small_sets = list(combinations(small_range, 3))
    for A in small_sets:
        if time.time() >= DEADLINE - 5:
            break
        A = list(A)
        for B in small_sets:
            B = list(B)
            if all_distinct_primes(A, B):
                best_A, best_B, best_score = A, B, 9
                found_prime_9 = True
                break
        if found_prime_9:
            break

    # Phase 2: random search over full range if deterministic scan failed.
    if not found_prime_9:
        attempt = 0
        while time.time() < DEADLINE - 5 and not found_prime_9:
            A = sorted(rng.sample(range(1, 129), 3))
            B = sorted(rng.sample(range(1, 129), 3))
            if all_distinct_primes(A, B):
                best_A, best_B, best_score = A, B, 9
                found_prime_9 = True
                break
            attempt += 1

    if found_prime_9:
        return [best_A, best_B]

    # If prime-9 search didn't find one, do general hill climbing
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]

    restarts = 0
    while time.time() < DEADLINE:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = valid_pair(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
        if s >= 9:
            break
        # Local search with a time budget
        per_climb = min(1.0, (DEADLINE - time.time()) * 0.15)
        s2, A2, B2 = hill_climb(A, B, time.time() + per_climb)
        if s2 > best_score:
            best_score = s2
            best_A, best_B = A2[:], B2[:]
        if best_score >= 9:
            break
        restarts += 1

    # Final verification
    final_s = valid_pair(best_A, best_B)
    return [best_A, best_B]
