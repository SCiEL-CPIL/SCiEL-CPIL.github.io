import time
import random
import math
from itertools import combinations, product

def construct():
    start = time.time()
    DEADLINE = start + 65  # safety margin before 90s kill

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

    def all_distinct_primes(A, B):
        """Check if all 9 sums are distinct primes > 2."""
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

    # Strategy 1: Algebraic search for 9 distinct primes
    # Key insight: fix A = {a1, a2, a3} and B = {b1, b2, b3}
    # The 9 sums form a grid. We search over structured families.
    
    # Phase 1: Search over small ranges where primes are dense
    # A and B elements in [1, 30] gives sums in [2, 60]
    # Number of primes in [2,60]: ~17, so 9 distinct primes is feasible
    
    found_prime_9 = False
    best_A = None
    best_B = None
    
    # Try structured families first for speed
    # Family 1: A = {x, x+d, x+2d}, B = {y, y+e, y+2e}
    for x in range(1, 15):
        if time.time() > DEADLINE - 5:
            break
        for d in range(1, 15):
            if x + 2*d > 128:
                continue
            A = [x, x+d, x+2*d]
            if len(set(A)) != 3:
                continue
            for y in range(1, 15):
                if time.time() > DEADLINE - 5:
                    break
                for e in range(1, 15):
                    if y + 2*e > 128:
                        continue
                    B = [y, y+e, y+2*e]
                    if len(set(B)) != 3:
                        continue
                    if all_distinct_primes(A, B):
                        found_prime_9 = True
                        best_A = A
                        best_B = B
                        break
                if found_prime_9:
                    break
            if found_prime_9:
                break
        if found_prime_9:
            break
    
    # Phase 2: Broader search over small sets if structured family didn't work
    if not found_prime_9:
        small_range = list(range(1, 25))
        small_sets = list(combinations(small_range, 3))
        for A in small_sets:
            if time.time() > DEADLINE - 5:
                break
            A = list(A)
            for B in small_sets:
                B = list(B)
                if all_distinct_primes(A, B):
                    found_prime_9 = True
                    best_A = A
                    best_B = B
                    break
            if found_prime_9:
                break
    
    # Phase 3: Random search over full range
    if not found_prime_9:
        rng = random.Random(42)
        attempt = 0
        while time.time() < DEADLINE - 5 and not found_prime_9:
            A = sorted(rng.sample(range(1, 129), 3))
            B = sorted(rng.sample(range(1, 129), 3))
            if all_distinct_primes(A, B):
                found_prime_9 = True
                best_A = A
                best_B = B
                break
            attempt += 1
    
    if found_prime_9:
        return [best_A, best_B]
    
    # Fallback: hill climbing to maximize pairwise coprime count
    def best_move_A(A, B):
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
            rng2 = random.Random()
            if rng2.random() < 0.5:
                i = rng2.randrange(3)
                while True:
                    v = rng2.randint(1, 128)
                    if v not in A_cur:
                        break
                A_cur[i] = v
            else:
                j = rng2.randrange(3)
                while True:
                    v = rng2.randint(1, 128)
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

    rng = random.Random(123)
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]

    while time.time() < DEADLINE:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = valid_pair(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
        if s >= 9:
            break
        per_climb = min(1.0, (DEADLINE - time.time()) * 0.15)
        s2, A2, B2 = hill_climb(A, B, time.time() + per_climb)
        if s2 > best_score:
            best_score = s2
            best_A, best_B = A2[:], B2[:]
        if best_score >= 9:
            break

    return [best_A, best_B]
