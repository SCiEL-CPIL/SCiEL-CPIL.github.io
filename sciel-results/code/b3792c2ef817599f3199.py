import time
import numpy as np
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill

    MAX_N = 256

    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # 1. Try ILP approach
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Build constraint matrix: A[p_idx, n-1] = 1 if p divides n
        A = np.zeros((num_primes, MAX_N), dtype=np.int8)
        for p_idx, p in enumerate(primes):
            for multiple in range(p, MAX_N + 1, p):
                A[p_idx, multiple - 1] = 1

        # Objective: Minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=np.float64)

        # Constraints: A @ x <= 1
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))

        # Integrality: all integer
        integrality = np.ones(MAX_N)

        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={'mip_rel_gap': 0})

        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]

            # Verify
            if len(selected) > 0:
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if gcd(selected[i], selected[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return [int(n) for n in selected]
        # If ILP fails or returns invalid, fall through
    except Exception:
        pass

    # 2. Fallback: Multi-start Greedy + Local Search
    best_set = set()
    best_sum = 0

    def run_greedy(order):
        S = [1]
        for n in order:
            if all(gcd(n, x) == 1 for x in S):
                S.append(n)
        return S

    # Try descending order
    S1 = run_greedy(range(256, 1, -1))
    if sum(S1) > best_sum:
        best_set = set(S1)
        best_sum = sum(S1)

    # Try ascending order (might pick small primes, allowing more numbers? No, we want max sum)
    # Ascending is likely worse for sum, but might allow more numbers?
    # Actually, for max sum, descending is usually better.
    # Let's try a "value density" or just random restarts if time permits.
    
    # Try a few random permutations if time allows
    import random
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 5: break
        perm = list(range(2, 257))
        random.shuffle(perm)
        S_rand = run_greedy(perm)
        if sum(S_rand) > best_sum:
            best_set = set(S_rand)
            best_sum = sum(S_rand)

    # Local Search Optimization
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_set)
        for a in S_list:
            if a == 1:
                continue
            # Try to replace a with b
            for b in range(a + 1, 257):
                if b in best_set:
                    continue
                new_set = (best_set - {a}) | {b}
                # Check if new_set is valid
                ok = True
                for x in new_set:
                    for y in new_set:
                        if x < y and gcd(x, y) != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new_sum = sum(new_set)
                    if new_sum > best_sum:
                        best_set = new_set
                        best_sum = new_sum
                        improved = True
                        break
        # Try adding a new number?
        if time.time() < deadline:
            for b in range(256, 1, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set):
                    best_set.add(b)
                    best_sum += b
                    improved = True
                    break

    return [int(x) for x in sorted(best_set)]
