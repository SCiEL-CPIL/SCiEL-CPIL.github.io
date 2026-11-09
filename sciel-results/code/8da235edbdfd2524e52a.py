"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random

DEADLINE = time.time() + 82

N = 128

# Precompute gcd table
GCD = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        GCD[i][j] = gcd(i, j)
        GCD[j][i] = GCD[i][j]


def _is_valid_addition(S, x):
    """Check if adding x to set S keeps the property: no three distinct with equal pairwise gcds."""
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            g_ab = GCD[a][b]
            g_ax = GCD[a][x]
            if g_ab == g_ax:
                g_bx = GCD[b][x]
                if g_bx == g_ab:
                    return False
    return True


def _greedy_forward(order, S=None):
    """Greedy: add elements in given order if valid."""
    if S is None:
        S = []
    S_set = set(S)
    for x in order:
        if x in S_set:
            continue
        if _is_valid_addition(S, x):
            S.append(x)
            S_set.add(x)
    return S


def _local_search(S, rng, max_iterations=5000):
    """Try to add elements from [1,128] not in S via random orderings."""
    S = list(S)
    S_set = set(S)
    not_in = [x for x in range(1, N + 1) if x not in S_set]
    for _ in range(max_iterations):
        if time.time() > DEADLINE:
            break
        if not not_in:
            break
        idx = rng.randrange(len(not_in))
        x = not_in[idx]
        if _is_valid_addition(S, x):
            S.append(x)
            S_set.add(x)
            not_in[idx] = not_in[-1]
            not_in.pop()
    return S


def _random_greedy(rng):
    """Random permutation greedy."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return _greedy_forward(order)


def _construct_with_strategy(order, rng, max_ls_iter=500):
    """Build a candidate via greedy on order, then local search."""
    S = _greedy_forward(order)
    S = _local_search(S, rng, max_ls_iter)
    return S


def _smallest_prime_factor(x):
    if x < 2:
        return 0
    d = 2
    while d * d <= x:
        if x % d == 0:
            return d
        d += 1
    return x


def _omega(x):
    """Count distinct prime factors."""
    cnt = 0
    n = x
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        cnt += 1
    return cnt


def construct():
    best = None
    best_len = 0
    rng = Random(42)

    def consider(S):
        nonlocal best, best_len
        if S and len(S) > best_len:
            best = list(S)
            best_len = len(S)

    # Strategy 1: Ascending
    S1 = _construct_with_strategy(list(range(1, N + 1)), rng, 200)
    consider(S1)

    # Strategy 2: Descending
    S2 = _construct_with_strategy(list(range(N, 0, -1)), rng, 200)
    consider(S2)

    # Strategy 3: By smallest prime factor (descending - larger spf first)
    S3 = _construct_with_strategy(sorted(range(1, N + 1), key=lambda x: -_smallest_prime_factor(x)), rng, 200)
    consider(S3)

    # Strategy 4: By omega (ascending - fewer distinct prime factors first)
    S4 = _construct_with_strategy(sorted(range(1, N + 1), key=lambda x: _omega(x)), rng, 200)
    consider(S4)

    # Strategy 5: By omega (descending)
    S5 = _construct_with_strategy(sorted(range(1, N + 1), key=lambda x: -_omega(x)), rng, 200)
    consider(S5)

    # Strategy 6: Odd numbers first
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    S6 = _construct_with_strategy(odds + evens, rng, 200)
    consider(S6)

    # Strategy 7: Numbers coprime to 6 first
    coprime6 = [x for x in range(1, N + 1) if x % 2 != 0 and x % 3 != 0]
    rest6 = [x for x in range(1, N + 1) if x not in coprime6]
    S7 = _construct_with_strategy(coprime6 + rest6, rng, 200)
    consider(S7)

    # Strategy 8: Prime powers first
    def is_prime_power(x):
        if x < 2:
            return False
        n = x
        d = 2
        while d * d <= n:
            if n % d == 0:
                while n % d == 0:
                    n //= d
                if n > 1:
                    return False
                return True
            d += 1
        return True

    prime_powers = [x for x in range(1, N + 1) if is_prime_power(x)]
    non_pp = [x for x in range(1, N + 1) if x not in prime_powers]
    S8 = _construct_with_strategy(prime_powers + non_pp, rng, 200)
    consider(S8)

    # Strategy 9: Many random greedy + local search
    num_random = 150
    for i in range(num_random):
        if time.time() > DEADLINE:
            break
        S = _construct_with_strategy(None, rng, 300)
        if S is None:
            S = _random_greedy(rng)
            S = _local_search(S, rng, 300)
        consider(S)

    # Strategy 10: More random with different seeds
    for seed_val in [1, 2, 3, 7, 13, 42, 100, 999, 12345, 54321]:
        if time.time() > DEADLINE:
            break
        rng2 = Random(seed_val)
        S = _random_greedy(rng2)
        S = _local_search(S, rng2, 500)
        consider(S)

    # Strategy 11: Improve best with more local search iterations
    if best:
        for i in range(20):
            if time.time() > DEADLINE:
                break
            S = _local_search(list(best), rng, 2000)
            consider(S)

    # Strategy 12: Try removing an element and replacing with a better one
    if best:
        best_sorted = sorted(best)
        not_in = [x for x in range(1, N + 1) if x not in set(best_sorted)]
        for trial in range(50):
            if time.time() > DEADLINE:
                break
            S = list(best_sorted)
            # Try removing a random element
            if len(S) <= 1:
                break
            rm_idx = rng.randrange(len(S))
            S.pop(rm_idx)
            # Try adding from not_in
            for x in not_in:
                if _is_valid_addition(S, x):
                    S.append(x)
                    consider(S)
                    break

    if best is None or len(best) == 0:
        return [1]

    return sorted(best)
# EVOLVE-BLOCK-END
