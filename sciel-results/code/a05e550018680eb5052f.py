"""Combined program: composite-pair partitioning + local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def _sieve_primes(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, n + 1) if is_prime[i]]

def _pair_primes(primes, n):
    """Pair small primes with large primes to form composites <= n.
    Return (composites, leftover_primes) where composites are p*q <= n
    with p < q, all primes used at most once, and composites are
    pairwise coprime (guaranteed since each prime is used once).
    Greedy: for each small prime in ascending order, pick the largest
    unused prime q such that p*q <= n and q > p.
    """
    used = [False] * (n + 1)
    composites = []
    leftover = []
    for p in primes:
        if used[p]:
            continue
        # Find largest q > p, unused, with p*q <= n
        best_q = None
        for q in range(n // p, p, -1):
            if q <= n and not used[q] and q in primes_set_global:
                # q must be prime (we only consider primes)
                best_q = q
                break
        if best_q is not None and best_q > p:
            c = p * best_q
            if c <= n:
                composites.append(c)
                used[p] = True
                used[best_q] = True
        else:
            # Cannot pair; keep p as leftover
            leftover.append(p)
    return composites, leftover

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill
    n = 256

    primes = _sieve_primes(n)
    primes_set_global_local = set(primes)

    # --- Build composite pairing ---
    # We need a global primes set for the pairing function; use local
    used = [False] * (n + 1)
    composites = []
    leftover = []
    for p in primes:
        if used[p]:
            continue
        # Find largest q > p, unused, prime, with p*q <= n
        best_q = None
        for q in range(n // p, p, -1):
            if q <= n and not used[q] and q in primes_set_global_local:
                best_q = q
                break
        if best_q is not None and best_q > p:
            c = p * best_q
            if c <= n:
                composites.append(c)
                used[p] = True
                used[best_q] = True
            else:
                leftover.append(p)
        else:
            leftover.append(p)

    candidate = [1] + composites + leftover
    candidate_sum = sum(candidate)

    # --- Sanity: verify pairwise coprime ---
    candidate_ok = True
    cand_set = set(candidate)
    if len(cand_set) != len(candidate):
        candidate_ok = False
    else:
        for i in range(len(candidate)):
            for j in range(i + 1, len(candidate)):
                if gcd(candidate[i], candidate[j]) != 1:
                    candidate_ok = False
                    break
            if not candidate_ok:
                break

    # Fallback to primes + 1 if pairing broke something
    if not candidate_ok:
        candidate = [1] + primes
        candidate_sum = sum(candidate)

    best_set = sorted(candidate)
    best_sum = candidate_sum
    best_in_set = set(best_set)

    # --- Local search: add any number coprime to all in best_set ---
    for a in range(n, 1, -1):
        if time.time() - start_time > deadline:
            break
        if a in best_in_set:
            continue
        can_add = True
        for b in best_set:
            if gcd(a, b) != 1:
                can_add = False
                break
        if can_add:
            best_set.append(a)
            best_in_set.add(a)
            best_sum += a

    # --- Local search: swaps ---
    max_swap_passes = 5
    for _pass in range(max_swap_passes):
        if time.time() - start_time > deadline:
            break
        improved = False
        for b in sorted(best_set, reverse=True):
            if time.time() - start_time > deadline:
                break
            if b == 1:
                continue
            others = [x for x in best_set if x != b]
            for a in range(n, b, -1):
                if a in best_in_set:
                    continue
                can_add = True
                for c in others:
                    if gcd(a, c) != 1:
                        can_add = False
                        break
                if can_add:
                    best_set.remove(b)
                    best_in_set.discard(b)
                    best_set.append(a)
                    best_in_set.add(a)
                    best_sum += a - b
                    improved = True
                    break
        if not improved:
            break

    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
