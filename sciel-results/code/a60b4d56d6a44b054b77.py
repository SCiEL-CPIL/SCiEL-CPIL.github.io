import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 80  # Safety margin before 90s kill
    LIMIT = 10000

    # 1. Generate primes using Sieve of Eratosthenes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False

    primes = [i for i, p in enumerate(is_prime) if p]
    n = len(primes)

    # 2. Precompute next_prime_ge[x]: smallest prime >= x, or -1 if none
    next_prime_ge = [-1] * (LIMIT + 2)
    next_p = -1
    for i in range(LIMIT, -1, -1):
        if is_prime[i]:
            next_p = i
        next_prime_ge[i] = next_p

    # 3. Greedy search over starting pairs
    # For each pair (p1, p2) with gap g1, greedily extend by choosing the
    # smallest prime >= p_curr + g_curr. This is optimal for fixed prefix.
    best_len = 1
    best_seq = [primes[0]] if primes else []

    # Upper bound: max possible length from p1 with min gap 1
    # Actually gaps are at least 2 for odd primes (except 2->3 gap=1)
    # Simple bound: (LIMIT - p1) // 2 + 1 is a loose upper bound on count
    # But we use (LIMIT - p1) // g1 + 1 for tighter bound per pair

    for i in range(n):
        p1 = primes[i]
        # If even with gap 1 we can't beat best_len, break outer loop
        if (LIMIT - p1) < best_len:
            break

        for j in range(i + 1, n):
            p2 = primes[j]
            g1 = p2 - p1

            # Upper bound on length from this pair
            max_possible_len = (LIMIT - p1) // g1 + 1
            if max_possible_len <= best_len:
                break

            # Greedy extension
            seq = [p1, p2]
            p_curr = p2
            g_curr = g1
            curr_len = 2

            while True:
                target = p_curr + g_curr
                if target > LIMIT:
                    break
                p_next = next_prime_ge[target]
                if p_next == -1 or p_next > LIMIT:
                    break
                g_next = p_next - p_curr
                seq.append(p_next)
                curr_len += 1
                p_curr = p_next
                g_curr = g_next

            if curr_len > best_len:
                best_len = curr_len
                best_seq = seq

            # Time check
            if time.time() - start_time > DEADLINE:
                return best_seq

    return best_seq
