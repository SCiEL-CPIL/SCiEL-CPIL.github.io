"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute phi via linear-ish sieve up to a bound
    # We need phi for numbers up to ~10^12, but we can't sieve that.
    # Instead, we compute phi on-the-fly with a precomputed prime list.
    # For the search, we check runs of length up to 200.
    # We'll use a hybrid: sieve up to SIEVE_LIMIT, and trial division beyond.

    SIEVE_LIMIT = 10_000_000  # 10 million

    # Build phi array up to SIEVE_LIMIT using a modified sieve
    phi_arr = list(range(SIEVE_LIMIT + 1))
    for i in range(2, SIEVE_LIMIT + 1):
        if phi_arr[i] == i:  # i is prime
            for j in range(i, SIEVE_LIMIT + 1, i):
                phi_arr[j] -= phi_arr[j] // i

    # Build list of primes up to sqrt(10^12) = 10^6 for trial division
    def sieve_primes(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve_primes(1_000_000)  # primes up to 10^6

    def phi(x):
        """Compute Euler's totient of x using precomputed primes."""
        if x <= SIEVE_LIMIT:
            return phi_arr[x]
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def check_run(s, max_k=200):
        """Check the longest run of distinct totients starting at s.
        Returns the length k of the longest valid run."""
        seen = set()
        k = 0
        for i in range(max_k):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    # Search strategy:
    # 1. First, do a quick scan of small starts to find a good baseline.
    # 2. Then, search in larger ranges, focusing on starts where the
    #    odd numbers have distinct phi values (necessary condition).
    # 3. Use the fact that phi(2n) = 2*phi(n) for odd n to prune.

    best_k = 1
    best_s = 1

    # Phase 1: Quick scan of first 100000 starts
    for s in range(1, 100_001):
        k = check_run(s)
        if k > best_k:
            best_k = k
            best_s = s
        if time_remaining() < 20:
            break

    # Phase 2: Search in larger ranges
    # We'll search starts in blocks, skipping starts that are unlikely
    # to give long runs. A necessary condition for a long run is that
    # the odd numbers in the range have distinct phi values.
    # We'll sample starts more sparsely in larger ranges.

    search_start = 100_001
    step = 1000  # Check every 1000th start initially
    while search_start <= 10**12 - 200:
        if time_remaining() < 10:
            break
        # Check a batch of starts
        batch_end = min(search_start + 10000, 10**12 - 200)
        for s in range(search_start, batch_end + 1, step):
            k = check_run(s)
            if k > best_k:
                best_k = k
                best_s = s
                # If we found a good run, do a finer search around it
                if time_remaining() > 15:
                    fine_start = max(1, s - 50)
                    fine_end = min(s + 50, 10**12 - 200)
                    for s2 in range(fine_start, fine_end + 1):
                        k2 = check_run(s2)
                        if k2 > best_k:
                            best_k = k2
                            best_s = s2
        search_start += 10000

    # Phase 3: Finer search around the best start found
    if time_remaining() > 5 and best_s > 1:
        fine_start = max(1, best_s - 200)
        fine_end = min(best_s + 200, 10**12 - 200)
        for s in range(fine_start, fine_end + 1):
            k = check_run(s)
            if k > best_k:
                best_k = k
                best_s = s

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
