import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_PRODUCT = 223092870  # 2*3*5*7*11*13*17*19*23

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24  # 1000024
    size = max_val - lower + 1

    # Precompute small-prime parts using NumPy for speed
    # sp_part[v] = product of all prime factors < 24 of v
    # For v coprime to _PRODUCT, sp_part[v] = 1

    # Start with array of 1s
    sp_parts = np.ones(size, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # Find all multiples of p in [lower, max_val]
        first = ((lower + p - 1) // p) * p
        if first < lower:
            first += p

        # Create indices of multiples
        indices = np.arange(first, max_val + 1, p)
        idx_array = indices - lower  # Convert to array indices

        # For each multiple, divide out all factors of p and multiply p back
        vals = indices.astype(np.int64)
        while np.any(vals % p == 0):
            mask = vals % p == 0
            vals[mask] //= p

        # sp_parts[idx] *= p for each multiple
        # But we need to multiply by p for each power? No - we want the COMPLETE factor.
        # The complete factor is p^k where k is the exponent of p in the factorization.
        # So sp_part[v] should include p^k for each prime p < 24.
        # We're building it multiplicatively: for each p, multiply by p^k where k = v_p(v)

        # Actually, let me reconsider. We want the "complete factor composed of primes STRICTLY BELOW 24"
        # This means: for v = p1^a1 * p2^a2 * ... * pk^ak * m where pi < 24 and gcd(m, _PRODUCT)=1
        # The extracted factor is p1^a1 * p2^a2 * ... * pk^ak

        # So we need to compute this properly. Let me use a different approach:
        # For each p, compute p^v_p(v) for each v, then multiply all together.

    # Better approach: compute v_p(v) for each p, then build the product
    # Let me redo this more carefully

    # Initialize result as 1 for all
    result = np.ones(size, dtype=np.int64)

    for p in _SMALL_PRIMES:
        first = ((lower + p - 1) // p) * p
        if first < lower:
            first += p

        # Get all multiples of p
        multiples = np.arange(first, max_val + 1, p)
        idx = multiples - lower

        # Compute v_p(v) for each multiple
        # v_p(v) = largest k such that p^k divides v
        vals = multiples.astype(np.int64)
        # Divide by p repeatedly to count exponent
        exp = np.zeros(len(vals), dtype=np.int64)
        while np.any(vals % p == 0):
            mask = vals % p == 0
            exp[mask] += 1
            vals[mask] //= p

        # Compute p^exp for each
        p_powers = p ** exp  # This works for small exp values

        # Multiply into result
        result[idx] *= p_powers

    # Now result[i] is the small-prime part of (lower + i)
    # We want to find n in [lower, upper] such that the number of distinct values
    # in {result[n+1-lower], result[n+2-lower], ..., result[n+24-lower]} is minimized

    # Use sliding window with a count dictionary
    from collections import defaultdict

    # Initial window: n = lower, values at indices 1..24
    window_counts = defaultdict(int)
    for i in range(1, 25):
        window_counts[result[i]] += 1

    best_count = len(window_counts)
    best_n = lower

    # Slide window
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break

        out_idx = n - lower  # value n leaves the window
        in_idx = n + 24 - lower  # value n+24 enters the window

        out_val = result[out_idx]
        in_val = result[in_idx]

        window_counts[out_val] -= 1
        if window_counts[out_val] == 0:
            del window_counts[out_val]

        window_counts[in_val] += 1

        cur = len(window_counts)
        if cur < best_count:
            best_count = cur
            best_n = n
            if best_count == 1:
                break

    return best_n
