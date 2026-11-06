import time
import numpy as np
from math import isqrt, gcd

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    Powerful = a^2 * b^3 where b is square-free.
    """
    max_b = int(limit ** (1.0 / 3.0)) + 2
    # Sieve for square-free numbers up to max_b
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[1] = True

    powerful = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return powerful

def construct():
    t0 = time.time()
    deadline = t0 + 140  # 140s safety margin (evaluator kills at 150s)

    best_result = None
    best_count = 0

    # Strategy: Use FFT convolution to find N with many representations
    # Try progressively larger limits, using FFT to efficiently find the best N
    # For each limit L, powerful numbers are <= L, so N can be up to 2L
    # We want N in [2, 10^18], but practical limits are constrained by memory/time

    # Pre-compute powerful numbers for several limit sizes
    # The key insight: larger limit -> more powerful numbers -> potentially more representations
    # But we need N <= 10^18, so limit can be up to 10^18/2 = 5*10^17, but memory is the constraint

    # For FFT approach, we need an array of size ~2*(limit+1)
    # limit = 5*10^7 -> array size 10^8 -> 800MB for float64, too much
    # limit = 10^7 -> array size 2*10^7 -> 160MB, feasible
    # limit = 5*10^6 -> array size 10^7 -> 80MB, feasible

    # Let's try multiple limits and pick the best N found
    # Use the largest feasible limit first to maximize potential pairs
    limits_to_try = [120_000_000, 100_000_000, 50_000_000, 20_000_000]

    for limit in limits_to_try:
        if time.time() > deadline - 30:
            break

        # Check if we have enough memory (rough estimate)
        # FFT needs array of size next_power_of_2(2*(limit+1))
        fft_size = 1
        while fft_size < 2 * (limit + 1):
            fft_size *= 2
        # Memory: 2 arrays of fft_size complex128 = 2 * fft_size * 16 bytes
        mem_bytes = 2 * fft_size * 16
        if mem_bytes > 2_000_000_000:  # 2GB limit
            continue

        # Generate powerful numbers
        powerful = _generate_powerful(limit)
        n_powerful = len(powerful)

        if n_powerful < 10:
            continue

        # Create indicator array using int32 for memory efficiency
        arr = np.zeros(limit + 1, dtype=np.int32)
        for p in powerful:
            arr[p] = 1

        # FFT convolution
        F = np.fft.fft(arr, n=fft_size)
        conv = np.fft.ifft(F * F).real

        # Find N with max count, N >= 2
        # conv[k] = sum_i arr[i] * arr[k-i] for valid i
        # We want the largest conv[k] for k >= 2
        max_idx = int(np.argmax(conv[2:])) + 2
        N_best = max_idx
        conv_val = int(round(conv[N_best]))

        # Compute actual number of unordered pairs with a <= b
        if N_best % 2 == 0 and N_best // 2 <= limit and arr[N_best // 2] > 0:
            n_pairs = (conv_val + 1) // 2
        else:
            n_pairs = conv_val // 2

        if n_pairs > best_count:
            best_count = n_pairs
            # Generate the actual pairs
            powerful_list = sorted(powerful)
            powerful_set = powerful
            pairs = []
            for a in powerful_list:
                if 2 * a > N_best:
                    break
                b = N_best - a
                if b in powerful_set:
                    pairs.append([a, b])
            best_result = {'N': N_best, 'pairs': pairs}

        if time.time() > deadline - 20:
            break

    if best_result is None:
        # Fallback: very small search
        limit = 100_000
        powerful = _generate_powerful(limit)
        arr = np.zeros(limit + 1, dtype=np.int32)
        for p in powerful:
            arr[p] = 1
        fft_size = 1
        while fft_size < 2 * (limit + 1):
            fft_size *= 2
        F = np.fft.fft(arr, n=fft_size)
        conv = np.fft.ifft(F * F).real
        max_idx = int(np.argmax(conv[2:])) + 2
        N_best = max_idx
        conv_val = int(round(conv[N_best]))
        if N_best % 2 == 0 and N_best // 2 <= limit and arr[N_best // 2] > 0:
            n_pairs = (conv_val + 1) // 2
        else:
            n_pairs = conv_val // 2
        powerful_set = powerful
        pairs = []
        for a in sorted(powerful_set):
            if 2 * a > N_best:
                break
            b = N_best - a
            if b in powerful_set:
                pairs.append([a, b])
        best_result = {'N': N_best, 'pairs': pairs}

    return best_result
