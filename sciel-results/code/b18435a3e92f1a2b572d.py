"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Generate powerful numbers up to L
    # Every powerful number n = prod p_i^{e_i}, e_i >= 2
    # Can be written as a^2 * b^3 (not uniquely, but generates all)
    L = 8 * 10**7  # Target limit, tuned for time budget

    powerful_set = set()
    powerful_set.add(1)

    max_b = int(round(L ** (1.0/3.0))) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)

    powerful = sorted(powerful_set)
    n_p = len(powerful)

    # If generation took too long, we still proceed with what we have
    if n_p < 100 and elapsed() < deadline - 80:
        L = 2 * 10**7
        powerful_set = set([1])
        max_b = int(round(L ** (1.0/3.0))) + 2
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful_set.add(n)
        powerful = sorted(powerful_set)
        n_p = len(powerful)

    # Use FFT convolution to compute all pair sums at once
    # Create indicator array of size M+1 where M = 2*L (max sum)
    M = 2 * L
    # We need array of size at least 2*L for convolution
    # Use next power of 2 for efficient FFT
    size = 1
    while size < 2 * M + 1:
        size *= 2
    # Cap size to avoid excessive memory
    max_size = 262144  # 2^18, ~2MB for complex array
    if size > max_size:
        size = max_size
        M = (size - 1) // 2
        # Filter powerful numbers to fit
        powerful = [x for x in powerful if x <= M]
        n_p = len(powerful)
        if n_p == 0:
            return {'N': 2, 'pairs': [[1, 1]]}

    # Build indicator array
    indicator = np.zeros(size, dtype=np.float64)
    for x in powerful:
        if x < size:
            indicator[x] = 1.0

    # FFT convolution: indicator * indicator gives counts of pairs (a,b) with a+b=k
    # But this counts ordered pairs. We want unordered with a <= b.
    # For a < b: counted twice in convolution (once as (a,b) and once as (b,a))
    # For a = b: counted once
    # So: unordered_count(k) = (conv[k] + (1 if k is even and k/2 in powerful else 0)) / 2
    # Actually: conv[k] = sum_{a+b=k} 1 = number of ordered pairs
    # Unordered with a<=b: for k even, if k/2 is powerful, that's a diagonal element
    # count_unordered = (conv[k] + diag) / 2 where diag = 1 if k even and k/2 in powerful_set else 0

    # Perform FFT
    fft = np.fft.fft(indicator)
    fft_sq = fft * fft
    conv = np.fft.ifft(fft_sq).real

    # conv[k] ≈ number of ordered pairs (a,b) with a+b=k
    # For our purposes: number of unordered pairs (a<=b) with a+b=k is:
    # (conv[k] + diag[k]) / 2 where diag[k] = 1 if k is even and k/2 in powerful_set

    # Find the best N
    best_N = 0
    best_count = 0

    # Check a range of N values
    # conv indices 0 to size-1 correspond to sums 0 to size-1
    # But sums > 2*L are not meaningful
    search_max = min(size - 1, 2 * M)

    # Build powerful set for diagonal check
    pset = set(powerful)

    # Scan for best N
    # We can vectorize the diagonal check
    for N in range(2, search_max + 1):
        if elapsed() > deadline - 30.0:
            break
        cnt_ordered = conv[N]
        if cnt_ordered < 1:
            continue
        # Check diagonal
        if N % 2 == 0 and (N // 2) in pset:
            cnt_unordered = (cnt_ordered + 1) / 2.0
        else:
            cnt_unordered = cnt_ordered / 2.0
        # Round to integer
        cnt_unordered = int(round(cnt_unordered))
        if cnt_unordered > best_count:
            best_count = cnt_unordered
            best_N = N

    # Verify and collect pairs for best_N
    if best_N > 0:
        pairs = []
        for a in powerful:
            b = best_N - a
            if b >= a and b in pset:
                pairs.append([a, b])
        # The actual count might differ slightly due to FFT rounding
        best_count = len(pairs)
    else:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}

    # Fine-tune: check neighbors of best_N more carefully
    # FFT has small numerical errors, so verify nearby values
    for offset in range(-50, 51):
        if elapsed() > deadline - 10.0:
            break
        N = best_N + offset
        if N < 2 or N > search_max:
            continue
        cnt_ordered = conv[N]
        if cnt_ordered < 1:
            continue
        if N % 2 == 0 and (N // 2) in pset:
            cnt_unordered = (cnt_ordered + 1) / 2.0
        else:
            cnt_unordered = cnt_ordered / 2.0
        cnt_unordered = int(round(cnt_unordered))
        if cnt_unordered > best_count:
            # Verify by explicit counting
            pairs_check = []
            for a in powerful:
                b = N - a
                if b >= a and b in pset:
                    pairs_check.append([a, b])
            if len(pairs_check) > best_count:
                best_count = len(pairs_check)
                best_N = N
                pairs = pairs_check

    # Final verification of best_N
    final_pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in pset:
            final_pairs.append([a, b])

    if not final_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': final_pairs}
# EVOLVE-BLOCK-END
