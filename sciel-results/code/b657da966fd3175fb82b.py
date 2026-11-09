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
    L = 5 * 10**7  # Target limit for FFT-based sumset computation

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

    # If generation took too long or we have very few, try a smaller L
    if (n_p < 100 or elapsed() > 15.0) and elapsed() < deadline - 80:
        L = 1 * 10**7
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

    # Structural approach: Use FFT convolution to compute the sumset counts
    # Create a boolean array where arr[x] = 1 if x is powerful
    # The convolution arr * arr gives counts of pairs (a,b) with a+b = N
    # Note: this counts ordered pairs; for a<=b we need to adjust
    
    max_N = 2 * L
    # Use FFT size that is a power of 2 for efficiency
    fft_size = 1
    while fft_size < max_N + 1:
        fft_size *= 2
    # Cap FFT size to avoid memory issues
    if fft_size > 2**24:  # ~16 million, ~128MB for complex
        fft_size = 2**24
        max_N = fft_size - 1

    # Create the indicator array
    arr = np.zeros(fft_size, dtype=np.float64)
    for x in powerful:
        if x < fft_size:
            arr[x] = 1.0

    # Compute convolution via FFT
    fft_arr = np.fft.fft(arr)
    conv = np.fft.ifft(fft_arr * fft_arr).real

    # conv[N] = number of ordered pairs (a,b) with a+b=N, both powerful
    # For a<=b: if N is even, pairs with a=b contribute 1, others contribute 2
    # count_unordered[N] = (conv[N] + (1 if N is even and N/2 is powerful else 0)) // 2
    
    # Find the N with maximum unordered pair count
    # We need N >= 2 and N <= 2*L (since both a,b <= L)
    # Also N <= 10^18 (always satisfied here since 2*L <= 10^8)
    
    best_N = 2
    best_count = 0
    
    # Search over valid N range
    search_start = 2
    search_end = min(max_N, 2 * L)
    
    for N in range(search_start, search_end + 1):
        ordered_count = int(round(conv[N]))
        # Adjust for unordered: pairs with a=b count once, others count twice
        if N % 2 == 0:
            half = N // 2
            # Check if half is powerful
            if half < len(arr) and arr[half] > 0.5:
                unordered_count = (ordered_count + 1) // 2
            else:
                unordered_count = ordered_count // 2
        else:
            unordered_count = ordered_count // 2
        
        if unordered_count > best_count:
            best_count = unordered_count
            best_N = N

    # Verify and extract pairs for best_N
    pset = set(powerful)
    pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in pset:
            pairs.append([a, b])

    # If FFT approach didn't find good results, fall back to O(n^2) for small n_p
    if best_count < 2 and n_p <= 8000 and elapsed() < deadline - 30:
        count = defaultdict(int)
        for i in range(n_p):
            a = powerful[i]
            for j in range(i, n_p):
                b = powerful[j]
                N = a + b
                if N > 10**18:
                    break
                count[N] += 1
            if elapsed() > deadline - 15.0:
                break

        if count:
            best_N = max(count, key=count.get)
            best_count = count[best_N]
            pset = set(powerful)
            pairs = []
            for a in powerful:
                b = best_N - a
                if b >= a and b in pset:
                    pairs.append([a, b])

    # If still not enough, do a local search around best_N
    if best_count > 0 and elapsed() < deadline - 20.0:
        # Try nearby N values to see if we can do better
        fine_range = max(100, best_count * 30)
        for offset in range(1, fine_range + 1):
            if elapsed() > deadline - 5.0:
                break
            for N in [best_N + offset, best_N - offset]:
                if N < 2 or N > 10**18:
                    continue
                # Count pairs for this N using two-pointer
                i = 0
                j = n_p - 1
                cnt = 0
                while i <= j:
                    s = powerful[i] + powerful[j]
                    if s == N:
                        cnt += 1
                        i += 1
                        j -= 1
                    elif s < N:
                        i += 1
                    else:
                        j -= 1
                if cnt > best_count:
                    best_count = cnt
                    best_N = N
                    pairs = []
                    for a in powerful:
                        b = best_N - a
                        if b >= a and b in pset:
                            pairs.append([a, b])

    if not pairs:
        # Fallback: find at least one valid pair
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
