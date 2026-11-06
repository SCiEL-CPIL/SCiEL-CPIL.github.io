import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # Strategy: generate powerful numbers up to a limit L, then use FFT convolution
    # to count sums a+b=N where both a and b are powerful. Pick the N with maximum count.

    def get_primes(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    # Generate powerful numbers up to L
    # A powerful number is of the form x^2 * y^3 where y is square-free.
    # For val = x^2 * y^3 <= L, we need x <= sqrt(L / y^3)
    # y^3 <= L means y <= L^(1/3)
    L = 10**7  # We'll use 10 million; FFT on 2*10^7 is feasible

    max_y = int(round(L ** (1/3))) + 2  # ~216

    # Sieve for square-free numbers up to max_y
    small_primes = get_primes(int(math.isqrt(max_y)) + 1)

    sq_free = []
    for y in range(1, max_y + 1):
        is_sq_free = True
        temp_y = y
        for p in small_primes:
            if p * p > temp_y:
                break
            if temp_y % (p * p) == 0:
                is_sq_free = False
                break
        if is_sq_free:
            sq_free.append(y)

    powerful_set = set()
    for y in sq_free:
        y_cubed = y ** 3
        if y_cubed > L:
            continue
        max_x = int(math.isqrt(L // y_cubed))
        for x in range(1, max_x + 1):
            val = x * x * y_cubed
            if val <= L:
                powerful_set.add(val)

    P = sorted(powerful_set)
    K = len(P)

    if time.time() > deadline - 30:
        # Fallback: return a simple result
        return {'N': 2, 'pairs': [[1, 1]]}

    # Build indicator array
    arr = np.zeros(L + 1, dtype=np.float64)
    for val in P:
        arr[val] = 1.0

    # FFT convolution: result[n] = number of pairs (a,b) with a+b=n
    # We need sums up to 2L, so array size must be at least 2L+1
    fft_size = 1
    while fft_size < 2 * L + 1:
        fft_size <<= 1

    # Perform FFT
    F = np.fft.rfft(arr, n=fft_size)
    F2 = F * F
    conv = np.fft.irfft(F2, n=fft_size)

    # conv[n] is the count of ordered pairs (a,b) with a+b=n
    # We want unordered pairs a<=b, so the count is roughly conv[n]/2 + (1 if n/2 is powerful)
    # Actually, conv[n] counts ordered pairs. For a < b, there are 2 ordered pairs.
    # For a = b, there's 1 ordered pair.
    # So unordered count = (conv[n] + (1 if n is even and n//2 is powerful)) / 2
    # But for finding the max, we can just look at conv[n] directly since the correction is small.
    # Actually, let's be more careful. Let's compute the actual count of unordered pairs.
    # conv[n] = sum over a: [a is powerful] * [n-a is powerful]
    # This counts ordered pairs (a, n-a) where both are powerful.
    # If a != n-a, this counts both (a, n-a) and (n-a, a), so 2 per unordered pair.
    # If a == n-a (i.e., n is even and a = n/2), this counts 1 per unordered pair.
    # So unordered_count = (conv[n] + (1 if n%2==0 and (n//2) is powerful)) // 2

    # For efficiency, let's just use conv[n] as a proxy for now, since the max will be at a large N
    # where most pairs have a != b.

    conv_int = np.round(conv[:2*L + 1]).astype(np.int64)

    # Find N with maximum count
    # We want N >= 2 (since a, b >= 1)
    # Also, we should probably look at a reasonable range of N.
    # The maximum count is likely for N around L to 2L.
    max_idx = np.argmax(conv_int)
    best_N = int(max_idx)

    # Verify: make sure best_N is at least 2
    if best_N < 2:
        best_N = 2

    # Now retrieve the actual pairs
    powerful_set_lookup = powerful_set
    pairs = []
    # Iterate over powerful numbers a, check if b = best_N - a is powerful
    # We need a <= b, so a <= best_N / 2
    half_N = best_N // 2
    for a in P:
        if a > half_N:
            break
        b = best_N - a
        if b < a:
            continue
        if b in powerful_set_lookup:
            pairs.append([a, b])

    # Sort pairs by a for consistency
    pairs.sort(key=lambda x: x[0])

    if len(pairs) == 0:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': pairs}
