# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_N = 1000000
    MAX_VAL = MAX_N + 25  # we need up to n+24 where n can be 1000000

    # Precompute smooth_part[x] = largest divisor of x with all prime factors < 24
    smooth_part = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in primes:
        # For each multiple of p, multiply smooth_part by the full power of p in that multiple
        for m in range(p, MAX_VAL + 1, p):
            power = 1
            tmp = m
            while tmp % p == 0:
                power *= p
                tmp //= p
            smooth_part[m] *= power

    if time.time() - start_time > DEADLINE:
        return 1000

    # We need smooth parts for positions 1001 to 1000000+24 = 1000024
    # Window for n: positions n+1 to n+24
    # n ranges from 1000 to 1000000, so positions range from 1001 to 1000024

    # Extract the relevant slice
    lo = 1001
    hi = 1000000 + 24  # inclusive
    sp = smooth_part[lo:hi + 1]  # length = hi - lo + 1 = 1000000 - 1001 + 24 + 1 = 999024

    # Map distinct values to IDs
    unique_vals, inverse = np.unique(sp, return_inverse=True)
    K = len(unique_vals)

    # For each ID, build a prefix sum of occurrences
    # prefix[k][i] = number of positions j in [0, i-1] where sp[j] == k
    # To save memory, we'll process in a smarter way.
    # Actually, K could be large. Let's check.
    # Number of distinct 23-smooth numbers up to 1000024 is not too large.
    # Let's estimate: 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h * 23^i <= 1000024
    # This is at most a few thousand.

    # For each distinct ID, create a binary array of length len(sp)
    # Then for window starting at position n (0-indexed in sp), the window covers sp[n:n+24]
    # n ranges from 0 to len(sp) - 24
    # For ID k, it's present in window n iff sum of binary_k[n:n+24] > 0

    # Build prefix sums for each ID
    n_windows = len(sp) - 24 + 1  # number of valid windows
    # distinct_count[n] = number of IDs present in window n

    # Initialize distinct count array
    distinct_count = np.zeros(n_windows, dtype=np.int32)

    # For each ID, compute where it appears and add to distinct_count for overlapping windows
    for k in range(K):
        # Find positions where sp == k
        positions = np.where(inverse == k)[0]
        if len(positions) == 0:
            continue

        # For each position p in positions, it contributes to windows [p-23, p] (clamped to [0, n_windows-1])
        # i.e., window n contains position p iff n <= p <= n+23, i.e., n in [p-23, p]
        # So for each position p, add 1 to distinct_count[max(0, p-23):min(n_windows, p+1)]
        # But we need to be careful: we want to add 1 for each window that contains at least one position with ID k.
        # The above approach adds 1 for each position, which would overcount.
        # Instead, we need: for window n, ID k is present iff any position in [n, n+23] has ID k.
        # So: present[n] = OR of (positions in [n, n+23])
        # We can compute this by: for each position p, mark windows [p-23, p] as "potentially containing k"
        # Then take the union. But union of intervals... 

        # Better: use a difference array approach.
        # For each position p, the windows that contain p are n in [max(0, p-23), min(n_windows-1, p)]
        # We want to know, for each window n, whether ANY position with ID k is in [n, n+23].
        # This is equivalent to: the minimum position with ID k that is >= n, and check if it's <= n+23.
        # Or: for window n, ID k is present iff there exists p in positions such that n <= p <= n+23.
        # i.e., p in [n, n+23].

        # Let's use a sliding approach: for each window n, check if positions intersects [n, n+23].
        # Since positions is sorted, we can use binary search.
        # For window n: find the first position >= n. If it exists and is <= n+23, then present.

        # This is O(n_windows * log(len(positions))) per ID, which might be too slow.
        # Alternative: mark all windows covered by any position.
        # For each position p, it covers windows [p-23, p]. The union of these intervals.
        # Since positions are sorted, we can merge intervals.

        # Actually, let's just use a boolean array for this ID and use numpy.
        # Create a boolean array of length n_windows, initially False.
        # For each position p, set windows[max(0, p-23):min(n_windows, p+1)] to True.
        # This is O(len(positions) * 24) which is fine.

        # But doing this for each ID in a Python loop might be slow if K is large.
        # Let's check K first.
        pass

    # Let's check how many distinct values there are
    if K < 5000:
        # For each ID, compute which windows contain it
        for k in range(K):
            positions = np.where(inverse == k)[0]
            if len(positions) == 0:
                continue
            # For each position, mark the windows it covers
            # Windows covered by position p: [max(0, p-23), min(n_windows-1, p)]
            # We can vectorize this
            starts = np.maximum(0, positions - 23)
            ends = np.minimum(n_windows - 1, positions) + 1  # exclusive end for slicing
            # For each (start, end) pair, we want to mark distinct_count[start:end] += 1
            # But this is still a loop. Let's use a difference array.
            diff = np.zeros(n_windows + 1, dtype=np.int32)
            for s, e in zip(starts, ends):
                diff[s] += 1
                diff[e] -= 1
            # Now prefix sum diff to get the count of positions with ID k in each window
            # But we only need whether count > 0, i.e., boolean
            # Actually, the prefix sum gives the number of positions with ID k in window n.
            # We just need > 0.
            prefix = np.cumsum(diff[:-1])
            present = (prefix > 0).astype(np.int32)
            distinct_count += present

            if time.time() - start_time > DEADLINE:
                break
    else:
        # Fallback: linear scan with rolling set
        best_n = 1000
        best_count = float('inf')
        window = list(sp[:24])
        freq = {}
        for x in window:
            freq[x] = freq.get(x, 0) + 1
        best_count = len(freq)
        for n in range(1, n_windows):
            out = sp[n - 1]
            inp = sp[n + 23]
            freq[out] -= 1
            if freq[out] == 0:
                del freq[out]
            freq[inp] = freq.get(inp, 0) + 1
            count = len(freq)
            if count < best_count:
                best_count = count
                best_n = 1000 + n
                if best_count == 1:
                    return best_n
            if time.time() - start_time > DEADLINE:
                break
        return best_n

    # Find the window with minimum distinct count
    min_count = np.min(distinct_count)
    # Find the first n with this count
    candidates = np.where(distinct_count == min_count)[0]
    best_n = 1000 + candidates[0]

    return best_n
# EVOLVE-BLOCK-END
