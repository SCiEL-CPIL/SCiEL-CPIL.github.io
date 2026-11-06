import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_PRODUCT = 223092870  # 2*3*5*7*11*13*17*19*23

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    lower = 1000
    upper = 1000000
    max_val = upper + 24  # 1000024
    size = max_val - lower + 1

    # Compute the 23-smooth part (product of all prime factors < 24) for each number
    # using vectorized NumPy operations
    nums = np.arange(lower, max_val + 1, dtype=np.int64)
    smooth = np.ones(size, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # For each multiple of p, divide out all powers of p and multiply into smooth
        mask = (nums % p) == 0
        if not mask.any():
            continue
        # Get indices of multiples
        idxs = np.where(mask)[0]
        vals = nums[idxs].copy()
        # Divide out all powers of p
        while np.any(vals % p == 0):
            still_div = (vals % p) == 0
            vals[still_div] //= p
            smooth[idxs[still_div]] *= p

    # Now smooth[i] is the 23-smooth part of (lower + i)
    # We need to find n in [lower, upper] minimizing distinct values in smooth[n+1-lower .. n+24-lower]
    # i.e., smooth indices (n+1-lower) to (n+24-lower), which is indices (n-lower+1) to (n-lower+24)

    # Encode smooth parts as compact integer IDs for fast counting
    unique_smooth, inverse = np.unique(smooth, return_inverse=True)
    num_unique = len(unique_smooth)

    # Sliding window distinct count using a frequency array
    # Window for n: smooth indices from (n-lower+1) to (n-lower+24)
    # For n=lower: indices 1 to 24
    # For n=lower+k: indices (k+1) to (k+24)

    # Use a frequency array indexed by inverse
    freq = np.zeros(num_unique, dtype=np.int32)
    distinct = 0

    # Initial window: indices 1..24
    for i in range(1, 25):
        uid = inverse[i]
        if freq[uid] == 0:
            distinct += 1
        freq[uid] += 1

    best_count = distinct
    best_n = lower

    if best_count == 1:
        return lower

    # Slide window: for n from lower+1 to upper
    # Remove index (n-lower), add index (n+24-lower)
    # For n = lower+k: remove index k, add index k+24
    for k in range(1, upper - lower + 1):
        if time.time() > deadline:
            break

        out_idx = k
        in_idx = k + 24

        out_uid = inverse[out_idx]
        freq[out_uid] -= 1
        if freq[out_uid] == 0:
            distinct -= 1

        in_uid = inverse[in_idx]
        if freq[in_uid] == 0:
            distinct += 1
        freq[in_uid] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = lower + k
            if best_count == 1:
                break

    return best_n
