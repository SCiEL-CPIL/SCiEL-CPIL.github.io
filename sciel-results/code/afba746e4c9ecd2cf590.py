"""Rewritten program: longest increasing prime subsequence <=10000 with nondecreasing gaps."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve(n):
    """Return a boolean array where index i is True if i is prime."""
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    t0 = time.time()
    deadline = t0 + 75.0  # safety margin before 90s kill

    N = 10000
    is_prime = _sieve(N)
    primes = np.array([i for i in range(2, N + 1) if is_prime[i]], dtype=np.int32)
    n = len(primes)

    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]

    # max_gap: maximum possible gap between any two primes <= N
    max_gap = int(primes[-1] - primes[0])

    # DP state:
    #   best[j, g] = maximum length of a valid subsequence ending at primes[j]
    #                 whose last gap is exactly g
    #   pref[j, g] = max_{h <= g} best[j, h]  (prefix maximum over gap dimension)
    #   arg[j, g]  = the h <= g that achieves pref[j, g] (used for reconstruction)
    #
    # Transition: for each j, for each i < j:
    #   gap = primes[j] - primes[i]
    #   candidate_len = pref[i, gap] + 1   (if pref[i, gap] > 0)
    #   if candidate_len > best[j, gap], update
    #
    # This is O(n^2) in the number of primes (~1229), so ~1.5M inner iterations,
    # which is fast in Python with numpy vectorization where possible.

    # We'll use a sparse representation: for each j, we only need to track the
    # best length for each gap value that actually occurs. But since gaps can
    # be up to ~10000, and n is only ~1229, the dense array is manageable.

    best = np.zeros((n, max_gap + 1), dtype=np.int32)
    pref = np.zeros((n, max_gap + 1), dtype=np.int32)
    arg = np.zeros((n, max_gap + 1), dtype=np.int32)

    # parent pointers for reconstruction
    parent_idx = np.full((n, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.full((n, max_gap + 1), -1, dtype=np.int32)

    # Initialize: every prime alone has length 1, gap = 0 (sentinel)
    # We'll handle length-1 sequences implicitly: if pref[i, gap] == 0,
    # then starting a new sequence at i gives length 1, so candidate = 2.

    for j in range(n):
        if time.time() > deadline:
            break

        # For each i < j, compute gap and update best[j, gap]
        # Vectorize the gap computation
        gaps = primes[j] - primes[:j]  # shape (j,)
        # Only consider positive gaps (which they all are since primes are increasing)
        # For each i, gap = gaps[i]
        # candidate = pref[i, gap] + 1 if pref[i, gap] > 0 else 2

        # We need to iterate over i because pref[i, gap] depends on i
        # But we can vectorize the lookup: for each i, look up pref[i, gaps[i]]
        # This is a fancy-index operation
        if j > 0:
            # idx: for each i in range(j), we want pref[i, gaps[i]]
            idx_i = np.arange(j, dtype=np.int32)
            idx_g = gaps.astype(np.int32)
            # Clip gaps to max_gap (shouldn't be needed but safe)
            idx_g = np.minimum(idx_g, max_gap)
            prev_best = pref[idx_i, idx_g]  # shape (j,)
            # candidate = prev_best + 1 if prev_best > 0, else 2
            candidates = np.where(prev_best > 0, prev_best + 1, 2)

            # Now for each gap value, we want the max candidate over all i that produce that gap
            # and the corresponding i (for parent pointer)
            # Use a loop over unique gaps for efficiency
            unique_gaps, inverse = np.unique(gaps, return_inverse=True)
            # For each unique gap, find the best candidate and the argmax i
            # We can do this with a reduce or a loop
            for ug_idx, ug in enumerate(unique_gaps):
                # Find all i where gaps[i] == ug
                mask = (gaps == ug)
                if not np.any(mask):
                    continue
                candidate_vals = candidates[mask]
                # Find the max candidate and the first i that achieves it
                max_cand = int(np.max(candidate_vals))
                # Find the index in the masked array that achieves max
                # We want the original i index
                masked_indices = np.where(mask)[0]
                best_local = np.argmax(candidate_vals)
                best_i = int(masked_indices[best_local])

                if max_cand > best[j, ug]:
                    best[j, ug] = max_cand
                    parent_idx[j, ug] = best_i
                    # The gap of the parent at best_i: it's the arg of pref[best_i, ug]
                    parent_gap[j, ug] = int(arg[best_i, ug])

        # Compute prefix max and arg for row j
        if j < n - 1:
            pref[j, :] = np.maximum.accumulate(best[j, :])
            # Compute arg: for each g, the h <= g with best[j, h] == pref[j, g]
            # Use a running max
            running_max = 0
            running_arg = 0
            # Vectorized approach: iterate
            for g in range(1, max_gap + 1):
                if best[j, g] > running_max:
                    running_max = best[j, g]
                    running_arg = g
                arg[j, g] = running_arg

    # Find the best (j, gap)
    # The maximum length is max over all j, g of best[j, g]
    # Also consider length-1 sequences (any single prime)
    max_len = 0
    best_j = -1
    best_gap = -1
    for j in range(n):
        row_max = int(np.max(best[j, :]))
        if row_max > max_len:
            max_len = row_max
            best_j = j
            best_gap = int(np.argmax(best[j, :]))

    # If no valid sequence of length >= 2 found, return the first prime
    if max_len < 2 or best_j < 0:
        return [int(primes[0])]

    # Reconstruct
    result = []
    j = best_j
    gap = best_gap
    while j >= 0:
        result.append(int(primes[j]))
        pi = int(parent_idx[j, gap])
        pg = int(parent_gap[j, gap])
        if pi < 0:
            break
        j = pi
        gap = pg
        if gap == 0:
            # This is the first element (no prior gap)
            result.append(int(primes[j]))
            break

    result.reverse()

    # Validate
    if len(result) >= 2:
        gaps_list = [result[i+1] - result[i] for i in range(len(result)-1)]
        valid = all(gaps_list[i+1] >= gaps_list[i] for i in range(len(gaps_list)-1))
        if not valid:
            # Fallback: greedy
            values = []
            for p in range(2, N + 1):
                if is_prime[p]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
            return values

    return result

# EVOLVE-BLOCK-END
