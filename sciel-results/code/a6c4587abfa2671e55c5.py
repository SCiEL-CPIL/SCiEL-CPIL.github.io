import time
import numpy as np

def _is_prime_sieve(n):
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
    deadline = time.time() + 85  # Safety margin for 90s limit

    N_LIMIT = 10000
    is_prime = _is_prime_sieve(N_LIMIT)
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    n_primes = len(primes)

    if n_primes == 0:
        return []

    max_gap = N_LIMIT
    # dp[i][g] = max length ending at prime i with last gap exactly g
    dp = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # PMax[i][g] = max_{h<=g} dp[i][h]
    PMax = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # argmax_PMax[i][g] = argmax_{h<=g} dp[i][h] (the gap h that achieves the max)
    argmax_PMax = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # parent tracking: for state (j, gap), which (i, parent_gap) produced it
    parent_i = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)

    p_arr = np.array(primes, dtype=np.int32)

    for j in range(n_primes):
        if time.time() > deadline:
            break

        if j > 0:
            i_range = np.arange(j, dtype=np.int32)
            gaps = p_arr[j] - p_arr[i_range]

            # Get previous max lengths: PMax[i][gap] = max_{h<=gap} dp[i][h]
            prev_lens = PMax[i_range, gaps]

            # Candidates: if prev_lens > 0, extend; otherwise base case length 2
            cands = np.where(prev_lens > 0, prev_lens + 1, 2)

            # Current dp values for these gaps
            current_dp = dp[j, gaps]

            # Find where candidate is better
            mask = cands > current_dp

            if np.any(mask):
                # Update dp
                dp[j, gaps[mask]] = cands[mask]

                # Update parents
                parent_i[j, gaps[mask]] = i_range[mask]
                # The parent gap is the gap that achieved the prefix max at (i, gap)
                parent_gap[j, gaps[mask]] = argmax_PMax[i_range[mask], gaps[mask]]

        # Compute PMax[j] and argmax_PMax[j]
        PMax[j, :] = np.maximum.accumulate(dp[j, :])

        # Compute argmax_PMax[j]: for each g, which h <= g achieves max dp[j][h]
        dp_j = dp[j]
        argmax_j = argmax_PMax[j]
        running_max = 0
        running_arg = 0

        for g in range(1, max_gap + 1):
            v = dp_j[g]
            if v > running_max:
                running_max = v
                running_arg = g
            argmax_j[g] = running_arg

    # Find global max
    max_len = 0
    best_j = -1
    best_gap = -1

    for j in range(n_primes):
        row_max = np.max(dp[j, :])
        if row_max > max_len:
            max_len = row_max
            best_j = j
            best_gap = int(np.argmax(dp[j, :]))

    if max_len < 2:
        return [primes[0]]

    # Reconstruct
    result = []
    cur_j = best_j
    cur_gap = best_gap

    while cur_j >= 0:
        result.append(primes[cur_j])
        pi = parent_i[cur_j, cur_gap]
        pg = parent_gap[cur_j, cur_gap]

        if pi < 0:
            break

        cur_j = pi
        cur_gap = pg

        # If parent_gap is 0, it means this was a base case (length-2 sequence)
        # The current state represents the second element of the pair
        # We need to add the first element (which is primes[cur_j])
        if cur_gap == 0:
            result.append(primes[cur_j])
            break

    result.reverse()

    # Verify
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if any(gaps[i+1] < gaps[i] for i in range(len(gaps)-1)):
            # Fallback: greedy
            values = []
            for p in primes:
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
            return values

    # Try to extend the sequence greedily from the right
    if len(result) >= 2:
        last_gap = result[-1] - result[-2]
        while True:
            target = result[-1]
            idx = np.searchsorted(p_arr, target, side='right')
            if idx < n_primes:
                gap = p_arr[idx] - target
                if gap >= last_gap and p_arr[idx] <= N_LIMIT:
                    result.append(int(p_arr[idx]))
                    last_gap = gap
                    continue
            break

    # Final verification
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if any(gaps[i+1] < gaps[i] for i in range(len(gaps)-1)):
            # Extension broke it; revert to DP result
            return result[:-1] if len(result) > 2 else result

    return result
