"""Longest increasing prime subsequence <=10000 with nondecreasing gaps.
Vectorized DP: state = last two primes; transition via suffix-max with
precomputed argmax, inner j-loop vectorized with NumPy."""
import time
import numpy as np

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before the 90s kill

    primes = _sieve(10000)
    P = len(primes)
    if P < 2:
        return list(primes)
    if P == 2:
        return list(primes)

    p0 = primes[0]
    primes_arr = np.array(primes, dtype=np.int64)

    # dp[i][j] = length of longest valid sequence ending with primes[i], primes[j]
    # par[i][j] = index k such that sequence is ... primes[k], primes[i], primes[j]
    dp = np.zeros((P, P), dtype=np.int32)
    par = np.full((P, P), -1, dtype=np.int32)

    # Base: any pair (0, j) for j >= 1 has length 2, predecessor 0
    dp[0, 1:] = 2
    par[0, 1:] = 0

    best_len = 2
    best_i = 0
    best_j = 1

    for i in range(1, P):
        if time.time() > deadline - 2.0:
            break

        # Compute suffix max over k in [0, i-1] of dp[k][i], and the argmax index
        # suff[m] = max(dp[k][i] for k in [m, i-1])
        # argmax_k[m] = the k in [m, i-1] that achieves suff[m] (smallest k on ties)
        col = dp[:i, i]  # shape (i,)
        # suffix max
        suff = np.maximum.accumulate(col[::-1])[::-1]  # suff[m] = max(col[m:])
        # argmax: for each m, find the smallest k >= m such that col[k] == suff[m]
        # Vectorized approach: create a 2D comparison? Too expensive for large i.
        # Instead, use a simple loop for argmax since i is at most ~1200
        argmax_k = np.empty(i, dtype=np.int32)
        # For efficiency, compute argmax in a backward pass
        best_val = col[i-1]
        argmax_k[i-1] = i-1
        for m in range(i-2, -1, -1):
            if col[m] > best_val:
                best_val = col[m]
                argmax_k[m] = m
            else:
                # keep the later index (which is smaller k? No, we want smallest k)
                # Actually we want smallest k achieving the max.
                # Since we're going backward, if col[m] == best_val, m is smaller, so use m.
                if col[m] == best_val:
                    argmax_k[m] = m
                else:
                    argmax_k[m] = argmax_k[m+1]

        pi = primes_arr[i]
        # For all j > i, compute T = 2*pi - primes[j]
        # kmin = smallest k < i such that primes[k] >= T, or 0 if T <= p0
        # If kmin < i, then dp[i][j] = 1 + suff[kmin], par[i][j] = argmax_k[kmin]
        # Else dp[i][j] = 2 (base), par[i][j] = 0

        js = np.arange(i+1, P)
        T = 2 * pi - primes_arr[js]  # shape (P-i-1,)
        # kmin: smallest k in [0, i) with primes[k] >= T
        # Use searchsorted on primes_arr[0:i]
        # searchsorted returns index in [0, i]; if result == i, no such k exists
        kmin = np.searchsorted(primes_arr[:i], T, side='left')  # shape (P-i-1,)
        # kmin is in [0, i]. If kmin < i, we can extend; else base case.
        valid_mask = kmin < i
        # For valid entries, compute candidate length
        cand = np.full(len(js), 2, dtype=np.int32)
        par_val = np.full(len(js), 0, dtype=np.int32)
        if np.any(valid_mask):
            kv = kmin[valid_mask]
            sv = suff[kv]
            av = argmax_k[kv]
            cand[valid_mask] = 1 + sv
            par_val[valid_mask] = av
            # Ensure at least 2 (base case)
            # cand is already >= 2 since sv >= 1 (dp[k][i] >= 1 for valid states)
            # Actually dp[k][i] could be 0 if no sequence ends at (k,i) with k>0?
            # No: dp[0][i] = 2 for all i >= 1, and dp[k][i] >= 2 for all valid k.
            # So sv >= 2, cand >= 3. But if suff[kmin] is 0 (shouldn't happen), cand=1.
            # Let's be safe:
            cand[valid_mask] = np.maximum(cand[valid_mask], 2)

        dp[i, js] = cand
        par[i, js] = par_val

        # Update best
        row_max = np.max(cand)
        if row_max > best_len:
            best_len = int(row_max)
            # Find the j that achieves this
            idx = np.argmax(cand)
            best_j = int(js[idx])
            best_i = i

    # Reconstruct from best pair
    seq = [primes[best_j], primes[best_i]]
    i, j = best_i, best_j
    while i != 0 and len(seq) < best_len:
        k = int(par[i, j])
        if k < 0:
            break
        seq.append(primes[k])
        j, i = i, k
    seq.reverse()

    # Validity fallback
    if time.time() < deadline - 0.5:
        def valid(seq):
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                if (b - a) > (c - b):
                    return False
            return True
        if not valid(seq):
            seq = [primes[best_j], primes[best_i]]
            i, j = best_i, best_j
            while i != 0:
                gap = primes[j] - primes[i]
                found = False
                for k in range(i - 1, -1, -1):
                    if (primes[i] - primes[k]) <= gap:
                        seq.append(primes[k])
                        j, i = i, k
                        found = True
                        break
                if not found:
                    break
            seq.reverse()

    return seq
