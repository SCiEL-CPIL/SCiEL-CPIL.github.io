import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 10_000_000
    
    # Compute sigma(n) for n=1..N
    # sigma(n) for n <= 10^7 fits in int32 (max ~ 45M for highly composite numbers)
    sigma = np.zeros(N + 1, dtype=np.int32)
    
    # Sieve: for i=1..N, add i to multiples of i
    # Optimization: only need i up to N/2 for non-trivial contributions
    # For i > N/2, only multiple is i itself, so sigma[i] += i (singleton)
    # We can do i=1..N//2 in the sieve, then handle the rest
    
    for i in range(1, N // 2 + 1):
        sigma[i::i] += i
    
    # For i > N//2, sigma[i] = i + (divisors from smaller i already added)
    # Actually the sieve above already handles all multiples correctly.
    # For i in (N//2, N], the only multiple of i that is <= N is i itself.
    # So sigma[i] should have i added to it. But our loop only went to N//2.
    # We need to add i to sigma[i] for i in (N//2, N].
    # These are just sigma[i] += i for each i.
    # But wait - the sieve loop `for i in range(1, N//2+1): sigma[i::i] += i`
    # For a given i, it adds i to sigma[i], sigma[2i], sigma[3i], ...
    # For j > N//2, j is a multiple of i only if i divides j and j/i >= 1.
    # If j > N//2 and i <= N//2, then j/i < 2, so j/i = 1, meaning i = j.
    # But i <= N//2 < j, so i != j. Thus no i <= N//2 divides any j > N//2 except i=j which is impossible.
    # Wait, that's wrong. If j = 7000000 and i = 3500000, then j = 2*i, so i divides j.
    # So j > N//2 CAN be a multiple of i <= N//2.
    
    # Let me reconsider. The sieve `sigma[i::i] += i` for i=1..N//2:
    # For each i, it adds i to positions i, 2i, 3i, ... up to N.
    # So for any j, sigma[j] = sum of all i such that i divides j AND i <= N//2.
    # But sigma(j) should be sum of ALL divisors of j, including those > N//2.
    # The divisors of j that are > N//2: if d > N//2 divides j, then j/d < 2, so j/d = 1, d = j.
    # So the only divisor of j that is > N//2 is j itself (when j > N//2).
    # Therefore: for j > N//2, sigma(j) = (sum of divisors of j that are <= N//2) + j
    # And our sieve gives us exactly (sum of divisors of j that are <= N//2) for all j.
    # So we need to add j to sigma[j] for all j > N//2.
    
    # For j <= N//2, all divisors of j are <= j <= N//2, so the sieve is complete.
    # For j > N//2, add j.
    
    half = N // 2
    if half < N:
        sigma[half+1:N+1] += np.arange(half+1, N+1, dtype=np.int32)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # Now group indices by sigma value using a dictionary
    # Only consider n >= 2 (sigma(1)=1, and we want distinct a,b with a/b close to 3/2)
    # Actually a=1, b=? sigma(1)=1, only n=1 has sigma=1, so no pair.
    
    # Build groups: dict from sigma_value -> list of indices
    # This is O(N) but with Python dict overhead. For 10M elements, this might be slow.
    # Alternative: use numpy to find groups after sorting.
    
    # Let's use the sort approach but more efficiently.
    # We only need to consider indices where sigma value appears more than once.
    
    # Sort indices 1..N by their sigma value
    idx = np.arange(1, N + 1, dtype=np.int32)
    sig = sigma[1:N+1]
    
    order = np.argsort(sig, kind='quicksort')
    sorted_sig = sig[order]
    sorted_idx = idx[order]
    
    # Find group boundaries
    changes = np.diff(sorted_sig) != 0
    boundaries = np.concatenate(([0], np.where(changes)[0] + 1, [N]))
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    best_frac = None  # Fraction for exact reporting
    
    num_groups = len(boundaries) - 1
    
    for gi in range(num_groups):
        if time.time() - start_time > DEADLINE - 3:
            break
            
        s = boundaries[gi]
        e = boundaries[gi+1]
        size = e - s
        
        if size <= 1:
            continue
        
        # Get the group's indices and sort them
        group = sorted_idx[s:e].copy()
        group.sort()
        
        # For each b in group, find a in group closest to 1.5*b
        # Vectorized: targets = 1.5 * group
        targets = (3 * group.astype(np.float64)) / 2.0
        pos = np.searchsorted(group, targets, side='left')
        
        # Check candidates at pos and pos-1
        size_int = len(group)
        
        # Candidate 1: group[pos] where pos < size
        mask1 = pos < size_int
        if np.any(mask1):
            p1 = pos[mask1]
            a_cand = group[p1]
            b_cand = group[mask1]
            # Exclude a == b
            valid = a_cand != b_cand
            if np.any(valid):
                a_v = a_cand[valid]
                b_v = b_cand[valid]
                # Compute error |a/b - 3/2| = |2a - 3b| / (2b)
                # Use exact integer arithmetic to avoid floating point issues
                # err = |2a - 3b| / (2b)
                num = np.abs(2 * a_v.astype(np.int64) - 3 * b_v.astype(np.int64))
                den = 2 * b_v.astype(np.int64)
                errs = num.astype(np.float64) / den.astype(np.float64)
                min_i = np.argmin(errs)
                err_val = errs[min_i]
                if err_val < best_error:
                    best_error = err_val
                    best_a = int(a_v[min_i])
                    best_b = int(b_v[min_i])
        
        # Candidate 2: group[pos-1] where pos > 0
        mask2 = pos > 0
        if np.any(mask2):
            p2 = pos[mask2] - 1
            a_cand = group[p2]
            b_cand = group[mask2]
            valid = a_cand != b_cand
            if np.any(valid):
                a_v = a_cand[valid]
                b_v = b_cand[valid]
                num = np.abs(2 * a_v.astype(np.int64) - 3 * b_v.astype(np.int64))
                den = 2 * b_v.astype(np.int64)
                errs = num.astype(np.float64) / den.astype(np.float64)
                min_i = np.argmin(errs)
                err_val = errs[min_i]
                if err_val < best_error:
                    best_error = err_val
                    best_a = int(a_v[min_i])
                    best_b = int(b_v[min_i])
    
    # Ensure we return valid Python ints
    return [int(best_a), int(best_b)]
