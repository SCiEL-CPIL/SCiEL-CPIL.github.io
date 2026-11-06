"""Additive bases with few representations - simulated annealing rewrite.

We model the problem as a 0/1 indicator vector x of length 129.
The ordered convolution y = x * x gives the ordered pair-sum multiplicities.
The task asks for y[k] > 0 for all k in [128, 256] and minimizes max y[k].
Note: The task description says "Pair representations are unordered and permit
equal summands". The current program uses ordered convolution.
Let's check the contract again: "Minimize the largest representation multiplicity".
"Local pairs are unordered... paper's ordered convention differs by symmetry".
"""
import time
import random
import numpy as np

_N = 128
_LEN = 129
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets
_DEADLINE = 55.0      # safety margin before the 60 s kill

_START = time.time()


def _elapsed():
    return time.time() - _START


def _conv(x):
    """Return the integer convolution of x with itself (length 2*_LEN-1)."""
    return np.convolve(x.astype(np.int64), x.astype(np.int64), mode="full")

def _unordered_counts(y_ordered, x):
    """
    Convert ordered convolution counts to unordered pair counts in the window.
    U_k = (O_k + D_k) / 2, where D_k is 1 if k is even and k/2 in A.
    """
    w = y_ordered[_T0:_T1 + 1].copy()
    # Calculate diagonal correction
    # k in [128, 256]. k is even if k%2==0.
    # if k is even, mid = k//2. if mid in A (x[mid]==1), then D_k = 1.
    # Vectorized check:
    ks = np.arange(_T0, _T1 + 1)
    is_even = (ks % 2 == 0)
    mids = ks // 2
    # mids range: 64 to 128.
    # x has length 129 (indices 0..128).
    # Check if x[mids] is 1.
    diag_corr = np.where(is_even, x[mids], 0)
    u_counts = (w + diag_corr) // 2
    return u_counts

def _valid(u_counts):
    """True iff every target in [128,256] has at least one representation."""
    return bool(np.all(u_counts > 0))

def _peak(u_counts):
    return int(u_counts.max())


def _make_seed(rng, size_hint=64):
    """Build a valid 0/1 indicator using a simple greedy coverage."""
    x = np.zeros(_LEN, dtype=np.int64)
    
    # Start with a dense subset to ensure coverage quickly
    # Use every 2nd or 3rd number?
    # Let's just pick random elements until valid.
    candidates = np.arange(_LEN)
    rng.shuffle(candidates)
    
    # Add elements until valid
    y = _conv(x)
    u = _unordered_counts(y, x)
    while not _valid(u):
        # Find uncovered
        uncovered = np.where(u == 0)[0] + _T0
        if len(uncovered) == 0:
            break
            
        # Pick the candidate that covers the most uncovered targets
        best_c = None
        best_gain = -1
        for c in candidates:
            if x[c] == 1:
                continue
            # How many uncovered targets does c cover?
            # c + a = t => a = t - c.
            # We need a in A.
            # Gain is number of t in uncovered such that t-c in A.
            # Also t=2c if c in A (but c is not in A yet, so we add it, then 2c is covered by c+c).
            gain = 0
            for t in uncovered:
                a = t - c
                if 0 <= a < _LEN and x[a] == 1:
                    gain += 1
                # If a == c, it will be covered by c+c after adding c.
                # But we are evaluating the gain of adding c to the *current* set.
                # If we add c, new pairs are (c, a) for a in A_old, and (c, c).
                # So (c,c) covers 2c.
                if 2*c == t:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c is None or best_gain <= 0:
            # No single element covers any uncovered target?
            # This should not happen if A is not empty and we have enough range.
            # Force add a random one
            best_c = candidates[0]
            
        x[best_c] = 1
        y = _conv(x)
        u = _unordered_counts(y, x)
        
    return x


def _greedy_load_balance(x, rng, max_time):
    """
    Greedy swap optimization minimizing peak unordered multiplicity.
    Uses precomputed shift vectors for fast convolution updates.
    """
    y = _conv(x)
    u = _unordered_counts(y, x)
    if not _valid(u):
        return x, _peak(u), y

    best_x = x.copy()
    best_peak = _peak(u)
    
    # Precompute shift vectors S_i for all i in [0, 128]
    # S_i[k] = 1 if k-i in [0, 128] and x[k-i] == 1
    # Note: S_i depends on x. We need to update them if x changes.
    # Instead of precomputing all 129 vectors of length 257 (33k ints),
    # we can just compute the delta on the fly using slicing, which is fast.
    
    current_peak = best_peak
    
    while _elapsed() < max_time:
        improved = False
        # Get list of indices in A and not in A
        in_A = np.where(x == 1)[0]
        not_in_A = np.where(x == 0)[0]
        
        if len(in_A) == 0 or len(not_in_A) == 0:
            break
            
        best_swap = None
        best_new_peak = current_peak
        best_new_sum_sq = np.sum(u**2) # Secondary metric
        
        # Iterate over all potential swaps
        # To speed up, we can sample or do full scan. Full scan is 129*129 ~ 16k.
        # Each evaluation involves updating y and u.
        # y_new = y - 2*S_i + delta_2i + 2*S_j - 2*delta_{i+j} + delta_2j
        # We can compute this in O(1) if we have S_i and S_j.
        
        # Let's precompute S_i for current x?
        # S_i is just x shifted by i.
        # We can create a 2D array Shifts[i, k] = x[k-i] if valid.
        # This is 129x257.
        Shifts = np.zeros((_LEN, 2 * _LEN - 1), dtype=np.int64)
        for i in range(_LEN):
            # x is length 129.
            # k ranges 0 to 256.
            # k-i ranges -i to 256-i.
            # Valid if 0 <= k-i < 129.
            lo = max(0, i)
            hi = min(2 * _LEN - 1, i + _LEN)
            # indices in y: lo to hi
            # indices in x: 0 to hi-lo
            Shifts[i, lo:hi] = x[0:hi-lo]
            
        # Now iterate swaps
        for i in in_A:
            # Removing i: y_temp = y - 2*Shifts[i] + delta_2i
            # We don't want to allocate new arrays for every i.
            # We can compute the effect on u directly?
            # u = (y + diag)/2.
            # This is getting complex. 
            # Alternative: Just do a random hill climbing with a few random swaps per iteration, 
            # but check the whole neighborhood of the *current* peak reducers?
            
            # Let's stick to a simpler, faster heuristic:
            # 1. Identify the 'hottest' targets (max u).
            # 2. Try to remove an element that contributes to these hot targets.
            # 3. Try to add an element that contributes to 'cold' targets (min u > 0) or doesn't increase hot targets.
            
            pass # Placeholder for complex loop, will implement simpler random swap search below
        
        # Simplified Random Swap Search with Validation
        # Try 50 random swaps
        for _ in range(50):
            if _elapsed() > max_time:
                break
            i = rng.choice(in_A)
            j = rng.choice(not_in_A)
            
            # Calculate new y
            # y_new = y - 2*Shifts[i] + delta_2i + 2*Shifts[j] - 2*delta_{i+j} + delta_2j
            y_new = y.copy()
            y_new -= 2 * Shifts[i]
            y_new += 2 * Shifts[j]
            
            # Diagonal corrections
            if 2*i < len(y_new):
                y_new[2*i] += 1
            if 2*j < len(y_new):
                y_new[2*j] += 1
            if i+j < len(y_new):
                y_new[i+j] -= 2
                
            # Check validity and peak
            u_new = _unordered_counts(y_new, x - np.eye(_LEN)[i] + np.eye(_LEN)[j]) 
            # Note: x changes, so diag_corr in _unordered_counts needs new x.
            # This is slow. 
            # Optimization: 
            # The diagonal term D_k only changes for k=2i and k=2j.
            # We can update u manually.
            
            # Let's just use the full _unordered_counts for correctness, it's fast enough for 50 swaps.
            x_new = x.copy()
            x_new[i] = 0
            x_new[j] = 1
            
            u_new = _unordered_counts(y_new, x_new)
            
            if not _valid(u_new):
                continue
                
            peak_new = _peak(u_new)
            if peak_new < best_new_peak:
                best_new_peak = peak_new
                best_swap = (i, j)
                best_new_sum_sq = np.sum(u_new**2)
                improved = True
                
        if improved:
            i, j = best_swap
            x[i] = 0
            x[j] = 1
            y = _conv(x) # Recalculate to avoid drift
            u = _unordered_counts(y, x)
            current_peak = best_new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_x = x.copy()
        else:
            # Stuck. Try a random restart or perturbation?
            # Just break to save time if we can't improve in a batch.
            # Or try a single random flip (add/remove) to escape local optimum?
            # Removing might break validity. Adding might increase peak.
            # Let's try adding a random element and then removing a different one (2-step).
            # For now, just break.
            break
            
    return best_x, best_peak, _conv(best_x)


def _one_run(rng, size_hint, max_time):
    x = _make_seed(rng, size_hint)
    u = _unordered_counts(_conv(x), x)
    if not _valid(u):
        return None
    x, peak, y = _greedy_load_balance(x, rng, max_time)
    return x, peak, y


def construct():
    global _START
    _START = time.time()

    rng = random.Random(12345)
    np.random.seed(12345)

    best_x = None
    best_peak = float("inf")

    # Phase 1: Generate a good seed and optimize
    # We don't need multiple sizes if the greedy optimizer is good.
    # Start with a medium size.
    size = 64
    remaining = _DEADLINE - _elapsed() - 2
    if remaining > 1:
        result = _one_run(rng, size, max_time=remaining)
        if result is not None:
            x, peak, y = result
            if peak < best_peak:
                best_peak = peak
                best_x = x.copy()

    # Phase 2: If we have time, try another seed with different size
    if best_x is not None and _elapsed() < _DEADLINE - 5:
        size2 = 72
        remaining = _DEADLINE - _elapsed() - 2
        if remaining > 1:
            result = _one_run(rng, size2, max_time=remaining)
            if result is not None:
                x, peak, y = result
                if peak < best_peak:
                    best_peak = peak
                    best_x = x.copy()

    # Fallback: if we somehow have no valid answer, use the full set
    if best_x is None:
        best_x = np.ones(_LEN, dtype=np.int64)
        best_peak = _peak(_conv(best_x))

    # Convert to a sorted list of integers
    result = [int(i) for i in np.where(best_x == 1)[0]]
    return result
