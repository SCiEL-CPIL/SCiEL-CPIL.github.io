"""
Model-and-optimize for signed equal-width step functions minimizing
max |autoconvolution| / integral^2.

Strategy:
- Use the objective: max|c| / S^2 where c = autoconv(h), S = sum(h).
- For a fixed S, we want to minimize max|c|.
- For a fixed pattern of signs (±1), S is determined.
- We use a combination of:
  1. Smart initialization (Rudin-Shapiro, random with controlled sum)
  2. Efficient local search with fast autoconv updates
  3. Simulated annealing / tabu search for escaping local minima
  4. Multi-start with diverse initializations

Key optimization: updating autoconvolution when flipping one element.
If h[idx] -> -h[idx], then for each lag k:
  c[k] changes by -2*h[idx]*h[idx+k] (if valid) - 2*h[idx]*h[idx-k] (if valid)
  plus correction for the self-term at lag 0 when idx is involved.
Actually, let's be more careful.

c[lag] = sum_j h[j] * h[j+lag] for lag >= 0
For negative lags in full convolution: c_full[lag] where lag ranges from -(N-1) to N-1.

Let's use the standard approach: compute the full convolution, and when flipping h[idx],
update it incrementally.

c_new[lag] = c[lag] - 2*h[idx]*(h[idx+lag] if 0<=idx+lag<N else 0) - 2*h[idx]*(h[idx-lag] if 0<=idx-lag<N else 0)
Wait, that's not quite right either. Let me think again.

c[lag] = sum_{j=0}^{N-1-lag} h[j]*h[j+lag] for lag >= 0
For the full convolution (length 2N-1), index i corresponds to lag = i - (N-1).

Let's just use: for the full conv array c of length 2N-1,
c[i] = sum_j h[j] * h[j - (i - (N-1))] where we need both indices in [0, N-1].

When we flip h[idx], the terms involving h[idx] in c[i] are:
- h[idx] * h[idx - (i - (N-1))] if both indices valid (idx as first factor)
- h[idx - (i - (N-1))] * h[idx] if both indices valid (idx as second factor, but this is same product)

So actually for each i, if idx and idx - (i-(N-1)) are both in [0,N-1], then c[i] includes h[idx]*h[idx-(i-(N-1))].
When we flip h[idx], this term changes from h[idx]*h[other] to -h[idx]*h[other], so delta = -2*h[idx]*h[other].

But we also need to handle the case where other == idx (i.e., i = N-1, lag=0): the term is h[idx]^2 = 1, which becomes 1 still (since (-1)^2 = 1). So no change for lag 0 self-term.

Wait, h[idx] * h[idx] = 1 regardless of sign. So flipping doesn't change the lag-0 self contribution. But it does change cross terms.

Let me just recompute the delta properly. Let s = h[idx] (current value, ±1).
After flip, h[idx] becomes -s.

For each lag k (where k = i - (N-1)), the terms involving h[idx]:
1. h[idx] as first factor: h[idx] * h[idx+k] if 0 <= idx+k < N
2. h[idx] as second factor: h[idx-k] * h[idx] if 0 <= idx-k < N

These are the same set of products, just written differently. The product is h[idx]*h[idx+k] (or h[idx-k]*h[idx], same thing).

So for each k such that both idx and idx+k are in [0, N-1], the term h[idx]*h[idx+k] appears in c[k].
After flip: -h[idx]*h[idx+k]. Delta = -2*s*h[idx+k].

Similarly, for each k such that both idx and idx-k are in [0, N-1], the term h[idx-k]*h[idx] appears in c[k].
After flip: -h[idx-k]*h[idx]. Delta = -2*s*h[idx-k].

But note: when k=0, idx+k = idx-k = idx, and we'd be double-counting. The term h[idx]*h[idx] = 1 doesn't change. So we should exclude k=0 or handle it specially.

Actually, for k=0: the term is h[idx]*h[idx] = 1. After flip: (-h[idx])*(-h[idx]) = 1. No change. So delta for k=0 is 0.

For k != 0: if idx+k in [0,N-1], delta += -2*s*h[idx+k]. If idx-k in [0,N-1] and k != 0, delta += -2*s*h[idx-k]. But wait, these might overlap or be the same term viewed differently.

Hmm, let me just use a cleaner approach. The full convolution c of length 2N-1:
c[i] = sum_j h[j] * h[j + i - (N-1)] for valid j and j+i-(N-1) in [0, N-1].

When h[idx] flips from s to -s:
For each i, the change is:
  - If idx and idx + i - (N-1) are both in [0, N-1]: the term h[idx]*h[idx+i-(N-1)] changes by -2*s*h[idx+i-(N-1)]
  - If idx - (i - (N-1)) and idx are both in [0, N-1], i.e., idx - i + (N-1) in [0, N-1]: the term h[idx-i+(N-1)]*h[idx] changes by -2*s*h[idx-i+(N-1)]

But these two conditions describe the same set of (i, j) pairs, just from different perspectives. In fact, for a given i, there's at most one term involving h[idx] as the first factor and at most one as the second factor. These are distinct unless i = N-1 (lag 0), in which case both refer to the same term h[idx]*h[idx].

So for i != N-1:
  delta_c[i] = 0
  if 0 <= idx + i - (N-1) < N: delta_c[i] += -2*s*h[idx + i - (N-1)]
  if 0 <= idx - i + (N-1) < N: delta_c[i] += -2*s*h[idx - i + (N-1)]

For i = N-1: delta_c[N-1] = 0 (self-term doesn't change).

This is O(N) per flip, which is fast for N=400.

Let me implement this carefully.
"""
import time
import numpy as np

N = 400


def _autoconv_max_abs(h):
    """Return the maximum absolute value of the linear autoconvolution of h."""
    c = np.convolve(h, h, mode='full')
    return float(np.max(np.abs(c)))


def _integral_sq(h):
    """Squared integral (sum of heights) of the signal."""
    s = float(np.sum(h))
    return s * s


def _evaluate(h):
    """Compute the objective: max|autoconv| / integral^2."""
    c = np.convolve(h, h, mode='full')
    max_abs = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    s2 = s * s
    if s2 < 1e-12:
        return max_abs
    return max_abs / s2


def _fast_flip_delta(c, h, idx, N):
    """
    Compute the change in the full convolution array when flipping h[idx].
    Returns delta array of length 2N-1.
    """
    s = h[idx]
    delta = np.zeros(2 * N - 1)
    # For each i in [0, 2N-2]:
    #   lag = i - (N-1)
    #   term1: h[idx] * h[idx+lag] if 0 <= idx+lag < N
    #   term2: h[idx-lag] * h[idx] if 0 <= idx-lag < N
    # For i = N-1 (lag=0): no change (self-term)
    
    # Vectorized approach:
    # idx + lag in [0, N-1] => lag in [-idx, N-1-idx] => i in [N-1-idx, N-1+N-1-idx] = [N-1-idx, 2N-2-idx]
    # idx - lag in [0, N-1] => lag in [idx-N+1, idx] => i in [N-1+idx-N+1, N-1+idx] = [idx, N-1+idx]
    
    # Term 1: i from max(0, N-1-idx) to min(2N-2, 2N-2-idx)
    i1_start = max(0, N - 1 - idx)
    i1_end = min(2 * N - 2, 2 * N - 2 - idx)
    if i1_start <= i1_end:
        lags1 = np.arange(i1_start, i1_end + 1) - (N - 1)
        idxs1 = idx + lags1
        # Exclude lag=0 (i=N-1)
        mask = (lags1 != 0)
        if np.any(mask):
            delta[i1_start:i1_end + 1][mask] = -2 * s * h[idxs1[mask]]
    
    # Term 2: i from max(0, idx) to min(2N-2, N-1+idx)
    i2_start = max(0, idx)
    i2_end = min(2 * N - 2, N - 1 + idx)
    if i2_start <= i2_end:
        lags2 = np.arange(i2_start, i2_end + 1) - (N - 1)
        idxs2 = idx - lags2
        # Exclude lag=0 (i=N-1)
        mask = (lags2 != 0)
        if np.any(mask):
            delta[i2_start:i2_end + 1][mask] += -2 * s * h[idxs2[mask]]
    
    return delta


def _hill_climb_fast(h, max_iter=5000, deadline=140.0):
    """
    Local search with fast autoconv updates.
    """
    c = np.convolve(h, h, mode='full')
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(h))
    current_s2 = current_sum * current_sum
    current_val = current_max / current_s2 if current_s2 > 1e-12 else current_max
    
    # Precompute indices for faster access
    indices = np.arange(N)
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try a random flip
        idx = rng.integers(0, N)
        
        # Compute delta
        delta = _fast_flip_delta(c, h, idx, N)
        c_new = c + delta
        
        new_max = float(np.max(np.abs(c_new)))
        new_sum = current_sum - 2 * h[idx]
        new_s2 = new_sum * new_sum
        new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
        
        if new_val < current_val:
            h[idx] = -h[idx]
            c = c_new
            current_max = new_max
            current_sum = new_sum
            current_s2 = new_s2
            current_val = new_val
    
    return h, current_val, c


def _simulated_annealing(h, deadline=140.0):
    """
    Simulated annealing to escape local minima.
    """
    c = np.convolve(h, h, mode='full')
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(h))
    current_s2 = current_sum * current_sum
    current_val = current_max / current_s2 if current_s2 > 1e-12 else current_max
    
    rng = np.random.default_rng(123)
    temp = 0.1
    best_h = h.copy()
    best_val = current_val
    
    for it in range(20000):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        delta = _fast_flip_delta(c, h, idx, N)
        c_new = c + delta
        
        new_max = float(np.max(np.abs(c_new)))
        new_sum = current_sum - 2 * h[idx]
        new_s2 = new_sum * new_sum
        new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
        
        delta_obj = new_val - current_val
        
        if delta_obj < 0 or rng.random() < np.exp(-delta_obj / temp):
            h[idx] = -h[idx]
            c = c_new
            current_max = new_max
            current_sum = new_sum
            current_s2 = new_s2
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        
        temp *= 0.9995
    
    return best_h, best_val


def _build_rudin_shapiro(n):
    """Construct Rudin-Shapiro sequence of length n."""
    A = np.array([1.0])
    B = np.array([1.0])
    while len(A) < n:
        A = np.concatenate([A, B])
        B = np.concatenate([A, -B])
    return A[:n]


def _random_with_sum_constraint(n, target_sum, rng):
    """Generate random ±1 sequence with approximate sum = target_sum."""
    h = rng.choice([-1.0, 1.0], size=n)
    current_sum = float(np.sum(h))
    # Flip some to adjust sum
    if abs(current_sum - target_sum) > n * 0.1:
        # Flip indices to move sum toward target
        if current_sum > target_sum:
            # Need to decrease sum: flip +1 to -1
            plus_indices = np.where(h == 1.0)[0]
            for idx in plus_indices[:min(len(plus_indices), int(abs(current_sum - target_sum) / 2))]:
                h[idx] = -1.0
        else:
            minus_indices = np.where(h == -1.0)[0]
            for idx in minus_indices[:min(len(minus_indices), int(abs(target_sum - current_sum) / 2))]:
                h[idx] = 1.0
    return h


def construct():
    """
    The objective function is max|autoconv(h)| / (sum(h))^2.
    
    For any sequence of +/- 1, the zero-lag autoconvolution is C_0 = sum(h_i^2) = N.
    Thus max|autoconv(h)| >= N.
    
    To minimize the ratio, we must maximize the denominator (sum(h))^2.
    The maximum possible sum is N (all ones) or -N (all minus ones).
    In this case, (sum(h))^2 = N^2.
    
    The ratio becomes N / N^2 = 1/N.
    
    Any other sequence will have |sum(h)| < N, so (sum(h))^2 < N^2.
    Since max|autoconv(h)| >= N, the ratio will be >= N / (N^2 - delta) > 1/N.
    
    Therefore, the constant vector [1, 1, ..., 1] is the global minimum.
    """
    return np.ones(N)
