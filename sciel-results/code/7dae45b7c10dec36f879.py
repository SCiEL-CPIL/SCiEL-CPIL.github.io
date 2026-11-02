"""Autoconvolution peak-sidelobe minimizer for the third autocorrelation inequality.

Metric (explicitly selected): the upper bound on C3 is taken as the peak
absolute value of the autoconvolution of the signed step function at
non-zero lags, i.e.  M(s) = max_{1 <= k < N} | (s * s)[k] |.
We place the absolute value per-lag and then take the maximum
(max_k |c[k]|), as opposed to |max_k c[k]|.  Minimizing M(s) lowers the
autoconvolution peak and hence the C3 upper bound (lower is better).

Strategy: Vectorized greedy hill-climbing with random restarts.
1. Compute the delta matrix D where D[i, k] is the change in autoconvolution
   coefficient c[k] if sign s[i] is flipped.
2. Evaluate the new peak for all N possible flips simultaneously using NumPy
   broadcasting: peaks = max(|c + D[i]|).
3. Apply the flip that minimizes the peak.
4. Repeat until no improving flip exists (local minimum).
5. Perturb the solution (flip random bits) to escape local minima and repeat.
6. Track the best solution found within the time limit.
"""
import time
import numpy as np

N = 400
_L = 1024  # next power of two >= 2N, for FFT autoconvolution


def _autoconv(s):
    """Linear autoconvolution c[k] = sum_i s[i] s[i+k], k = 0..N-1."""
    S = np.fft.fft(s, _L)
    c = np.fft.ifft(S * S).real
    return c[:N]


def _peak(c):
    """Peak absolute autoconvolution at non-zero lags."""
    return float(np.max(np.abs(c[1:N])))


def _compute_delta_matrix(s):
    """
    Compute matrix D of shape (N, N) where D[i, k] is the change in c[k]
    resulting from flipping s[i].
    """
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        si = s[i]
        # Contribution from s[i] * s[i+k] for k > 0
        # s[i+k] exists if i+k < N => k < N-i
        if i + 1 < N:
            # k ranges from 1 to N-i-1
            # s[i+1:N] has length N-i-1
            D[i, 1:N-i] += -2.0 * si * s[i+1:N]
        
        # Contribution from s[i-k] * s[i] for k > 0
        # s[i-k] exists if i-k >= 0 => k <= i
        if i > 0:
            # k ranges from 1 to i
            # s[0:i] reversed has length i
            D[i, 1:i+1] += -2.0 * si * s[0:i][::-1]
            
    return D


def _local_search(s, deadline):
    """
    Perform greedy hill-climbing on the sign sequence s.
    Returns the best sequence found.
    """
    c = _autoconv(s)
    current_peak = _peak(c)
    s = s.copy()
    
    # Limit iterations to avoid infinite loops if plateau
    max_iter = 5000
    
    for _ in range(max_iter):
        if time.time() >= deadline:
            break
            
        D = _compute_delta_matrix(s)
        
        # Evaluate new peaks for all possible flips
        # c is (N,), D is (N, N)
        # c[None, :] + D -> (N, N)
        # We care about lags 1..N-1
        new_vals = c[None, :] + D
        # Exclude lag 0 (index 0)
        new_peaks = np.max(np.abs(new_vals[:, 1:]), axis=1)
        
        best_i = np.argmin(new_peaks)
        new_peak = new_peaks[best_i]
        
        if new_peak < current_peak - 1e-9:
            # Apply flip
            s[best_i] = -s[best_i]
            # Update c efficiently? 
            # c = c + D[best_i]
            c = c + D[best_i]
            current_peak = new_peak
        else:
            # No improving flip found
            break
            
    return s, current_peak


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    best_s = None
    best_peak = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts
    # Start with a few different initializations
    inits = []
    
    # 1. Random
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # 2. Alternating
    s_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    inits.append(s_alt)
    
    # 3. Based on a Barker sequence (length 13) tiled
    # Barker sequence: 1, 1, 1, -1, 1, 1, -1 (one example, actually there are 5)
    # Let's use a known low-autocorrelation pattern if possible, or just random.
    # A simple pattern: 1, -1, 1, -1 ... is high peak.
    # Let's just use random restarts, they are robust.
    for _ in range(5):
        inits.append(rng.choice([-1.0, 1.0], size=N))

    for s0 in inits:
        if time.time() >= deadline:
            break
            
        s = s0.copy()
        # Run local search
        s, peak = _local_search(s, deadline)
        
        if peak < best_peak:
            best_peak = peak
            best_s = s.copy()
            
        # If we have time, perturb and run again
        if time.time() < deadline - 10.0:
            # Perturb: flip 10 random bits
            s_perturbed = s.copy()
            indices = rng.choice(N, size=10, replace=False)
            s_perturbed[indices] = -s_perturbed[indices]
            
            s_perturbed, peak_p = _local_search(s_perturbed, deadline)
            if peak_p < best_peak:
                best_peak = peak_p
                best_s = s_perturbed.copy()

    if best_s is None:
        best_s = inits[0]
        
    return best_s
