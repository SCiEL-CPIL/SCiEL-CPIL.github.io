"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(x):
    """Compute max absolute autocorrelation / sum of squares."""
    n = len(x)
    autocorr = np.correlate(x, x, mode='full')
    sum_sq = np.sum(x**2)
    if sum_sq == 0:
        return float('inf')
    max_abs = np.max(np.abs(autocorr))
    return max_abs / sum_sq

def _peak_targeting_search(x, max_iters=20000, deadline=None):
    """
    Peak-targeting local search: identify which lag has the maximum absolute
    autocorrelation, then try flipping bits that most reduce that peak.
    """
    n = len(x)
    best_x = x.copy()
    best_score = _compute_score(x)
    x = x.copy()
    
    # Precompute autocorrelation
    def get_autocorr(x):
        return np.correlate(x, x, mode='full')
    
    autocorr = get_autocorr(x)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the lag with maximum absolute autocorrelation
        peak_lag = np.argmax(np.abs(autocorr))
        peak_val = autocorr[peak_lag]
        
        # The lag in terms of offset: autocorr index k corresponds to lag k - (n-1)
        # For a flip at position p, the change in autocorr at lag d is:
        # delta_autocorr[d] = -2 * x[p] * (sum of x[p+d] contributions)
        # Actually, let's think more carefully.
        # autocorr[k] = sum_i x[i] * x[i + k - (n-1)]
        # When we flip x[p] -> -x[p], the change is:
        # For each i where i == p or i + k - (n-1) == p:
        #   term changes from x[p]*x[j] to -x[p]*x[j], delta = -2*x[p]*x[j]
        # So delta_autocorr[k] = -2*x[p] * (x[p + k - (n-1)] + x[p - (k - (n-1))])
        #   (with bounds checking)
        
        # To reduce |autocorr[peak_lag]|, we want to find p that maximizes the
        # reduction in |autocorr[peak_lag]|.
        
        # Let lag = peak_lag - (n-1)
        lag = peak_lag - (n - 1)
        
        # delta_autocorr[peak_lag] = -2 * x[p] * (x[p+lag] if valid + x[p-lag] if valid)
        # We want this to reduce |peak_val|, i.e., we want delta to have opposite sign to peak_val
        
        # For each p, compute the contribution
        # This is O(n) per iteration, which is acceptable
        
        best_p = -1
        best_reduction = 0
        
        # Sample a subset of positions for efficiency
        sample_size = min(n, 50)
        candidates = np.random.choice(n, size=sample_size, replace=False)
        
        for p in candidates:
            # Compute delta in autocorr at peak_lag
            delta = 0.0
            # Term 1: i = p, j = p + lag
            j1 = p + lag
            if 0 <= j1 < n:
                delta += -2.0 * x[p] * x[j1]
            # Term 2: i = p - lag, j = p
            j2 = p - lag
            if 0 <= j2 < n:
                delta += -2.0 * x[j2] * x[p]
            
            # We want to reduce |peak_val + delta|
            new_abs = abs(peak_val + delta)
            old_abs = abs(peak_val)
            reduction = old_abs - new_abs
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_p = p
        
        if best_p < 0:
            # No good flip found, try a random flip
            best_p = np.random.randint(n)
        
        old_val = x[best_p]
        x[best_p] = -old_val
        
        new_score = _compute_score(x)
        if new_score < best_score:
            best_score = new_score
            best_x = x.copy()
            autocorr = get_autocorr(x)
        else:
            x[best_p] = old_val  # revert
    
    return best_x, best_score

def _spectral_relaxation(x, max_iters=5000, deadline=None):
    """
    Spectral relaxation: optimize the DFT magnitude to be flat.
    Use gradient descent on a continuous relaxation, then round to ±1.
    """
    n = len(x)
    best_x = x.copy()
    best_score = _compute_score(x)
    
    # Continuous relaxation: work with real values in [-1, 1]
    y = x.astype(float).copy()
    
    lr = 0.1
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute DFT
        Y = np.fft.fft(y)
        mag = np.abs(Y)
        
        # We want to minimize the variance of |Y| (flat spectrum)
        # Gradient of sum(|Y|^2 - mean|^2) with respect to y
        # This is related to the autocorrelation peak
        
        # Actually, let's use a simpler approach:
        # Minimize max_k |autocorr[k]| by gradient descent on y
        autocorr = np.correlate(y, y, mode='full')
        peak_lag = np.argmax(np.abs(autocorr))
        peak_val = autocorr[peak_lag]
        
        # Gradient of |autocorr[peak_lag]| with respect to y
        # autocorr[k] = sum_i y[i] * y[i + k - (n-1)]
        # d(autocorr[k])/dy[p] = y[p + k - (n-1)] + y[p - (k - (n-1))] (with bounds)
        
        lag = peak_lag - (n - 1)
        grad = np.zeros(n)
        for p in range(n):
            g = 0.0
            j1 = p + lag
            if 0 <= j1 < n:
                g += y[j1]
            j2 = p - lag
            if 0 <= j2 < n:
                g += y[j2]
            grad[p] = g * np.sign(peak_val)
        
        # Gradient descent step
        y -= lr * grad
        
        # Project to [-1, 1]
        y = np.clip(y, -1.0, 1.0)
        
        # Round to ±1 and check
        x_rounded = np.where(y >= 0, 1.0, -1.0)
        score_rounded = _compute_score(x_rounded)
        if score_rounded < best_score:
            best_score = score_rounded
            best_x = x_rounded.copy()
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate initial m-sequence
    def lfsr_sequence(length, n_bits=8, taps=None):
        if taps is None:
            taps = [4, 3, 2]
        state = 1
        seq = []
        for _ in range(length):
            out = state & 1
            seq.append(out)
            fb = 0
            for t in taps:
                fb ^= (state >> t) & 1
            state = (state >> 1) | (fb << (n_bits - 1))
        return seq
    
    raw = lfsr_sequence(N)
    x = np.array([1.0 if b == 1 else -1.0 for b in raw])
    best_x = x.copy()
    best_score = _compute_score(x)
    
    # Peak-targeting local search
    best_x, best_score = _peak_targeting_search(x, max_iters=15000, deadline=deadline)
    
    # If time remains, try spectral relaxation
    if time.time() < deadline - 10.0:
        x2, score2 = _spectral_relaxation(best_x, max_iters=3000, deadline=deadline)
        if score2 < best_score:
            best_x = x2
            best_score = score2
    
    # If more time, another round of peak-targeting from best
    if time.time() < deadline - 5.0:
        best_x, best_score = _peak_targeting_search(best_x, max_iters=10000, deadline=deadline)
    
    return best_x
# EVOLVE-BLOCK-END
