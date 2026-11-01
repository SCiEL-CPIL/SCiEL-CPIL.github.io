"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_sidelobe(h):
    """
    Compute the maximum absolute value of the aperiodic autocorrelation sidelobes.
    h: numpy array of length N containing 1s and -1s.
    """
    n = len(h)
    # Zero-pad to at least 2*N-1 to avoid circular convolution effects
    # Using next power of 2 for FFT speed
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real
    # acf[0] is energy (N). acf[1:n] are lags 1 to n-1.
    # Note: acf[n:] are negative lags, which are symmetric to positive lags for real signals.
    return np.max(np.abs(acf[1:n]))


def _flip_delta(h, i, n):
    """
    Compute the change in the autocorrelation vector for all lags k=1..n-1
    if we flip h[i].
    
    acf[k] = sum_{t=0}^{n-1-k} h[t] * h[t+k]
    
    Flipping h[i] (h[i] -> -h[i]) changes terms where h[i] is involved.
    For lag k:
    - If i is in [0, n-1-k], then term h[i]*h[i+k] is present.
      Change: h[i]*h[i+k] -> -h[i]*h[i+k]. Delta = -2*h[i]*h[i+k].
    - If i-k is in [0, n-1-k], i.e., i is in [k, n-1], then term h[i-k]*h[i] is present.
      Change: h[i-k]*h[i] -> h[i-k]*(-h[i]). Delta = -2*h[i-k]*h[i].
    
    These two conditions are disjoint for k>0?
    Condition 1: 0 <= i <= n-1-k
    Condition 2: k <= i <= n-1
    
    If i satisfies both:
    0 <= i <= n-1-k AND k <= i <= n-1
    => k <= i <= n-1-k
    This requires k <= n-1-k => 2k <= n-1.
    
    So for small k, both terms might contribute.
    For large k, only one might contribute.
    
    Let's compute the delta vector for all k=1..n-1.
    delta[k] = -2 * h[i] * ( (h[i+k] if i+k < n else 0) + (h[i-k] if i-k >= 0 else 0) )
    
    Wait, let's verify.
    Original term at (i, i+k) in sum for lag k: exists if i+k < n. Value h[i]h[i+k].
    New term: -h[i]h[i+k]. Diff: -2 h[i] h[i+k].
    
    Original term at (i-k, i) in sum for lag k: exists if i-k >= 0. Value h[i-k]h[i].
    New term: h[i-k](-h[i]). Diff: -2 h[i-k] h[i].
    
    So delta[k] = -2 * h[i] * ( (h[i+k] if i+k < n else 0) + (h[i-k] if i-k >= 0 else 0) )
    """
    # Vectorized computation for all k
    k = np.arange(1, n)
    
    # h[i+k] part
    idx1 = i + k
    valid1 = idx1 < n
    # We need h at indices idx1 where valid1 is true, 0 otherwise
    # Create a zero array and fill
    h_plus = np.zeros(n)
    if valid1.any():
        h_plus[valid1] = h[idx1[valid1]]
        
    # h[i-k] part
    idx2 = i - k
    valid2 = idx2 >= 0
    h_minus = np.zeros(n)
    if valid2.any():
        h_minus[valid2] = h[idx2[valid2]]
        
    delta = -2.0 * h[i] * (h_plus + h_minus)
    return delta


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with a random sequence
    rng = np.random.RandomState(42)
    best_h = rng.choice([-1.0, 1.0], size=N).astype(np.float64)
    best_val = _max_sidelobe(best_h)
    
    # Precompute FFT size
    n = N
    n_fft = 1 << (2 * n - 1).bit_length()
    
    # Compute initial ACF for cost estimation
    F = np.fft.rfft(best_h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    
    # Local Search Phase
    # We will use a "steepest descent" approach: find the flip that minimizes the new max sidelobe.
    # To do this efficiently, we estimate the new max sidelobe for each candidate flip.
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        best_improvement = 0.0
        best_i = -1
        
        # Current max sidelobe
        current_max = best_val
        
        for i in range(n):
            # Calculate delta for flipping i
            delta = _flip_delta(best_h, i, n)
            
            # New ACF
            new_acf = acf + delta
            
            # New max sidelobe
            new_max = np.max(np.abs(new_acf[1:n]))
            
            # Check for improvement
            if new_max < current_max - 1e-12:
                improvement = current_max - new_max
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i = i
                    best_new_acf = new_acf
                    best_new_max = new_max
        
        if best_i != -1:
            # Apply the best flip
            best_h[best_i] *= -1.0
            best_val = best_new_max
            acf = best_new_acf
            improved = True
            
    # If time remains, try Simulated Annealing to escape local minima
    if time.time() < deadline - 5.0:
        h = best_h.copy()
        current_val = best_val
        T = 0.1
        cooling = 0.99
        
        # Recompute ACF for current h (should be same as acf above, but just in case)
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft).real[:n]
        
        while time.time() < deadline - 2.0 and T > 0.001:
            i = rng.randint(n)
            
            # Flip
            h[i] *= -1.0
            
            # Compute new ACF (expensive, but necessary for exact SA)
            # Optimization: We can use the delta method to update ACF quickly
            delta = _flip_delta(h, i, n) # Note: h is already flipped. 
            # Wait, _flip_delta computes delta based on current h.
            # If h is flipped, the delta calculation assumes we are flipping h[i] again?
            # No, _flip_delta(h, i) computes the change if we flip h[i] in the sequence h.
            # But we already flipped it. So we need to revert the ACF or calculate from scratch.
            # Let's calculate from scratch for simplicity in SA, or revert and recalc.
            # Actually, let's just revert and use the delta from the pre-flip state.
            
            # Revert h
            h[i] *= -1.0
            # Pre-flip h is the state before flip.
            # Delta calculated from pre-flip h gives the change to get to post-flip state.
            delta_pre = _flip_delta(h, i, n)
            new_acf = acf + delta_pre
            new_val = np.max(np.abs(new_acf[1:n]))
            
            delta_val = new_val - current_val
            
            if delta_val < 0 or rng.random() < np.exp(-delta_val / T):
                # Accept move
                h[i] *= -1.0
                acf = new_acf
                current_val = new_val
                
                if current_val < best_val:
                    best_h = h.copy()
                    best_val = current_val
            # Else reject (h is already reverted)
            
            T *= cooling
            
            # Periodic restart of temperature if stuck?
            # For now, simple cooling.
    
    return best_h
# EVOLVE-BLOCK-END
