import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin for 120s limit (or 150s? Prompt says 120s in one place, 150s in another. "Each evaluation must finish within 120 s". "evaluator KILLS... after 150s". Let's use 115s to be safe for 120s limit).

def _compute_c3_max(h):
    """
    Compute max |C3[k]| for a binary sequence h.
    C3[k] = sum_i h[i] h[i+k] h[i+2k]
    """
    h_float = h.astype(np.float64)
    # Zero-pad to avoid circular convolution
    # We need length of conv(h, h) to be 2N-1.
    # Then conv(conv(h,h), h) to be 3N-2.
    # FFT size should be at least 3N-2. Next power of 2 is 2048 for N=400 (3*400=1200).
    size = 1 << (2 * N - 1).bit_length()
    
    h_fft = np.fft.fft(h_float, n=size)
    h2_fft = h_fft * h_fft
    c3_fft = h2_fft * h_fft
    c3_full = np.fft.ifft(c3_fft).real
    
    # The valid lags k are such that we have at least one i where i, i+k, i+2k are in [0, N-1].
    # Max k is (N-1)//2.
    # C3[k] in the linear convolution output corresponds to index k.
    # Wait, standard linear convolution c[n] = sum h[i] g[n-i].
    # Here we are computing h * h * h.
    # The peak value is what we want.
    # Usually, for aperiodic autocorrelation, we look at the central part.
    # For C3, the "DC" is k=0.
    # The prompt implies "upper bound on C3", likely max over all k.
    
    # We take the max absolute value over the valid range.
    # Valid range for k: 0 to N-1.
    # Actually, for k > N, the terms don't overlap in the same way?
    # The definition C3[k] = sum_i h[i] h[i+k] h[i+2k] implies valid i range shrinks.
    # The linear convolution result contains all these sums at indices corresponding to the lags.
    
    return np.max(np.abs(c3_full[:N]))

def _score_normalized(h):
    """
    The score reported in the prompt seems to be max|C3| / N.
    """
    return _compute_c3_max(h) / N

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    best_score = float('inf')
    best_h = np.random.choice([1, -1], N)
    
    # Pre-allocate work buffer
    h_work = best_h.copy().astype(np.float64)
    
    # We will run several restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time_left() < 5.0: # Need at least 5s for a few passes
            break
            
        # Initialization strategies
        if r == 0:
            # Start with random
            h_work = np.random.choice([1, -1], N).astype(np.float64)
        elif r == 1:
            # Sine wave initialization
            x = np.linspace(0, 1, N, endpoint=False)
            h_work = np.sign(np.sin(2 * np.pi * 3 * x)).astype(np.float64)
        elif r == 2:
            # Block pattern +1+1-1-1
            pattern = np.array([1, 1, -1, -1])
            h_work = np.tile(pattern, N // 4 + 1)[:N].astype(np.float64)
        else:
            # Random restarts
            h_work = np.random.choice([1, -1], N).astype(np.float64)
            
        current_score = _score_normalized(h_work)
        if current_score < best_score:
            best_score = current_score
            best_h = h_work.copy()
            
        # Greedy Coordinate Descent
        # We iterate over indices. If we improve, we keep it.
        # We do multiple passes until no improvement or time runs out.
        
        # Optimization: Pre-compute indices? No, just iterate.
        
        while time_left() > 2.0:
            improved = False
            # Shuffle order to break symmetry? Optional, but can help.
            # For N=400, random permutation is cheap.
            order = np.random.permutation(N)
            
            for idx in order:
                if time_left() < 1.0:
                    break
                    
                # Flip
                h_work[idx] *= -1
                
                # Evaluate
                new_score = _score_normalized(h_work)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Update global best
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h_work.copy()
                else:
                    # Revert
                    h_work[idx] *= -1
            
            if not improved:
                break
            
            # Check time again
            if time_left() < 2.0:
                break
                
    # Ensure final output is +/- 1
    best_h = np.sign(best_h).astype(np.float64)
    if np.all(best_h == 0):
        best_h = np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score_normalized(h))
