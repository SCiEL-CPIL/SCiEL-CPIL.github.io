"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from numpy.fft import fft, ifft

N = 400

# EVOLVE-BLOCK-START

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def _construct_kronecker(N, alpha):
    """
    Generate a signed sequence based on the Kronecker sequence {n*alpha}.
    We map the fractional part to a sign. To ensure a non-zero sum (integral),
    we can bias the threshold.
    x[i] = 1 if frac(i*alpha) < 0.5 else -1.
    This creates a pseudo-random sequence with low discrepancy.
    """
    i = np.arange(N)
    # Use golden ratio conjugate or sqrt(2) for good distribution
    # frac(n * alpha)
    frac_parts = (i * alpha) % 1.0
    # Map to -1 and 1
    # If we want a specific bias, we can shift the threshold
    # For now, standard 50/50 split
    x = np.where(frac_parts < 0.5, 1.0, -1.0)
    return x

def _local_search(x, max_iter=2000, time_limit=140.0):
    """
    Sparse sign-flipping local search.
    We try to flip signs of elements that contribute most to the peak autoconvolution.
    """
    best_x = x.copy()
    best_score = _score(best_x)
    start_time = time.time()
    
    # Precompute autoconvolution
    A = _autoconv(best_x)
    S = np.sum(best_x)
    
    # Identify the index of the maximum absolute autoconvolution
    peak_idx = np.argmax(np.abs(A))
    peak_val = A[peak_idx]
    
    # Gradient of score w.r.t flipping x[k]
    # Flipping x[k] changes A[j] by -2 * x[k] * x[j-k] (for j where x[j-k] exists)
    # Specifically, A_new[j] = A[j] - 2*x[k]*x[j-k] (linear conv logic needs care with boundaries)
    # Actually, A = x * x. 
    # d(A[j])/d(x[k]) = x[j-k] + x[j-N+k] (if we consider circular? No, linear)
    # For linear convolution A[j] = sum_i x[i] x[j-i].
    # If we change x[k] to -x[k], the change in A[j] is -2*x[k]*x[j-k] (if j-k is valid index) 
    # plus -2*x[k]*x[j-(N-1-k)]? No.
    # Let's just recompute A for candidate flips in a batch or one by one.
    # One by one is O(N^2) per flip if naive, but with FFT it's O(N log N).
    # N=400, N^2=160k. 2000 iters * 160k = 320M ops. Might be slow in Python.
    # Let's use FFT-based updates or just check top candidates.
    
    # Better: Compute the "sensitivity" of the peak to each flip.
    # The peak is at `peak_idx`.
    # Flipping x[k] changes A[peak_idx] by -2 * x[k] * x[peak_idx - k] (if 0 <= peak_idx-k < N)
    # It also affects other A[j].
    # Simple heuristic: Try flipping x[k] where x[k] * x[peak_idx - k] has the same sign as peak_val.
    # This will reduce |A[peak_idx]|.
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
        
        A = _autoconv(best_x)
        S = np.sum(best_x)
        if abs(S) < 1e-12:
             # If sum is zero, score is inf. We need to avoid this.
             # The Kronecker sequence usually has small sum but not zero.
             # If it becomes zero, flip the element with largest magnitude (all are 1) to change sum.
             # Actually, if sum is 0, the denominator is 0. The task says "divided by squared integral".
             # If integral is 0, it's undefined or inf. We should avoid S=0.
             # Let's just pick a random flip to escape.
             k = np.random.randint(N)
        else:
            peak_idx = np.argmax(np.abs(A))
            peak_val = A[peak_idx]
            
            # Calculate potential improvement for each flip
            # We want to minimize max(|A|).
            # Let's focus on the current peak.
            # Flipping x[k] changes A[j] for all j.
            # Approximate change in peak:
            # delta_A_peak = -2 * x[k] * x[peak_idx - k] (if valid)
            
            best_improvement = 0
            best_k = -1
            
            # Check a subset of k to save time, or all if fast enough
            # N=400, checking all 400 is fast.
            for k in range(N):
                # Calculate new A[peak_idx]
                # A_new[peak_idx] = A[peak_idx] - 2 * x[k] * x[peak_idx - k] (if 0 <= peak_idx-k < N)
                # Note: x[peak_idx - k] might be out of bounds for linear conv.
                # A[j] = sum_{i} x[i] x[j-i] where 0<=i<N and 0<=j-i<N.
                # So for a specific k, it contributes to A[j] if 0 <= j-k < N.
                # Here j = peak_idx.
                # Condition: 0 <= peak_idx - k < N  =>  peak_idx - N < k <= peak_idx.
                
                if 0 <= peak_idx - k < N:
                    contrib = x[k] * x[peak_idx - k]
                    new_peak_val = peak_val - 2 * contrib
                    # We want to reduce |new_peak_val|
                    # Current penalty is |peak_val|
                    # New penalty for this specific term is |new_peak_val|
                    # But other terms in A might increase.
                    # Heuristic: just look at the peak reduction.
                    if abs(new_peak_val) < abs(peak_val) - best_improvement:
                        best_improvement = abs(peak_val) - abs(new_peak_val)
                        best_k = k
                else:
                    # k does not affect the current peak directly in the sum?
                    # Wait, A[peak_idx] depends on x[k] only if x[peak_idx-k] is in range.
                    # If not, flipping x[k] doesn't change A[peak_idx].
                    # But it changes other A[j].
                    # For simplicity, ignore these or treat them as no immediate peak reduction.
                    pass

            if best_k != -1:
                # Apply flip
                best_x[best_k] *= -1
                new_score = _score(best_x)
                if new_score < best_score:
                    best_score = new_score
                else:
                    # Revert if it didn't improve the global score (max abs / S^2)
                    # The local peak reduction might have increased another peak or reduced S^2 significantly.
                    best_x[best_k] *= -1
            else:
                # No single flip reduces the main peak. Try random flip or stop.
                # Try a few random flips to escape local minima
                found = False
                for _ in range(5):
                    k = np.random.randint(N)
                    best_x[k] *= -1
                    new_score = _score(best_x)
                    if new_score < best_score:
                        best_score = new_score
                        found = True
                        break
                    else:
                        best_x[k] *= -1
                if not found:
                    # Stuck? Maybe try flipping multiple bits? 
                    # Or just break if we've tried many times.
                    # Let's continue a bit longer with randoms.
                    pass
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_x = None
    best_score = np.inf
    
    # Try a few different Kronecker constants
    # alpha = 1/sqrt(2), 1/sqrt(3), golden ratio conjugate, etc.
    alphas = [
        1.0 / np.sqrt(2),
        1.0 / np.sqrt(3),
        (1 + np.sqrt(5)) / 2, # Golden ratio
        np.sqrt(2),
        np.sqrt(3),
        0.7,
        0.3
    ]
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
            
        x = _construct_kronecker(N, alpha)
        
        # Ensure sum is not zero. 
        # If sum is zero, the score is inf. 
        # We can add a small bias by flipping one bit to make sum non-zero.
        S = np.sum(x)
        if abs(S) < 1e-6:
            # Flip the first element to make sum non-zero
            x[0] *= -1
            S = np.sum(x)
            if abs(S) < 1e-6:
                # Still zero? Flip another
                x[1] *= -1
                S = np.sum(x)
        
        # Local search
        x_opt, sc = _local_search(x, max_iter=500, time_limit=deadline - 5.0)
        
        if sc < best_score:
            best_score = sc
            best_x = x_opt
            
        # Check time
        if time.time() > deadline - 15.0:
            break
    
    # If no valid solution found (e.g., all inf), return a safe fallback
    if best_x is None or not np.isfinite(best_score):
        # Fallback: all ones
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
