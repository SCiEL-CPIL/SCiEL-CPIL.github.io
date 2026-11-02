"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(a):
    """Compute C3' = max|autocorr| / (sum(a))^2 for signed step heights."""
    S = np.sum(a)
    if abs(S) < 1e-12:
        return np.inf
    # Autoconvolution: (f*f)(k) = sum_i a_i * a_{i+k} for k = -(N-1)..(N-1)
    # We only need k >= 0 since autocorr is symmetric
    # Use FFT for efficiency
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    F = np.fft.rfft(a, n=n_fft)
    auto = np.fft.irfft(F * F.conj(), n=n_fft)[:2*N-1]
    # auto[k] = sum_{i=0}^{N-1-k} a[i] * a[i+k] for k=0..N-1
    max_abs = np.max(np.abs(auto))
    return max_abs / (S * S)

def _try_flip(a, i, score):
    """Try flipping sign at position i, return new score if improved."""
    a[i] = -a[i]
    new_score = _compute_score(a)
    if new_score < score:
        return a.copy(), new_score
    a[i] = -a[i]
    return None, score

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    best_a = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Try many random initializations
    max_inits = 50
    for init in range(max_inits):
        if time.time() > deadline:
            break
        
        # Random signs
        a = np.random.choice([-1.0, 1.0], size=N)
        
        # Ensure sum is not too small; if |sum| < threshold, adjust
        S = np.sum(a)
        if abs(S) < 5:
            # Flip some signs to make sum larger
            # Find positions where flipping increases |sum|
            for _ in range(20):
                S = np.sum(a)
                if abs(S) >= 5:
                    break
                # Flip a random position
                i = np.random.randint(N)
                old_val = a[i]
                a[i] = -a[i]
                new_S = S + 2 * a[i]  # since a[i] changed from old to -old, delta = -2*old = 2*new
                S = new_S
        
        score = _compute_score(a)
        
        if score < best_score:
            best_score = score
            best_a = a.copy()
        
        # Local optimization: flip individual signs
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping each position (random order)
            for i in np.random.permutation(N):
                if time.time() > deadline:
                    break
                old_val = a[i]
                a[i] = -a[i]
                new_score = _compute_score(a)
                if new_score < score:
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_a = a.copy()
                else:
                    a[i] = old_val
    
    if best_a is None:
        # Fallback: all ones
        return np.ones(N)
    
    return best_a
# EVOLVE-BLOCK-END
