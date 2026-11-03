import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(h):
    """Compute the autoconvolution of h."""
    n = len(h)
    # Use FFT for efficient computation
    size = 1
    while size < 2*n - 1:
        size *= 2
    H = np.fft.fft(h, size)
    C = np.fft.ifft(H * np.conj(H))[:n]
    return C.real

def _score(h):
    """Compute the score: max|autoconv| / (sum h)^2."""
    S = np.sum(h)
    if abs(S) < 1e-10:
        return 1e10
    C = _autoconv(h)
    return np.max(np.abs(C)) / (S * S)

def _objective(x):
    """Objective function for optimization."""
    h = np.tanh(x)
    return _score(h)

def construct():
    deadline = time.time() + 140.0
    
    # Start with all ones
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try a few random initializations
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
            
        # Random initialization with some structure
        if restart == 0:
            x0 = np.zeros(N)
        elif restart == 1:
            x0 = np.random.randn(N) * 0.5
        else:
            # Create structured patterns
            pattern = np.zeros(N)
            block_size = N // (restart + 1)
            for i in range(N):
                pattern[i] = np.sin(2 * np.pi * i * (restart + 1) / N)
            x0 = pattern * 0.5
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 100, 'ftol': 1e-8})
            h = np.tanh(result.x)
            s = _score(h)
            
            if s < best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            pass
    
    # Local refinement: try flipping individual signs
    h = best_h.copy()
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(N):
            h[i] = -h[i]
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
                improved = True
            else:
                h[i] = -h[i]  # revert
    
    return best_h
