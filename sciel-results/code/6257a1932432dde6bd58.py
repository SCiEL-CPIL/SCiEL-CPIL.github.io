import numpy as np
import time

N = 400

def _compute_objective(h):
    """Compute max_k |(h*h)[k]| / (sum(h))^2"""
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_abs = np.max(np.abs(conv))
    return max_abs / (S * S)

def construct():
    """
    Returns the sequence of signed step heights that minimizes:
    max_k |(h*h)[k]| / (sum(h))^2
    
    Structural change: Use a random initialization with local search (coordinate descent)
    to find a signed sequence that may achieve a lower objective than the constant vector.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with constant vector (known to give 1/N)
    best_h = np.ones(N)
    best_obj = _compute_objective(best_h)
    
    # Try several random signed initializations
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random signed sequence
        h = rng.choice([-1.0, 1.0], size=N).astype(float)
        
        # Local search: try flipping individual signs
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            obj = _compute_objective(h)
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                # Try flipping sign at position i
                h[i] = -h[i]
                new_obj = _compute_objective(h)
                if new_obj < obj:
                    obj = new_obj
                    improved = True
                else:
                    h[i] = -h[i]  # revert
            
            if obj < best_obj:
                best_obj = obj
                best_h = h.copy()
    
    return best_h
