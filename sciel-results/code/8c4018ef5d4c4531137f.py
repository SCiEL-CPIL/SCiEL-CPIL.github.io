"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

def _autocorr(x):
    """Compute full autocorrelation using FFT."""
    n = len(x)
    # Pad to avoid circular convolution
    size = 2 * n
    X = fft(x, n=size)
    ac = np.real(ifft(X * np.conj(X), n=size))
    return ac[:n]  # lags 0 to n-1

def _score(x):
    """Compute the objective: max|autocorr| / (sum x)^2"""
    ac = _autocorr(x)
    max_abs = np.max(np.abs(ac))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return float('inf')
    return max_abs / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with a quadratic residue-like sequence or random ±1
    np.random.seed(42)
    x = np.random.choice([-1, 1], size=N).astype(np.float64)
    
    # Try a few structured initializations
    # Legendre symbol based (for prime p, use quadratic residues)
    # N=400 is not prime, so use a different approach
    # Start with random, then local search
    
    best_x = x.copy()
    best_score = _score(x)
    
    # Local search with simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    for iteration in range(1000000):
        if time.time() > deadline:
            break
        if T < T_min:
            break
            
        # Random flip
        idx = np.random.randint(N)
        x[idx] *= -1
        
        new_score = _score(x)
        
        if new_score < best_score:
            best_x = x.copy()
            best_score = new_score
        
        delta = new_score - best_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            pass
        else:
            # Reject
            x[idx] *= -1
        
        T *= cooling
    
    return best_x
# EVOLVE-BLOCK-END
