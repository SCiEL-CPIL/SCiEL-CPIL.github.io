"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Construct a signed step function of length N that minimizes the maximum absolute autoconvolution
    divided by the squared integral.
    
    The score is the upper bound on C3' (maximum absolute autoconvolution / integral^2).
    Lower is better.
    
    Strategy:
    1. Start with random sign vectors.
    2. Use gradient-based optimization on a continuous relaxation, then round to signs.
    3. Use local search (flip single bits) to improve.
    4. Deadline guard: return best found before 150s limit (with safety margin).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def autoconv(x):
        """Compute autoconvolution of x with itself (linear, not circular)."""
        return np.convolve(x, x, mode='full')
    
    def score(x):
        """Compute the C3' metric: max |autoconv| / (sum x)^2.
        If sum is near zero, return a large penalty."""
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        ac = np.convolve(x, x, mode='full')
        return np.max(np.abs(ac)) / (s * s)
    
    # Start with several random sign vectors
    best_x = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try a few initializations
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Random signs
        x = rng.choice([-1.0, 1.0], size=N)
        val = score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Try a simple pattern: alternating
        if i == 1:
            x = np.ones(N) * (1 - 2 * (np.arange(N) % 2))
            val = score(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
        
        # Try blocks
        if i == 2:
            block_size = N // 4
            x = np.zeros(N)
            for b in range(4):
                x[b*block_size:(b+1)*block_size] = 1 if b % 2 == 0 else -1
            val = score(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    
    # Local search: flip individual bits to reduce the score
    # Use a greedy hill-climbing approach with restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        x = best_x.copy() if best_x is not None else rng.choice([-1.0, 1.0], size=N)
        current_val = score(x)
        
        # Greedy local search: try flipping each bit, keep best improvement
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_flip = None
            best_flip_val = current_val
            
            # Sample a subset of positions to try (for speed)
            if N <= 200:
                positions = np.arange(N)
            else:
                positions = rng.choice(N, size=min(200, N), replace=False)
            
            for idx in positions:
                if time.time() > deadline - 5:
                    break
                
                x_old = x[idx]
                x[idx] = -x[idx]
                new_val = score(x)
                x[idx] = x_old
                
                if new_val < best_flip_val:
                    best_flip_val = new_val
                    best_flip = idx
            
            if best_flip is not None:
                x[best_flip] = -x[best_flip]
                current_val = best_flip_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_x = x.copy()
        
        # After local search, if we improved, keep going
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
    
    # One more pass: try block flips (flip consecutive segments)
    if time.time() < deadline - 10 and best_x is not None:
        x = best_x.copy()
        current_val = score(x)
        
        # Try flipping blocks of various sizes
        for block_len in [2, 4, 8, 16]:
            if time.time() > deadline - 10:
                break
            for start in range(0, N - block_len + 1, max(1, block_len // 2)):
                if time.time() > deadline - 10:
                    break
                
                old_vals = x[start:start+block_len].copy()
                x[start:start+block_len] = -old_vals
                new_val = score(x)
                x[start:start+block_len] = old_vals
                
                if new_val < current_val:
                    x[start:start+block_len] = -old_vals
                    current_val = new_val
                    if current_val < best_val:
                        best_val = current_val
                        best_x = x.copy()
    
    return best_x if best_x is not None else np.ones(N)

# EVOLVE-BLOCK-END
