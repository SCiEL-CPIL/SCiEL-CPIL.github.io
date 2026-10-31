"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(a):
    """Compute max absolute value of linear autoconvolution of a."""
    n = len(a)
    size = 2 * n
    f = np.fft.rfft(a, size=size)
    conv = np.fft.irfft(f * np.conj(f), size=size)[:2 * n - 1]
    return np.max(np.abs(conv))


def _score(a):
    """Compute the score: max_abs_autoconv / (sum a)^2"""
    s = np.sum(a)
    if abs(s) < 1e-12:
        return float('inf')
    max_ac = _autoconv_max_abs(a)
    return max_ac / (s * s)


def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Start with all ones (baseline)
    best = np.ones(N)
    best_score = _score(best)
    
    np.random.seed(42)
    
    def _local_search(a, max_iter=20000):
        """Simulated annealing on ±1 flips."""
        a = a.copy()
        current_score = _score(a)
        best_local = a.copy()
        best_local_score = current_score
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            pos = np.random.randint(N)
            old_val = a[pos]
            a[pos] = -old_val
            new_score = _score(a)
            
            temp = 1.0 * (0.01 / 1.0) ** (i / max_iter)
            if new_score < current_score or np.random.random() < np.exp(
                -((new_score - current_score) / max(1e-10, temp))
            ):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local = a.copy()
            else:
                a[pos] = old_val  # revert
        
        return best_local, best_local_score
    
    # Try a few restarts with different initial conditions
    for r in range(8):
        if time.time() > deadline - 10:
            break
        
        if r == 0:
            a = np.ones(N)
        elif r == 1:
            # Start with a pattern that has good autocorrelation properties
            a = np.ones(N)
            # Flip every other element in a structured way
            for i in range(0, N, 3):
                a[i] = -1
        else:
            a = np.random.choice([-1, 1], size=N)
        
        local_best, local_score = _local_search(a, max_iter=10000)
        
        if local_score < best_score:
            best_score = local_score
            best = local_best
    
    return best
# EVOLVE-BLOCK-END
