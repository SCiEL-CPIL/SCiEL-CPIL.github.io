"""
Discrete sign-pattern search for signed step functions minimizing max|autoconv|/(sum x)^2.
Uses Simulated Annealing on the space of +/- 1 vectors.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _local_search(x, max_iters=5000):
    """
    Greedy local search: flip single bits if it improves the objective.
    """
    current_val = _objective(x)
    best_x = x.copy()
    best_val = current_val
    
    # Pre-compute indices
    indices = np.arange(N)
    
    for _ in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        # Shuffle indices to avoid bias
        np.random.shuffle(indices)
        
        for i in indices:
            # Flip bit i
            x[i] *= -1
            
            val = _objective(x)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
                improved = True
            else:
                # Revert
                x[i] *= -1
        
        if not improved:
            break
            
    return best_x, best_val

def _simulated_annealing(x0, T0=1.0, T_end=0.01, steps=50000):
    """
    Simulated Annealing on +/- 1 vectors.
    """
    x = x0.copy()
    current_val = _objective(x)
    best_x = x.copy()
    best_val = current_val
    
    rng = np.random.default_rng()
    
    for step in range(steps):
        if _check_deadline():
            break
            
        # Decay temperature
        T = T0 * (T_end / T0) ** (step / steps)
        
        # Propose move: flip a random index
        i = rng.integers(0, N)
        x[i] *= -1
        
        new_val = _objective(x)
        
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            # Revert
            x[i] *= -1
            
    return best_x, best_val

def _generate_initialization(seed):
    """Generate a random +/- 1 vector."""
    rng = np.random.default_rng(seed)
    x = rng.choice([-1, 1], size=N)
    # Ensure sum is not zero (though unlikely for random +/-1)
    s = np.sum(x)
    if abs(s) < 1:
        # Flip one bit to ensure non-zero sum
        x[0] *= -1
    return x

def construct():
    best_x = None
    best_val = float('inf')
    
    # Phase 1: Multiple restarts of Simulated Annealing
    # We use a fixed number of restarts but check time
    restarts = 20
    steps_per_sa = 20000
    
    for r in range(restarts):
        if _check_deadline():
            break
            
        x0 = _generate_initialization(seed=1000 + r)
        
        # Run SA
        x_sa, val_sa = _simulated_annealing(x0, T0=0.5, T_end=0.001, steps=steps_per_sa)
        
        if val_sa < best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            
    if best_x is None:
        # Fallback to random
        best_x = _generate_initialization(0)
        best_val = _objective(best_x)
        
    if _check_deadline():
        return best_x
        
    # Phase 2: Local Search Polish
    # Run local search on the best candidate found
    x_ls, val_ls = _local_search(best_x, max_iters=2000)
    
    if val_ls < best_val:
        best_val = val_ls
        best_x = x_ls.copy()
        
    if _check_deadline():
        return best_x
        
    # Phase 3: Try a few more specific structures if time permits
    # e.g., Random but with a specific sum constraint or bias
    # Actually, let's just do more SA restarts with different T schedules
    for r in range(10):
        if _check_deadline():
            break
            
        x0 = _generate_initialization(seed=5000 + r)
        # Different cooling schedule
        x_sa, val_sa = _simulated_annealing(x0, T0=1.0, T_end=0.01, steps=steps_per_sa)
        
        if val_sa < best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            
            # Polish immediately
            x_ls, val_ls = _local_search(best_x, max_iters=1000)
            if val_ls < best_val:
                best_val = val_ls
                best_x = x_ls.copy()
                
    return best_x
# EVOLVE-BLOCK-END
