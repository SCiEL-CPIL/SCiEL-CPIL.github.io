"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_max_abs(x):
    """Compute max absolute autocorrelation (excluding lag 0) via FFT."""
    n = len(x)
    # Zero-pad to avoid circular convolution issues
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)[:n]
    # Exclude lag 0
    return np.max(np.abs(acf[1:]))

def _greedy_polish(x, max_iters=2000):
    """Greedy coordinate descent: flip signs to reduce max abs autocorrelation."""
    x = x.copy()
    best_val = _autocorr_max_abs(x)
    no_improve = 0
    
    for it in range(max_iters):
        improved = False
        # Shuffle order to avoid bias
        order = np.random.permutation(N)
        for i in order:
            # Try flipping x[i]
            x[i] *= -1
            new_val = _autocorr_max_abs(x)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                x[i] *= -1  # revert
        if not improved:
            no_improve += 1
            if no_improve >= 2:
                break
        else:
            no_improve = 0
    return x, best_val

def _anneal(x, start_temp=0.1, end_temp=0.001, max_iters=5000):
    """Simulated annealing to escape local minima."""
    x = x.copy()
    best_x = x.copy()
    best_val = _autocorr_max_abs(x)
    current_val = best_val
    n_iters = max_iters
    temp = start_temp
    
    for it in range(n_iters):
        # Decay temperature
        t = it / n_iters
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Pick random position to flip
        i = np.random.randint(N)
        x[i] *= -1
        new_val = _autocorr_max_abs(x)
        
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[i] *= -1  # revert
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_x = None
    best_val = np.inf
    
    # Multiple restarts with different initial conditions
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize: random, or with some structure
        if restart == 0:
            x = np.ones(N)
        elif restart == 1:
            x = np.sign(np.sin(np.linspace(0, 2*np.pi, N, endpoint=False)))
            x[x == 0] = 1
        elif restart == 2:
            x = np.sign(np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)))
            x[x == 0] = 1
        elif restart == 3:
            # Random walk pattern
            x = np.ones(N)
            for i in range(1, N):
                if np.random.random() < 0.3:
                    x[i] = -x[i-1]
                else:
                    x[i] = x[i-1]
        else:
            x = np.random.choice([-1, 1], N)
        
        # Greedy polish
        x, val = _greedy_polish(x, max_iters=1500)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        if time.time() > deadline:
            break
        
        # Annealing phase if time permits
        x, val = _anneal(x, start_temp=0.05, end_temp=0.001, max_iters=3000)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        if time.time() > deadline:
            break
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
