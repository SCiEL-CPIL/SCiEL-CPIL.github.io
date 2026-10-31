import numpy as np
import time

N = 128
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    C = np.fft.irfft(F * F, n=fft_size)
    return C[:size]

def _score(f):
    """Compute max(|autoconv(f)|) / (sum(f))^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _local_search(f, time_budget=5.0, rng=None):
    """
    Fast local search (hill climbing) on binary {-1, 1} sequences.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    f = f.copy()
    S = np.sum(f)
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    best_f = f.copy()
    best_score = score
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try flipping a random subset of positions
        num_candidates = 10
        indices = rng.choice(N, size=num_candidates, replace=False)
        
        for i in indices:
            if time.time() > DEADLINE:
                break
            
            v = f[i]
            
            # Compute delta_C for flipping position i
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            k_end_clamped = min(k_end, len(C))
            f_slice = f[:k_end_clamped - k_start]
            
            delta_C = np.zeros_like(C)
            delta_C[k_start:k_end_clamped] = -2.0 * v * f_slice
            
            new_C = C + delta_C
            new_S = S - 2.0 * v
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score:
                f[i] = -v
                S = new_S
                C = new_C
                score = new_score
                improved = True
                
                if score < best_score:
                    best_score = score
                    best_f = f.copy()
        
        if not improved:
            break
    
    return best_f, best_score

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # Initializations
    initializations = [
        np.ones(N),
        rng.choice([-1.0, 1.0], size=N),
    ]
    
    # Add some structured patterns
    for freq in [1, 2, 4]:
        t = np.arange(N)
        f_sin = np.sign(np.sin(2 * np.pi * freq * t / N) + 0.5)
        f_sin[f_sin == 0] = 1.0
        initializations.append(f_sin)
    
    # Evaluate initializations
    for f_init in initializations:
        if time.time() > DEADLINE:
            break
        s = _score(f_init)
        if s < best_score:
            best_score = s
            best_f = f_init.copy()
    
    # Run local search from best and a few others
    num_runs = 5
    for r in range(num_runs):
        if time.time() > DEADLINE - 5.0:
            break
        
        if r == 0:
            f_start = best_f.copy()
        elif r < len(initializations):
            f_start = initializations[r].copy()
        else:
            f_start = rng.choice([-1.0, 1.0], size=N)
        
        # Time budget for each run
        remaining = DEADLINE - time.time() - 5.0
        time_budget = min(remaining / (num_runs - r), 10.0)
        if time_budget < 1.0:
            break
        
        f_ls, score_ls = _local_search(f_start, time_budget=time_budget, rng=rng)
        
        if score_ls < best_score:
            best_score = score_ls
            best_f = f_ls.copy()
    
    return best_f
