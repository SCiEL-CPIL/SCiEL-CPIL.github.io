import time
import numpy as np

N = 400

def _autoconv_full(x):
    """Compute the full autoconvolution of x (length 2*N-1 effectively)."""
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    conv = np.fft.irfft(f * np.conj(f), n=2*n)
    return conv

def _score_sequence(x):
    """
    Compute the score: max|autoconv| / (sum x)^2.
    Excludes lag 0.
    """
    n = len(x)
    s = np.sum(x)
    denom = s * s
    if abs(denom) < 1e-12:
        return np.inf
    
    conv = _autoconv_full(x)
    # conv has length 2*n
    # Lag 0 is at index 0.
    # We want max abs of conv excluding index 0.
    # Note: conv[n] corresponds to lag n (which is 0 for linear autocorr of length n? No).
    # For linear autocorr of length n, lags are -(n-1)..(n-1).
    # FFT of length 2n gives circular autocorr.
    # Indices 0..n-1 correspond to lags 0..n-1.
    # Indices n..2n-1 correspond to lags n..2n-1 (which wrap to negative lags).
    # Specifically, index k corresponds to lag k.
    # Index n+k corresponds to lag n+k, which is equivalent to lag k-n.
    # We want lags 1..n-1 and -(n-1)..-1.
    # These are indices 1..n-1 and n+1..2n-1.
    # Index n is lag n, which is 0 for a signal of length n? 
    # Actually, for linear autocorr, A[n] = 0.
    # So we can just exclude index 0. The value at index n should be 0.
    # To be safe, exclude 0 and n.
    
    mask = np.ones(2*n, dtype=bool)
    mask[0] = False
    mask[n] = False
    
    max_abs = np.max(np.abs(conv[mask]))
    return max_abs / denom

def _local_search(x, time_budget=10.0, rng=None):
    """
    Greedy local search with single-bit flips.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    start_time = time.time()
    n = len(x)
    x = x.copy()
    current_score = _score_sequence(x)
    
    while time.time() - start_time < time_budget:
        improved = False
        best_i = -1
        best_score = current_score
        
        # Try all flips
        for i in range(n):
            x[i] = -x[i]
            new_score = _score_sequence(x)
            x[i] = -x[i] # revert
            
            if new_score < best_score:
                best_score = new_score
                best_i = i
        
        if best_i != -1:
            x[best_i] = -x[best_i]
            current_score = best_score
            improved = True
        else:
            break
            
    return x, current_score

def _simulated_annealing(x, start_temp=1.0, end_temp=0.01, steps=20000, time_budget=20.0, rng=None):
    """
    Simulated Annealing to escape local minima.
    """
    if rng is None:
        rng = np.random.RandomState()
        
    start_time = time.time()
    n = len(x)
    x = x.copy()
    current_score = _score_sequence(x)
    best_x = x.copy()
    best_score = current_score
    
    for step in range(steps):
        if time.time() - start_time > time_budget:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        i = rng.randint(n)
        x[i] = -x[i]
        new_score = _score_sequence(x)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[i] = -x[i]
            
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_x = None
    best_score = np.inf
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # All 1s
    candidates.append(np.ones(N, dtype=int))
    
    # All 1s with K random -1s
    rng_init = np.random.RandomState(42)
    for K in [1, 2, 5, 10, 20, 50, 100]:
        x = np.ones(N, dtype=int)
        idx = rng_init.choice(N, size=K, replace=False)
        x[idx] = -1
        candidates.append(x)
        
    # Random sequences with bias towards 1
    for p_one in [0.8, 0.9, 0.95]:
        for seed in [1, 7, 13]:
            rng = np.random.RandomState(seed)
            x = rng.choice([1, -1], size=N, p=[p_one, 1-p_one])
            candidates.append(x)
            
    # 2. Evaluate and optimize each candidate
    for i, x in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
            
        # Initial score
        s = _score_sequence(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
            
        # Local search
        x_opt, s_opt = _local_search(x, time_budget=5.0, rng=np.random.RandomState(i))
        if s_opt < best_score:
            best_score = s_opt
            best_x = x_opt.copy()
            
        # Simulated Annealing on the locally optimized result
        x_sa, s_sa = _simulated_annealing(x_opt, start_temp=0.5, end_temp=0.01, steps=5000, time_budget=10.0, rng=np.random.RandomState(i+100))
        if s_sa < best_score:
            best_score = s_sa
            best_x = x_sa.copy()
            
    return best_x
