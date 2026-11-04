"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak(h):
    """Compute max absolute autocorrelation at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[k] for k in 1..n-1 gives positive lags
    # For real signals, acf is symmetric, so max over 1..n-1 suffices
    # Note: acf[0] is the energy, we ignore it
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _local_search(h, max_iters=1000):
    """
    Greedy local search: flip bits if it reduces the peak autocorrelation.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    
    for _ in range(max_iters):
        improved = False
        # Try flipping each bit
        # To speed up, we can randomize the order or check all
        # For N=400, checking all is 400 * FFT cost. FFT is fast.
        # 400 * (400 log 400) ~ 1.5M ops per iteration. 
        # If we do 100 iterations, 150M ops. Might be slow in Python if not vectorized.
        
        # Better: Calculate the change in score for each flip?
        # Hard to do analytically for max operation.
        # So we just do a loop, but limit it.
        
        # Randomize order to avoid bias
        order = np.random.permutation(n)
        for i in order:
            trial = current_h.copy()
            trial[i] *= -1
            s = _score(trial)
            if s < current_score - 1e-12:
                current_h = trial
                current_score = s
                improved = True
                break # Restart order after improvement
        
        if not improved:
            break
            
    return current_h, current_score

def _iterated_local_search(h_init, time_limit=10.0):
    """
    ILS: Local search, then perturb, then local search again.
    """
    n = len(h_init)
    best_h = h_init.copy()
    best_score = _score(best_h)
    current_h = h_init.copy()
    current_score = best_score
    
    start_time = time.time()
    perturbation_size = 10
    
    while time.time() - start_time < time_limit:
        # Local search
        current_h, current_score = _local_search(current_h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
        
        # Perturb
        # Flip a random number of bits
        k = np.random.randint(5, 20)
        indices = np.random.choice(n, size=k, replace=False)
        current_h[indices] *= -1
        
    return best_h, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Initial candidates
    # 1. Random
    h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
    # 2. Thue-Morse (might be a good starting point)
    h_tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    
    candidates = [h_rand, h_tm]
    
    # Run ILS on each candidate
    for h_init in candidates:
        if time.time() > deadline - 10:
            break
        h_opt, score = _iterated_local_search(h_init, time_limit=20.0)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Continue with random restarts if time permits
    while time.time() < deadline - 10:
        h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
        # Quick local search first to see if it's promising
        h_opt, score = _local_search(h_rand, max_iters=100)
        
        # If it looks good, do full ILS
        if score < best_score * 1.1: # 10% better or close
            h_opt, score = _iterated_local_search(h_opt, time_limit=5.0)
            
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # Final polish
    if best_h is not None and time.time() < deadline - 5:
        best_h, best_score = _local_search(best_h, max_iters=500)
        
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
