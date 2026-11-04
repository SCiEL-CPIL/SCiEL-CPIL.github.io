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
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _local_search(h, time_limit=10.0):
    """
    Combined local search:
    - Flip signs of active indices.
    - Zero out active indices.
    - Activate zero indices with +/- 1.
    Uses random ordering and early break on improvement.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        nonzero_idx = np.nonzero(best_h)[0]
        zero_idx = np.nonzero(best_h == 0)[0]
        
        if len(nonzero_idx) == 0 and len(zero_idx) == 0:
            break
            
        # 1. Flip signs
        if len(nonzero_idx) > 0:
            order = np.random.permutation(len(nonzero_idx))
            for idx_pos in order:
                i = nonzero_idx[idx_pos]
                trial = best_h.copy()
                trial[i] = -trial[i]
                s = _score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
        
        # 2. Zero out
        if not improved and len(nonzero_idx) > 0:
            order = np.random.permutation(len(nonzero_idx))
            for idx_pos in order:
                i = nonzero_idx[idx_pos]
                trial = best_h.copy()
                trial[i] = 0.0
                s = _score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
        
        # 3. Activate zeros
        if not improved and len(zero_idx) > 0:
            k = min(20, len(zero_idx))
            order = np.random.choice(len(zero_idx), size=k, replace=False)
            for idx_pos in order:
                i = zero_idx[idx_pos]
                for sign in [1.0, -1.0]:
                    trial = best_h.copy()
                    trial[i] = sign
                    s = _score(trial)
                    if s < best_score - 1e-12:
                        best_h = trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return best_h, best_score

def _simulated_annealing(h, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    Moves: flip, zero, activate.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start < time_limit:
        i = np.random.randint(n)
        trial = current_h.copy()
        
        r = np.random.random()
        if r < 0.6:
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        elif r < 0.8:
            trial[i] = 0.0
        else:
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        
        new_score = _score(trial)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = trial
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        
        T *= cooling
        
    return best_h, best_score

def _quasi_periodic_init(num_harmonics=5, seed=None):
    """
    Generate a quasi-periodic signed step function.
    h[i] = sign( sum_{k=1}^{K} cos(2*pi*f_k*i + phi_k) )
    where f_k are irrational-like frequencies.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    i = np.arange(N)
    signal = np.zeros(N)
    
    # Use incommensurate frequencies to avoid periodicity
    # sqrt(2), sqrt(3), sqrt(5), sqrt(7), sqrt(11) scaled to [0.1, 0.5]
    freqs = []
    for k in range(1, num_harmonics + 1):
        # Add some randomness to the frequency base
        base = np.sqrt(k * k + 1) / 10.0
        f = base + rng.uniform(-0.01, 0.01)
        freqs.append(f)
    
    for f in freqs:
        phase = rng.uniform(0, 2 * np.pi)
        signal += np.cos(2 * np.pi * f * i + phase)
    
    # Convert to signs
    h = np.sign(signal)
    # Ensure we have some zeros? No, sign gives -1, 0, 1.
    # The task asks for "signed step function". Usually implies values in {-1, 0, 1} or {-1, 1}.
    # The seed program used sparse (0s) and dense (-1, 1).
    # Let's keep the 0s if they occur naturally, or force them?
    # sign(0) is 0. This creates a sparse-ish structure if the signal crosses zero.
    # However, for autocorrelation minimization, dense +/-1 often works better if balanced.
    # Let's try mapping 0 to random +/-1 to make it dense, or keep 0s.
    # The current best (0.02) likely uses sparse or specific dense patterns.
    # Let's try both: one dense, one sparse.
    
    # Dense version: replace 0s with random signs
    h_dense = h.copy()
    zeros = np.nonzero(h_dense == 0)[0]
    if len(zeros) > 0:
        h_dense[zeros] = rng.choice([-1.0, 1.0], size=len(zeros))
        
    return h_dense, h

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Generate quasi-periodic initializations (structural change).
    2. Refine with local search and simulated annealing (refinement machinery).
    3. Also try a few random sparse/dense initializations as a fallback.
    4. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Quasi-periodic initializations
    # Try different numbers of harmonics and random seeds
    for trial in range(5):
        if time.time() > deadline - 20:
            break
            
        h_qp, h_qp_sparse = _quasi_periodic_init(num_harmonics=3 + trial, seed=trial)
        
        # Refine the dense quasi-periodic signal
        h_opt, score = _local_search(h_qp, time_limit=5.0)
        if time.time() < deadline - 10:
            h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
            
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
            
        # Refine the sparse quasi-periodic signal (if it has zeros)
        if np.count_nonzero(h_qp_sparse == 0) > 0:
            h_opt2, score2 = _local_search(h_qp_sparse, time_limit=3.0)
            if time.time() < deadline - 10:
                h_opt2, score2 = _simulated_annealing(h_opt2, time_limit=3.0)
                
            if score2 < best_score:
                best_score = score2
                best_h = h_opt2.copy()

    # Phase 2: Random initializations (fallback/diversity)
    # Try a few random dense and sparse sequences
    for trial in range(3):
        if time.time() > deadline - 20:
            break
            
        # Random dense
        h_rand = np.random.choice([-1.0, 1.0], size=N)
        h_opt, score = _local_search(h_rand, time_limit=5.0)
        if time.time() < deadline - 10:
            h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
            
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
            
        # Random sparse (density 50)
        h_sparse = np.zeros(N)
        idx = np.random.choice(N, size=50, replace=False)
        h_sparse[idx] = np.random.choice([-1.0, 1.0], size=50)
        h_opt2, score2 = _local_search(h_sparse, time_limit=5.0)
        if time.time() < deadline - 10:
            h_opt2, score2 = _simulated_annealing(h_opt2, time_limit=5.0)
            
        if score2 < best_score:
            best_score = score2
            best_h = h_opt2.copy()

    # Phase 3: Final polish on the best solution found
    if best_h is not None and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 0:
            best_h, best_score = _local_search(best_h, time_limit=remaining_time * 0.5)
            if time.time() < deadline - 5:
                best_h, best_score = _simulated_annealing(best_h, time_limit=remaining_time * 0.5)

    # Fallback: if no solution was found (should not happen)
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
