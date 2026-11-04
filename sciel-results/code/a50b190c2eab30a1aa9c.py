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
    # Zero pad to 2*n to avoid circular convolution effects
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is the energy. acf[1:n] are the non-zero lags.
    # Due to floating point, acf might have small non-zero values at the end, 
    # but we only care about 1 to n-1.
    if n == 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _greedy_local_search(h, time_limit=1.0):
    """
    Fast greedy local search.
    Tries flips, zeroing, and activating.
    Uses first-improvement strategy.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    while time.time() - start < time_limit:
        improved = False
        # Random order to avoid bias
        indices = np.random.permutation(n)
        
        for i in indices:
            # 1. Flip sign
            if best_h[i] != 0:
                trial = best_h.copy()
                trial[i] = -trial[i]
                s = _score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
            
            # 2. Zero out
            if best_h[i] != 0:
                trial = best_h.copy()
                trial[i] = 0.0
                s = _score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
            
            # 3. Activate (if zero)
            if best_h[i] == 0:
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
            # No improvement in a full pass, try a few random perturbations before giving up on this local minimum
            # Or just break. ILS will handle perturbation.
            break
            
    return best_h, best_score

def _ils_restart(h_init, time_limit=5.0):
    """
    Iterated Local Search.
    1. Local Search
    2. Perturb
    3. Repeat
    """
    current_h = h_init.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    n = len(current_h)
    
    while time.time() - start < time_limit:
        # 1. Local Search
        current_h, current_score = _greedy_local_search(current_h, time_limit=0.5)
        
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
            
        # 2. Perturb
        # Number of bits to perturb
        # Scale perturbation size with time remaining or just keep it constant
        k_perturb = np.random.randint(1, max(2, int(0.05 * n)))
        perturbed_h = current_h.copy()
        for _ in range(k_perturb):
            i = np.random.randint(n)
            r = np.random.random()
            if r < 0.5:
                # Flip or activate
                if perturbed_h[i] == 0:
                    perturbed_h[i] = 1.0 if np.random.random() < 0.5 else -1.0
                else:
                    perturbed_h[i] = -perturbed_h[i]
            else:
                # Zero out
                perturbed_h[i] = 0.0
        
        # Accept perturbation if it's better, or always accept in ILS?
        # Standard ILS: Accept if better, or with probability if worse.
        # For simplicity and speed, let's just move to the perturbed state and let local search fix it.
        # Or use Simulated Annealing acceptance.
        new_score = _score(perturbed_h)
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / 1.0):
            current_h = perturbed_h
            current_score = new_score
            
    return best_h, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    """
    deadline = time.time() + 140  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Strategy: Try different sparsities
    # Low sparsity often yields lower autocorrelation peaks
    densities = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    # Number of restarts per density
    # We have ~140s. Let's allocate time dynamically.
    
    for d in densities:
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations for this density
        for trial in range(3):
            if time.time() > deadline - 10:
                break
                
            # Create sparse random initialization
            h = np.zeros(N)
            num_active = int(N * d)
            indices = np.random.choice(N, size=num_active, replace=False)
            signs = np.random.choice([-1.0, 1.0], size=num_active)
            h[indices] = signs
            
            # Run ILS with a time slice
            # Allocate more time to lower densities initially as they are more promising
            time_slice = 10.0 if d < 0.2 else 5.0
            
            h_opt, score = _ils_restart(h, time_limit=time_slice)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
            # If we found a very good score (e.g., < 10), we can spend more time polishing
            if best_score < 10.0 and time.time() < deadline - 5:
                 # Do a final polish on the best found so far
                 h_polished, score_polished = _greedy_local_search(best_h, time_limit=5.0)
                 if score_polished < best_score:
                     best_score = score_polished
                     best_h = h_polished

    # Final Polish with remaining time
    if best_h is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            best_h, best_score = _greedy_local_search(best_h, time_limit=remaining_time - 2.0)

    if best_h is None:
        # Fallback: Single 1
        best_h = np.zeros(N)
        best_h[0] = 1.0
        best_score = _score(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
