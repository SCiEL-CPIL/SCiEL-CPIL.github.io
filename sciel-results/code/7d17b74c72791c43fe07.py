import numpy as np
import time

N = 400
# FFT length must be at least 2*N - 1 to avoid circular convolution effects
# Next power of 2 >= 799 is 1024
FFT_LEN = 1024

def _autocorr_fast(h):
    """
    Compute the linear autocorrelation of h.
    Returns an array of length N where ac[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k].
    """
    F = np.fft.rfft(h, n=FFT_LEN)
    ac = np.fft.irfft(F * np.conj(F), n=FFT_LEN)[:N]
    return ac

def _hard_score(ac):
    """
    The metric to minimize: peak absolute value of the autocorrelation sidelobes.
    ac[0] is the main lobe (always N for +/-1). We look at ac[1:].
    """
    if len(ac) <= 1:
        return 0.0
    return float(np.max(np.abs(ac[1:])))

def _smooth_score(ac, beta):
    """
    Smooth surrogate for the peak sidelobe using log-sum-exp.
    This helps SA escape local minima of the raw max function.
    """
    if len(ac) <= 1:
        return 0.0
    # We care about the magnitude of the sidelobes
    y = np.abs(ac[1:])
    # Normalize by N to keep numbers reasonable, though not strictly necessary for SA
    y = y / N 
    if np.max(y) == 0:
        return 0.0
    m = np.max(y)
    # log-sum-exp trick for stability
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta

def _anneal_round(h_init, seed, time_budget, beta_start=50.0, beta_end=200.0):
    """
    One round of Simulated Annealing.
    Uses smooth score for acceptance, tracks hard score for best candidate.
    """
    rng = np.random.default_rng(seed)
    h = h_init.copy()
    ac = _autocorr_fast(h)
    cur_hard = _hard_score(ac)
    cur_smooth = _smooth_score(ac, beta_start)
    
    best_h = h.copy()
    best_hard = cur_hard
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    # Decay rate: we want T to drop from 0.1 to 1e-6 over the budget
    # Assuming ~10000 steps, decay per step = (T_min/T)^(1/steps)
    # But we don't know steps exactly, so use a fixed decay factor that works well
    decay = 0.9995 
    
    # Block sizes for flipping
    sizes = np.array([1, 2, 3, 4, 8, 16, 32])
    
    start_time = time.time()
    while time.time() - start_time < time_budget:
        # Adaptive beta: increase sharpness over time
        progress = min(1.0, (time.time() - start_time) / max(time_budget, 1e-9))
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Choose move type: 70% block flip, 30% single bit
        if rng.random() < 0.7:
            size = int(rng.choice(sizes))
            start = int(rng.integers(0, N - size + 1))
            sl = slice(start, start + size)
            old_vals = h[sl].copy()
            h[sl] *= -1.0
        else:
            i = int(rng.integers(0, N))
            old_vals = np.array([h[i]])
            h[i] *= -1.0
            sl = slice(i, i+1)

        ac = _autocorr_fast(h)
        new_hard = _hard_score(ac)
        new_smooth = _smooth_score(ac, beta)
        
        dS = new_smooth - cur_smooth
        
        # Acceptance criterion
        if dS <= 0 or rng.random() < np.exp(-dS / T):
            cur_hard = new_hard
            cur_smooth = new_smooth
            if new_hard < best_hard:
                best_hard = new_hard
                best_h = h.copy()
        else:
            # Revert
            h[sl] = old_vals
        
        T = max(T_min, T * decay)

    return best_h, best_hard

def _greedy_polish(h, time_budget):
    """
    Greedy local search: try flipping each bit (and small blocks) to improve hard score.
    """
    best_h = h.copy()
    best_hard = _hard_score(_autocorr_fast(h))
    start_time = time.time()
    
    # Try single bit flips first, then small blocks if time permits
    # We iterate until no improvement is found or time runs out
    while time.time() - start_time < time_budget:
        improved = False
        
        # Single bit flips
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            old_val = best_h[i]
            best_h[i] *= -1.0
            ac = _autocorr_fast(best_h)
            new_hard = _hard_score(ac)
            
            if new_hard < best_hard - 1e-12:
                best_hard = new_hard
                improved = True
            else:
                best_h[i] = old_val
        
        if not improved:
            # Try small block flips if single bits didn't help
            improved = False
            for size in [2, 4]:
                for start in range(0, N - size + 1):
                    if time.time() - start_time > time_budget:
                        break
                    sl = slice(start, start + size)
                    old_vals = best_h[sl].copy()
                    best_h[sl] *= -1.0
                    ac = _autocorr_fast(best_h)
                    new_hard = _hard_score(ac)
                    
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        improved = True
                    else:
                        best_h[sl] = old_vals
                if improved:
                    break
            
            if not improved:
                break

    return best_h, best_hard

def construct():
    try:
        t0 = time.time()
        # Evaluator kills at 150s. We aim for 145s to be safe.
        deadline = t0 + 145.0
        
        rng = np.random.default_rng(12345)
        
        # 1. Generate diverse initializations
        # Random
        h_rand = rng.choice([-1.0, 1.0], size=N)
        # Quadratic phase (chirp)
        idx = np.arange(N)
        h_chirp1 = np.sign(np.cos(2 * np.pi * idx**2 / (2*N)))
        h_chirp2 = np.sign(np.cos(2 * np.pi * idx**2 / (3*N)))
        # Sine
        h_sin = np.sign(np.sin(2 * np.pi * idx / N))
        
        inits = [h_rand, h_chirp1, h_chirp2, h_sin]
        
        best_h = None
        best_hard = float('inf')
        
        # Evaluate initializations
        for h in inits:
            ac = _autocorr_fast(h)
            hard = _hard_score(ac)
            if hard < best_hard:
                best_hard = hard
                best_h = h.copy()
                
        # 2. Simulated Annealing Rounds
        # Allocate ~100s for SA, leaving ~40s for polish
        sa_budget_total = 100.0
        num_rounds = 4
        time_per_round = sa_budget_total / num_rounds
        
        current_h = best_h.copy()
        
        for r in range(num_rounds):
            if time.time() > deadline - 45.0: # Leave 45s for polish
                break
                
            round_deadline = time.time() + time_per_round
            if round_deadline > deadline - 45.0:
                round_deadline = deadline - 45.0
                
            round_budget = round_deadline - time.time()
            if round_budget < 1.0:
                break
            
            # Perturb current best for diversity in subsequent rounds
            if r == 0:
                h_init = current_h.copy()
            else:
                h_init = current_h.copy()
                # Flip 5-10% of bits
                n_flip = int(N * 0.05)
                flip_idx = rng.choice(N, n_flip, replace=False)
                h_init[flip_idx] *= -1.0
                
            seed = 1000 + r * 7919
            h_round, hard_round = _anneal_round(h_init, seed, round_budget)
            
            if hard_round < best_hard:
                best_hard = hard_round
                best_h = h_round.copy()
                current_h = h_round.copy()
            else:
                # Keep the better of the two as the new current state
                if hard_round < _hard_score(_autocorr_fast(current_h)):
                    current_h = h_round.copy()
                else:
                    current_h = best_h.copy()

        # 3. Greedy Polish
        # Use remaining time
        polish_start = time.time()
        polish_deadline = deadline - 2.0
        polish_budget = polish_deadline - polish_start
        
        if polish_budget > 1.0:
            h_polished, hard_polished = _greedy_polish(best_h, polish_budget)
            if hard_polished < best_hard:
                best_hard = hard_polished
                best_h = h_polished

        # Ensure output is strictly +/- 1
        best_h = np.sign(best_h).astype(np.float64)
        best_h = np.where(best_h == 0, 1.0, best_h)
        
        return best_h

    except Exception:
        # Fallback: return a random sequence if something goes wrong
        # This is better than all-ones (score 1.0)
        rng = np.random.default_rng(0)
        return rng.choice([-1.0, 1.0], size=N)
