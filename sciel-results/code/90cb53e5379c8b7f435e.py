import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT

def _autocorr_fast(h):
    # Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    # Use real FFT trick: autocorr of real signal = IFFT(|FFT|^2)
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac

def _hard_score(h):
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # The metric is the max of the absolute values of the sidelobes (k=1..N-1) normalized by DC
    return float(np.max(np.abs(ac[1:])) / dc)

def _smooth_score_from_ac(ac, beta):
    """
    Smooth surrogate: log-sum-exp of squared normalized sidelobes.
    This approximates max(y) where y = (ac[1:]/dc)^2.
    """
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    # log(sum(exp(beta * (y - m)))) / beta + m
    # = m + log(sum(exp(beta * (y - m)))) / beta
    # To avoid overflow, we subtract m.
    z = beta * (y - m)
    # Clip z to prevent exp overflow, though y-m <= 0 so z <= 0
    z = np.clip(z, -100, 0) 
    return m + float(np.log(np.sum(np.exp(z)))) / beta

def _greedy_polish(h, deadline, rng):
    """
    Greedy local search to minimize the hard score.
    Tries single bit flips and small block flips.
    """
    cur_ac = _autocorr_fast(h)
    cur_hard = _hard_score(h)
    best_h = h.copy()
    best_hard = cur_hard
    
    sizes = [1, 2, 4, 8]
    
    # Iterate until no improvement or time runs out
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Shuffle order to avoid bias
        order = rng.permutation(N)
        
        # Single bit flips
        for i in order:
            if time.time() > deadline - 2.0:
                break
            h[i] *= -1.0
            new_ac = _autocorr_fast(h)
            new_hard = _hard_score(h)
            if new_hard < best_hard - 1e-12:
                best_hard = new_hard
                best_h = h.copy()
                cur_ac = new_ac
                cur_hard = new_hard
                improved = True
            else:
                h[i] *= -1.0 # revert
        
        # Small block flips (optional, can be expensive)
        if time.time() < deadline - 5.0:
             for size in [2, 4]:
                 for start in range(0, N - size + 1, 2): # step 2 to save time
                     if time.time() > deadline - 2.0:
                         break
                     sl = slice(start, start + size)
                     old = h[sl].copy()
                     h[sl] *= -1.0
                     new_ac = _autocorr_fast(h)
                     new_hard = _hard_score(h)
                     if new_hard < best_hard - 1e-12:
                         best_hard = new_hard
                         best_h = h.copy()
                         cur_ac = new_ac
                         cur_hard = new_hard
                         improved = True
                     else:
                         h[sl] = old
                         # revert
                         h[sl] = old
    return best_h, best_hard

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0  # Safety margin for 120s limit
        rng = np.random.default_rng(20240517)
        
        best_h_global = np.ones(N)
        best_score_global = 1e9
        
        # We will run multiple restarts.
        # Estimate cost of one SA run + polish.
        # Let's try to fit as many as possible, but ensure at least 1-2.
        # If N=400, FFT is fast. SA steps ~ 1000-5000.
        
        max_restarts = 5
        time_per_restart = (deadline - t0 - 5.0) / max_restarts
        
        for r in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
                
            # 1. Initialize
            h = rng.choice(np.array([-1.0, 1.0]), size=N)
            
            # 2. Simulated Annealing
            cur_ac = _autocorr_fast(h)
            cur_hard = _hard_score(h)
            beta = 20.0
            cur_smooth = _smooth_score_from_ac(cur_ac, beta)
            
            T = 0.05
            T_min = 1e-4
            decay = 0.9995
            
            best_h_sa = h.copy()
            best_hard_sa = cur_hard
            
            # SA Loop
            steps = 0
            max_steps = 2000 # Adjust based on time
            while time.time() < t0 + r*time_per_restart + time_per_restart*0.8:
                # Generate move: flip a random contiguous block
                # Size distribution: mostly small, occasionally larger
                size = 1
                if rng.random() < 0.2:
                    size = int(rng.integers(2, 10))
                elif rng.random() < 0.05:
                    size = int(rng.integers(10, 20))
                
                if size >= N:
                    size = N - 1
                
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                
                old_vals = h[sl].copy()
                h[sl] *= -1.0
                
                new_ac = _autocorr_fast(h)
                new_hard = _hard_score(h)
                new_smooth = _smooth_score_from_ac(new_ac, beta)
                
                dS = new_smooth - cur_smooth
                
                # Accept if better in smooth score, or with probability exp(-dS/T)
                # Also, always accept if hard score improves significantly (optional, can be noisy)
                if dS <= 0 or rng.random() < np.exp(-dS / T):
                    cur_ac = new_ac
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard_sa:
                        best_hard_sa = new_hard
                        best_h_sa = h.copy()
                else:
                    h[sl] = old_vals
                
                T = max(T_min, T * decay)
                steps += 1
            
            # 3. Greedy Polish
            # Allocate remaining time for this restart to polish
            polish_deadline = t0 + (r+1)*time_per_restart
            if r == max_restarts - 1:
                polish_deadline = deadline - 5.0 # Ensure final restart has enough time
                
            polished_h, polished_score = _greedy_polish(best_h_sa, polish_deadline, rng)
            
            if polished_score < best_score_global:
                best_score_global = polished_score
                best_h_global = polished_h.copy()
                
        return best_h_global
    except Exception:
        return np.ones(N)
