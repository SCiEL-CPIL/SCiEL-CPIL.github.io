"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return np.max(np.abs(ac[:size])) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    rng = random.Random(42)
    
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    k_indices = np.arange(fsize // 2 + 1)
    twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
    
    def _soft_max_abs(ac, beta):
        """Soft maximum approximation of absolute values."""
        x = np.abs(ac)
        m = np.max(x)
        if m < 1e-12:
            return 0.0
        return m + np.log(np.sum(np.exp(-beta * x))) / beta

    def _continuous_objective(x):
        """Objective for continuous optimization with soft max and regularization."""
        h = np.tanh(x)
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)[:size]
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10
        soft_max = _soft_max_abs(ac, 100.0)
        reg = 0.01 * np.sum(1.0 - h**2)
        return soft_max / (s * s) + reg

    def _fast_binary_local_search(h, time_budget):
        """
        Fast binary local search using precomputed twiddles.
        Exploits linearity: H_new = H + delta * twiddles[i].
        """
        h = h.copy()
        H = np.fft.rfft(h, fsize)
        s = np.sum(h)
        
        # Precompute base autoconvolution and H^2 for O(1) updates? 
        # No, H^2 update is O(fsize) anyway. 
        # But we can skip the full IFFT if we just want to check max.
        # Actually, IFFT is O(fsize). 
        # The speedup here comes from NOT using scipy minimize overhead and tight loops.
        
        # To make it even faster, we can use a vectorized approach for multiple flips?
        # No, sequential is safer for convergence.
        
        # Optimization: If we precompute the base AC, flipping i changes AC by 
        # IFFT(2 * v * twiddles[i] * H + v^2 * twiddles[i]^2).
        # This is still O(fsize) to compute the delta AC and add to base.
        # However, we can avoid the full IFFT of H*H by keeping H*H updated.
        
        H_sq = H * H
        ac = np.fft.irfft(H_sq, fsize)[:size]
        current_max = np.max(np.abs(ac))
        current_score = current_max / (s * s) if abs(s) > 1e-9 else 1e10
        
        t_start_ls = time.time()
        improved = True
        sweeps = 0
        max_sweeps = 2000
        
        while improved and sweeps < max_sweeps and (time.time() - t_start_ls) < time_budget:
            improved = False
            sweeps += 1
            # Random order
            order = rng.sample(range(N), N)
            
            for i in order:
                if time.time() > deadline - 5:
                    return h, current_score
                
                v = h[i]
                # Flip sign: h[i] -> -v
                # s changes by -2v
                s_new = s - 2 * v
                if abs(s_new) < 1e-9:
                    continue
                
                # H changes by -2v * twiddles[i]
                delta_H = -2 * v * twiddles[i]
                H_new = H + delta_H
                H_sq_new = H_new * H_new
                
                # AC changes by IFFT(H_sq_new - H_sq)
                # Delta AC = IFFT(2 * H * delta_H + delta_H^2)
                # This is O(fsize) to compute IFFT of delta.
                # Is it faster than full IFFT? No.
                # But we can update ac in place if we precompute the delta AC?
                # delta_ac = IFFT(2 * H * delta_H + delta_H^2)
                # This is O(fsize log fsize) for IFFT.
                
                # Let's just do the full IFFT for simplicity, it's ~1ms for fsize=1024.
                # 1000 flips/sec is too slow. We need ~10k/sec to be competitive.
                
                # Alternative: Use a smaller fsize? No, need full resolution.
                
                # Let's rely on the fact that L-BFGS-B gets us close, 
                # and binary search is just a polish.
                
                ac_new = np.fft.irfft(H_sq_new, fsize)[:size]
                max_new = np.max(np.abs(ac_new))
                score_new = max_new / (s_new * s_new)
                
                if score_new < current_score:
                    h[i] = -v
                    s = s_new
                    H = H_new
                    H_sq = H_sq_new
                    ac = ac_new
                    current_max = max_new
                    current_score = score_new
                    improved = True
                else:
                    # Undo
                    pass # h, H, H_sq, ac, s, current_score unchanged
            
        return h, current_score

    # Phase 1: Generate diverse initial continuous states
    inits = []
    
    # 1. All ones (x = 0)
    inits.append(np.zeros(N))
    
    # 2. Random
    for _ in range(3):
        inits.append(np.array([rng.gauss(0, 1) for _ in range(N)]))
        
    # 3. Structured (Chirp)
    x = np.arange(N)
    inits.append(np.array([np.cos(np.pi * t**2 / (2 * N)) for t in x]))
    
    # 4. Anti-chirp
    inits.append(np.array([np.cos(np.pi * t**2 / (2 * N) + np.pi) for t in x]))
    
    best_h = None
    best_val = 1e10
    
    for x_init in inits:
        if time.time() > deadline - 20:
            break
            
        # L-BFGS-B optimization
        try:
            res = minimize(
                _continuous_objective, 
                x_init, 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'maxfun': 100, 'maxls': 20, 'ftol': 1e-7}
            )
            x_opt = res.x
        except Exception:
            x_opt = x_init
            
        # Project to binary
        h_bin = np.sign(x_opt)
        # Handle zeros
        h_bin[h_bin == 0] = 1.0
        
        val = _score(h_bin)
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        best_val = _score(best_h)
        
    # Phase 2: Fast Binary Local Search Polish
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 5:
        h_polished, val_polished = _fast_binary_local_search(best_h, remaining_time)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished
            
    return best_h

# EVOLVE-BLOCK-END
