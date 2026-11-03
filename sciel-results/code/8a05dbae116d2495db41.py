"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    This is differentiable and allows gradient-based optimization.
    """
    abs_ac = np.abs(ac)
    # Shift for numerical stability
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    # log-sum-exp trick: log(sum(exp(x_i))) = max(x_i) + log(sum(exp(x_i - max(x_i))))
    # We want (1/beta) * log(sum(exp(beta * |ac_i|)))
    # = (1/beta) * (beta * m + log(sum(exp(beta * (|ac_i| - m)))))
    # = m + (1/beta) * log(sum(exp(beta * (|ac_i| - m))))
    scaled = beta * (abs_ac - m)
    # Clip to avoid overflow/underflow issues in exp
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    We optimize this to find good candidates, then evaluate the true objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_chirp(n):
    """Generate a chirp sequence: h[i] = exp(i * pi * i^2 / n)"""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    # Real part for signed step function
    return np.cos(phase)

def _gen_barker_like(n):
    """
    Generate a sequence with good autocorrelation properties.
    Uses a simple construction: alternating signs with some structure.
    """
    h = np.zeros(n)
    for i in range(n):
        # Use a simple pattern that tends to have lower sidelobes
        # This is a heuristic; not guaranteed to be optimal
        if i % 4 == 0:
            h[i] = 1.0
        elif i % 4 == 1:
            h[i] = -1.0
        elif i % 4 == 2:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _peak_suppression(h, deadline, max_iters=50):
    """
    Directly target the argmax of the autoconvolution and adjust h to reduce it.
    This is a dedicated phase that complements the smooth surrogate optimization.
    """
    h = h.copy()
    best_val = _score(h)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        ac = _autoconv_fft(h)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        if peak_val < 1e-10:
            break
            
        # Gradient-like step: perturb h in direction that reduces the peak
        # The autoconvolution at lag k is sum_i h[i]*h[i+k]
        # To reduce |ac[peak_idx]|, we can try small perturbations
        
        # Try a few small perturbations around the peak
        step_sizes = [0.01, 0.005, 0.001]
        improved_this_iter = False
        
        for step in step_sizes:
            if time.time() > deadline:
                break
            
            # Perturb elements that contribute to the peak
            # ac[peak_idx] = sum_i h[i] * h[i+peak_idx] for valid i
            # We can try adjusting h[i] or h[i+peak_idx]
            
            # Simple approach: try small random perturbations weighted by contribution
            rng = np.random.default_rng(hash(_score(h)) % (2**32))
            
            for _ in range(5):
                if time.time() > deadline:
                    break
                    
                h_perturbed = h.copy()
                # Perturb a random subset of elements
                n_perturb = max(1, N // 10)
                indices = rng.choice(N, n_perturb, replace=False)
                h_perturbed[indices] += step * rng.standard_normal(n_perturb)
                
                # Normalize to maintain sum
                s = np.sum(h_perturbed)
                if abs(s) > 1e-9:
                    h_perturbed = h_perturbed / s
                
                val = _score(h_perturbed)
                if val < best_val:
                    h = h_perturbed
                    best_val = val
                    improved_this_iter = True
                    break
        
        if not improved_this_iter:
            # Try more structured perturbations
            for i in range(0, N, max(1, N//20)):
                if time.time() > deadline:
                    break
                
                for delta in [-step, step]:
                    h_perturbed = h.copy()
                    h_perturbed[i] += delta
                    s = np.sum(h_perturbed)
                    if abs(s) > 1e-9:
                        h_perturbed = h_perturbed / s
                    val = _score(h_perturbed)
                    if val < best_val:
                        h = h_perturbed
                        best_val = val
                        improved_this_iter = True
                        break
                if improved_this_iter:
                    break
        
        if not improved_this_iter:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 7. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    # Adaptive beta schedule: tighter progression
    beta_schedule = [60, 120, 250]
    
    # Optimization loop using smooth surrogate
    for h_init in inits:
        if time.time() > deadline - 25:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Use smooth objective for gradient-based optimization with adaptive beta
        try:
            for beta in beta_schedule:
                if time.time() > deadline - 15:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-6})
                h_opt = res.x
                
                # Check validity
                if abs(np.sum(h_opt) - 1.0) < 0.5:
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
                        # Update h_norm for next beta stage
                        h_norm = h_opt / np.sum(h_opt)
                else:
                    # Try to renormalize
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-6:
                        h_opt = h_opt / s_opt
                        val = _score(h_opt)
                        if val < best_score:
                            best_score = val
                            best_h = h_opt.copy()
                            h_norm = h_opt.copy()
        except Exception:
            pass

    # Dedicated peak-suppression phase
    if time.time() < deadline - 10:
        h_suppressed, val_suppressed = _peak_suppression(best_h, deadline - 5, max_iters=30)
        if val_suppressed < best_score:
            best_score = val_suppressed
            best_h = h_suppressed.copy()

    # Final local search refinement
    if time.time() < deadline - 3:
        h_refined, val_refined = _local_search_flip(best_h, max_passes=5, deadline=deadline-2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
