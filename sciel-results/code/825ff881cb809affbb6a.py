import numpy as np
import time
from scipy.optimize import minimize

N = 50
dx = 0.5 / N
_DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _compute_ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    
    h = np.maximum(h, 0.0) # Ensure non-negative for calculation stability
    
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    
    l2_sq = np.sum(h * h)
    if l2_sq <= 0:
        return 0.0
    
    # Discrete convolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    
    return l2_sq / (dx * l1 * max_conv)

def _smooth_max(x, sigma=0.01):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_complex(g_flat):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    # Normalize h to have L1 norm 1 to ensure scale invariance and stability
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h = h / l1
    
    l2sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=0.01)
    
    if peak == 0:
        return 0.0
    
    # Since l1 is 1, ratio is l2sq / (dx * peak)
    return -l2sq / (dx * peak)

def _optimize_complex(g0, time_limit):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    
    try:
        result = minimize(
            _objective_complex,
            g0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            # Return h corresponding to initial g0 if timeout
            g = g0.reshape(N, 2)
            h = g[:, 0]**2 + g[:, 1]**2
            return h
        
        g_opt = result.x.reshape(N, 2)
        h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
        return h_opt
    except Exception:
        g = g0.reshape(N, 2)
        h = g[:, 0]**2 + g[:, 1]**2
        return h

def _simulated_annealing(h_start, time_limit):
    """
    Perform Simulated Annealing on the complex representation g.
    h = |g|^2.
    """
    start_time = time.time()
    
    # Initialize g from h
    g_real = np.sqrt(np.maximum(h_start, 0))
    g_imag = np.zeros(N)
    g = np.concatenate([g_real, g_imag])
    
    h_curr = g_real**2 + g_imag**2
    score_curr = _compute_ratio(h_curr)
    
    best_h = h_curr.copy()
    best_score = score_curr
    
    # Annealing parameters
    T_start = 0.1
    T_end = 0.001
    alpha = 0.995 # Cooling rate
    step_size = 0.05
    
    T = T_start
    rng = np.random.RandomState(42)
    
    while T > T_end:
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Generate perturbation
        # Perturb both real and imaginary parts
        delta_real = rng.randn(N) * step_size
        delta_imag = rng.randn(N) * step_size
        
        g_new = g.copy()
        g_new[:N] += delta_real
        g_new[N:] += delta_imag
        
        # Compute new h and score
        h_new = g_new[:N]**2 + g_new[N:]**2
        score_new = _compute_ratio(h_new)
        
        # Metropolis criterion
        if score_new > score_curr:
            # Always accept if better
            g = g_new
            h_curr = h_new
            score_curr = score_new
            
            if score_new > best_score:
                best_score = score_new
                best_h = h_new.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = score_curr - score_new
            if delta < 1e-12:
                accept = True
            else:
                accept_prob = np.exp(-delta / T)
                accept = rng.random() < accept_prob
            
            if accept:
                g = g_new
                h_curr = h_new
                score_curr = score_new
        
        # Cool down
        T *= alpha
        
    return best_h

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Generate diverse initializations for h, then convert to g
    initializations_h = []
    
    # 1. Uniform
    initializations_h.append(np.ones(N))
    
    # 2. Gaussian-like
    x = np.linspace(-1, 1, N)
    initializations_h.append(np.exp(-x**2 / 0.1))
    initializations_h.append(np.exp(-x**2 / 0.5))
    initializations_h.append(np.exp(-x**2 / 2.0))
    
    # 3. Bimodal
    initializations_h.append(np.exp(-((x - 0.4)**2) / 0.1) + np.exp(-((x + 0.4)**2) / 0.1))
    initializations_h.append(np.exp(-((x - 0.6)**2) / 0.05) + np.exp(-((x + 0.6)**2) / 0.05))
    
    # 4. Triangle / tent
    initializations_h.append(1 - np.abs(x))
    
    # 5. Cosine window
    initializations_h.append(0.5 + 0.5 * np.cos(np.pi * x))
    
    # 6. Random smooth
    for seed in [1, 2, 3, 4, 5]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        initializations_h.append(h_smooth)
        
    # 7. Peak in center
    h_center = np.zeros(N)
    h_center[N//2 - 2:N//2 + 3] = 1.0
    initializations_h.append(h_center)
    
    # 8. Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:3] = 1.0
    h_edge[-3:] = 1.0
    initializations_h.append(h_edge)
    
    # 9. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    initializations_h.append(h_alt)
    
    # 10. Linear ramp
    initializations_h.append(np.linspace(0.1, 1.0, N))
    initializations_h.append(np.linspace(1.0, 0.1, N))
    
    # 11. DFT-based
    for seed in [10, 20, 30]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        h_dft = h_dft / np.max(h_dft)
        initializations_h.append(h_dft)
        
    # 12. More structured patterns
    initializations_h.append(np.sin(np.pi * (x + 1) / 2)**2)
    initializations_h.append(np.cos(np.pi * x / 2)**2)
    
    # Phase 1: Local Optimization (L-BFGS-B)
    time_per_opt = 5.0  # Reduced time per L-BFGS to leave room for SA
    
    for i, h0 in enumerate(initializations_h):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        # Convert h0 to g0
        g0_real = np.sqrt(h0)
        g0_imag = np.zeros(N)
        g0_flat = np.concatenate([g0_real, g0_imag])
        
        h_opt = _optimize_complex(g0_flat, time_per_opt)
        
        score = _compute_ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Phase 2: Simulated Annealing from the best solution found so far
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        # Run SA from the best L-BFGS result
        h_sa = _simulated_annealing(best_h, _DEADLINE - elapsed - 2)
        score_sa = _compute_ratio(h_sa)
        if score_sa > best_score:
            best_score = score_sa
            best_h = h_sa
            
    # Phase 3: If time remains, run SA from a few other diverse starts
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        # Pick a few diverse initializations that weren't the best
        diverse_indices = [0, 5, 10, 15, 20]
        for idx in diverse_indices:
            if idx < len(initializations_h):
                h0 = initializations_h[idx]
                # Quick local opt first
                g0_real = np.sqrt(h0)
                g0_imag = np.zeros(N)
                g0_flat = np.concatenate([g0_real, g0_imag])
                
                time_left = _DEADLINE - (time.time() - start_time) - 2
                if time_left < 2:
                    break
                    
                # Split time: 1/3 for L-BFGS, 2/3 for SA
                t_lbfgs = min(2.0, time_left / 3)
                t_sa = time_left - t_lbfgs - 1
                
                h_opt = _optimize_complex(g0_flat, t_lbfgs)
                score_opt = _compute_ratio(h_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = h_opt
                    
                if t_sa > 2:
                    h_sa = _simulated_annealing(h_opt, t_sa)
                    score_sa = _compute_ratio(h_sa)
                    if score_sa > best_score:
                        best_score = score_sa
                        best_h = h_sa
                        
    # Final check
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
        
    return [float(x) for x in best_h]
