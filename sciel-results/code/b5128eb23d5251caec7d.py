"""
Strategy: Multi-start optimization with smooth log-space surrogate for the ratio objective.
The value function is scale-invariant (value = l2/(norm_1*norm_inf)), so only shape matters.
Key improvements:
1. More diverse initializations targeting flat autocorrelation
2. Adaptive beta for log-sum-exp (coarse-to-fine)
3. Nelder-Mead complement for boundary cases
4. Better time allocation with safety margin
5. Final high-precision polish with exact value
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """Compute the exact value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    value = l2 / (norm_1 * norm_inf)
    return value


def compute_log_value_smooth(h, beta=50.0):
    """Smooth log-space surrogate for the value metric."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(0, h)
    if np.sum(h) < 1e-12:
        return -1e10
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    if l2 < 1e-12:
        return -1e10
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 < 1e-12:
        return -1e10
    abs_g = g
    max_abs = np.max(abs_g)
    if max_abs < 1e-12:
        return -1e10
    shifted = beta * (abs_g - max_abs)
    log_sum_exp_term = np.log(np.sum(np.exp(shifted)))
    log_smooth_max = max_abs + (1.0 / beta) * log_sum_exp_term
    log_val = np.log(l2) - np.log(norm_1) - log_smooth_max
    return log_val


def _generate_initializations():
    """Generate diverse initial height profiles targeting flat autocorrelation."""
    x = np.linspace(0, N - 1, N)
    inits = []
    
    # Uniform (seed)
    inits.append(np.ones(N))
    
    # Gaussian centered
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/4))**2))
    
    # Bimodal: two peaks
    peak1 = np.exp(-0.5 * ((x - N*0.25) / (N/8))**2)
    peak2 = np.exp(-0.5 * ((x - N*0.75) / (N/8))**2)
    inits.append(peak1 + peak2)
    
    # Plateau
    plateau = np.zeros(N)
    plateau[N//4 : 3*N//4] = 1.0
    inits.append(plateau)
    
    # Cosine bump
    inits.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    
    # Triangular
    inits.append(1 - 2 * np.abs(x - (N-1)/2) / (N-1))
    
    # Semicircle
    inits.append(np.sqrt(np.maximum(0, 1 - ((2*x/(N-1) - 1))**2)))
    
    # Sharp central peak
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/10))**2))
    
    # Broad flat top with edges
    inits.append(np.maximum(0, 1 - np.abs(x - (N-1)/2) / (N/3)))
    
    # Step-like: high in center, low at edges
    step = np.ones(N)
    step[:N//5] = 0.2
    step[-N//5:] = 0.2
    inits.append(step)
    
    # Double step
    ds = np.ones(N)
    ds[:N//8] = 0.1
    ds[-N//8:] = 0.1
    ds[N//3 : 2*N//3] = 1.5
    inits.append(ds)
    
    # Spike: very narrow peak
    spike = np.zeros(N)
    spike[N//2] = 1.0
    inits.append(spike)
    
    # Double spike
    ds2 = np.zeros(N)
    ds2[N//4] = 1.0
    ds2[3*N//4] = 1.0
    inits.append(ds2)
    
    # Random smooth profiles via random Fourier coefficients
    rng = np.random.RandomState(42)
    for _ in range(10):
        freqs = rng.randint(1, 8, size=6)
        phases = rng.uniform(0, 2*np.pi, size=6)
        amps = rng.uniform(0.5, 1.0, size=6)
        profile = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            profile += a * np.cos(2 * np.pi * f * x / N + p)
        profile = np.maximum(0.1, profile)
        inits.append(profile)
    
    # More random smooth profiles
    for _ in range(10):
        points = np.sort(rng.uniform(0, N, size=5))
        heights = rng.uniform(0.5, 2.0, size=5)
        profile = np.interp(x, points, heights)
        inits.append(profile)
    
    # NEW: Profiles designed to produce flat autocorrelation
    # Alternating pattern (period 2) - reduces off-diagonal autocorrelation
    alt = np.ones(N)
    alt[::2] = 1.0
    alt[1::2] = 0.5
    inits.append(alt)
    
    # Period 3 pattern
    p3 = np.ones(N)
    p3[::3] = 1.0
    p3[1::3] = 0.6
    p3[2::3] = 0.8
    inits.append(p3)
    
    # Period 4 pattern
    p4 = np.ones(N)
    p4[::4] = 1.0
    p4[1::4] = 0.7
    p4[2::4] = 0.9
    p4[3::4] = 0.8
    inits.append(p4)
    
    # Ramp up then down (trapezoid)
    trap = np.zeros(N)
    ramp_len = N // 4
    trap[:ramp_len] = np.linspace(0.3, 1.0, ramp_len)
    trap[ramp_len:N-ramp_len] = 1.0
    trap[N-ramp_len:] = np.linspace(1.0, 0.3, ramp_len)
    inits.append(trap)
    
    # Double trapezoid
    dtrap = np.zeros(N)
    half = N // 2
    ramp = N // 8
    dtrap[:ramp] = np.linspace(0.2, 1.0, ramp)
    dtrap[ramp:half-ramp] = 1.0
    dtrap[half-ramp:half] = np.linspace(1.0, 0.3, ramp)
    dtrap[half:half+ramp] = np.linspace(0.3, 1.0, ramp)
    dtrap[half+ramp:N-ramp] = 1.0
    dtrap[N-ramp:] = np.linspace(1.0, 0.2, ramp)
    inits.append(dtrap)
    
    # Sinc-like (for flat spectrum, hence flat autocorrelation)
    t = (x - N/2) / (N/8)
    sinc = np.sinc(t / np.pi)  # sin(pi*t)/(pi*t)
    inits.append(np.maximum(0.05, sinc))
    
    # Raised cosine with wider support
    rc = 0.5 * (1 + np.cos(2 * np.pi * x / N * 2))
    inits.append(np.maximum(0.1, rc))
    
    # Parabolic
    par = 1 - ((x - N/2) / (N/2))**2
    inits.append(np.maximum(0.05, par))
    
    # Exponential decay from center
    exp_decay = np.exp(-np.abs(x - N/2) / (N/6))
    inits.append(exp_decay)
    
    # Two Gaussians with different widths
    g1 = np.exp(-0.5 * ((x - N*0.3) / (N/12))**2)
    g2 = 0.7 * np.exp(-0.5 * ((x - N*0.7) / (N/15))**2)
    inits.append(g1 + g2)
    
    # Cosine window (Hann)
    hann = 0.5 * (1 + np.cos(2 * np.pi * x / N))
    inits.append(np.maximum(0.05, hann))
    
    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * x / N) + 0.08 * np.cos(4 * np.pi * x / N)
    inits.append(np.maximum(0.05, blackman))
    
    # Flat-top window (very flat spectrum)
    ft = 0.21557895 - 0.41663158 * np.cos(2 * np.pi * x / N) + 0.277263158 * np.cos(4 * np.pi * x / N) - 0.083578947 * np.cos(6 * np.pi * x / N) + 0.006947368 * np.cos(8 * np.pi * x / N)
    inits.append(np.maximum(0.05, ft))
    
    # Random periodic patterns
    for period in [3, 4, 5, 6, 7]:
        pattern = rng.uniform(0.3, 1.0, size=period)
        prof = np.tile(pattern, N // period + 1)[:N]
        inits.append(prof)
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # Evaluate the seed
    seed_val = compute_value(np.ones(N))
    best_val = seed_val
    best_h = np.ones(N).copy()
    
    inits = _generate_initializations()
    
    # Phase 1: Evaluate all initializations directly with exact value
    for init in inits:
        if time.time() > deadline - 30:
            break
        init = np.asarray(init, dtype=float)
        val = compute_value(init)
        if val > best_val:
            best_val = val
            best_h = init.copy()
    
    # Phase 2: Sort by value, add perturbed versions of best
    init_vals = [(compute_value(np.asarray(init, dtype=float)), init) for init in inits]
    init_vals.sort(key=lambda x: -x[0])
    
    # Add perturbed versions of top initializations
    extra_starts = []
    rng = np.random.RandomState(123)
    for top_idx in range(min(5, len(init_vals))):
        base = init_vals[top_idx][1]
        for _ in range(5):
            perturbed = base + rng.normal(0, 0.3, N)
            perturbed = np.maximum(0, perturbed)
            extra_starts.append(perturbed)
    
    all_starts = [iv[1] for iv in init_vals] + extra_starts
    
    # Phase 3: L-BFGS-B refinement with adaptive beta
    # Use lower beta for exploration, higher for exploitation
    betas = [20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        def objective(h):
            h = np.maximum(0, h)
            log_val = compute_log_value_smooth(h, beta=beta)
            return -log_val
        
        # Refine top candidates
        n_to_refine = min(len(all_starts), max(5, int((deadline - 15 - time.time()) / 3.0)))
        
        for idx in range(n_to_refine):
            if time.time() > deadline - 15:
                break
            
            init = np.asarray(all_starts[idx], dtype=float).copy()
            
            try:
                res = minimize(
                    objective,
                    init,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-10,
                        'gtol': 1e-7
                    }
                )
                
                h_opt = np.maximum(0, res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead for cases where gradient methods struggle
    if time.time() < deadline - 10:
        nm_candidates = []
        # Best solution and nearby perturbations
        for scale in [0.1, 0.2, 0.3]:
            perturbed = best_h + rng.normal(0, scale, N)
            perturbed = np.maximum(0, perturbed)
            nm_candidates.append(perturbed)
        
        # Also try from top initializations
        for iv in init_vals[:3]:
            nm_candidates.append(np.asarray(iv[1], dtype=float))
        
        def nm_objective(h):
            h = np.maximum(0, h)
            return -compute_log_value_smooth(h, beta=50.0)
        
        for cand in nm_candidates:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    nm_objective,
                    cand,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
                )
                h_opt = np.maximum(0, res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 5: Final high-precision polish on the best solution
    if time.time() < deadline - 5:
        # Multiple perturbation attempts with tight tolerance
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Decreasing perturbation scale
            scale = 0.1 * (1 - attempt / 50.0) + 0.01
            perturbed = best_h + np.random.normal(0, scale, N)
            perturbed = np.maximum(0, perturbed)
            
            def final_objective(h):
                h = np.maximum(0, h)
                log_val = compute_log_value_smooth(h, beta=200.0)
                return -log_val
            
            try:
                res = minimize(
                    final_objective,
                    perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
                
                h_opt = np.maximum(0, res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final: ensure non-negativity and finiteness
    best_h = np.maximum(0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_value(h)}")
