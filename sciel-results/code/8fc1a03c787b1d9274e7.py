import numpy as np
import time
from scipy.optimize import minimize

N = 50
w = 1.0 / 100.0

def compute_value(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    # FFT based convolution
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    
    # Ensure g is real and non-negative (up to numerical noise)
    g = np.real(g)
    g[g < 0] = 0.0
    
    m = len(g)
    # l2 calculation
    # y = [0, g, 0]
    # sum (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # = (w/3) * (2 * sum(g^2) + sum(g[:-1] * g[1:]))
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def neg_value(log_h):
    # log_h is the variable. h = exp(log_h)
    # Centering log_h to keep exp values near 1 improves numerical stability
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value(h)

def neg_value_direct(h):
    # Direct optimization in h-space with non-negativity handled by bounds
    # h is the variable directly.
    # Scale invariance means gradient along radial direction is 0.
    # L-BFGS-B with bounds [0, inf) should handle it.
    # To help, we can normalize h inside? No, that breaks gradient.
    # Just compute value.
    if np.any(h < 0):
        return 0.0 # Should not happen with bounds
    return -compute_value(h)

def construct():
    deadline = time.time() + 145.0 # Safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(123)
    
    # Define a set of structured initial guesses in log space
    starts_log = []
    
    # 1. Uniform (log 0)
    starts_log.append(np.zeros(N))
    
    # 2. Gaussians with different widths
    x = np.arange(N)
    center = N / 2.0
    for sigma_sq in [1, 2, 5, 10, 20, 40, 80]:
        g = np.exp(-((x - center)**2) / (2 * sigma_sq))
        starts_log.append(np.log(g + 1e-9))
        
    # 3. Raised Cosine
    g = 0.5 * (1 + np.cos(np.pi * x / (N - 1)))
    starts_log.append(np.log(g + 1e-9))
    
    # 4. Triangle
    g = 1 - np.abs(x - center) / center
    g[g < 0] = 1e-9
    starts_log.append(np.log(g))
    
    # 5. Two peaks
    for dist in [5, 10, 15]:
        c1 = center - dist
        c2 = center + dist
        g = np.exp(-((x - c1)**2) / 10.0) + np.exp(-((x - c2)**2) / 10.0)
        starts_log.append(np.log(g + 1e-9))
        
    # 6. Ramp
    g = np.linspace(1e-9, 1.0, N)
    starts_log.append(np.log(g))
    
    # 7. Alternating (smoothed)
    g = 0.5 + 0.5 * np.cos(2 * np.pi * x / 5.0)
    starts_log.append(np.log(g + 1e-9))
    
    # 8. Random smooth
    for _ in range(15):
        x_rand = rng.randn(N)
        kernel = np.ones(5) / 5.0
        x_rand = np.convolve(x_rand, kernel, mode='same')
        x_rand = x_rand - np.mean(x_rand)
        starts_log.append(x_rand)
        
    # Optimization loop in Log Space
    for i, start_log in enumerate(starts_log):
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                neg_value,
                start_log,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'maxfun': 15000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_cand = np.exp(res.x - np.mean(res.x))
            v_cand = compute_value(h_cand)
            
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
                
        except Exception as e:
            pass

    # Phase 2: Direct Space Optimization from Best Log Result
    # This helps escape flat regions or refine near zero values where log space is ill-conditioned
    if time.time() < deadline - 5.0:
        # Start from best_h found in log space
        # Normalize to unit L2 norm to fix scale for direct optimization?
        # The objective is scale invariant, so we can fix sum(h^2)=1 or sum(h)=1 to remove flat direction.
        # Let's fix sum(h^2) = 1.
        h_norm = best_h / np.linalg.norm(best_h)
        
        # Direct optimization with bounds h >= 0
        # We need to constrain sum(h^2) = 1? 
        # L-BFGS-B doesn't support equality constraints.
        # We can just optimize h with bounds [0, inf] and hope it stays in a good region.
        # Or we can re-parameterize? 
        # Let's just try direct optimization on h starting from best_h.
        # Since scale is flat, it might drift. 
        # To prevent drift, we can add a penalty for deviation from unit norm?
        # Or just rely on the local optimum being robust.
        
        try:
            # Use a tighter tolerance
            res_direct = minimize(
                neg_value_direct,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_cand = res_direct.x
            # Ensure non-negative
            h_cand[h_cand < 0] = 0.0
            v_cand = compute_value(h_cand)
            
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
                
        except Exception as e:
            pass

    # Phase 3: Polish with random perturbations in direct space
    if time.time() < deadline - 2.0:
        for _ in range(10):
            if time.time() > deadline - 1.0:
                break
            # Small perturbations
            perturb = rng.randn(N) * 0.01
            h_cand = best_h + perturb
            h_cand[h_cand < 0] = 0.0
            
            try:
                res_polish = minimize(
                    neg_value_direct,
                    h_cand,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                h_cand = res_polish.x
                h_cand[h_cand < 0] = 0.0
                v_cand = compute_value(h_cand)
                
                if v_cand > best_val:
                    best_val = v_cand
                    best_h = h_cand.copy()
            except:
                pass

    return list(np.asarray(best_h, dtype=float))
