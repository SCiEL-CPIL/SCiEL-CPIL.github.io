import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _soft_max(v, tau=0.1):
    """Soft maximum of vector v."""
    v_max = np.max(v)
    if v_max > 700:
        return v_max
    return v_max + tau * np.log(np.sum(np.exp((v - v_max) / tau)))

def _autoconv_peak(h):
    """Compute the peak of autoconvolution (normalized)."""
    h_conv = np.convolve(h, h, mode='full')
    return np.max(h_conv)

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Parametric search for good initializations
    # Uniform
    h_init = np.ones(N) / N
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussians
    for c in np.linspace(5, N-5, 15):
        for w in np.linspace(1, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(1, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # Random sparse initialization
    for _ in range(200):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(3, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = np.random.rand(num_peaks) * 0.5 + 0.5
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using L-BFGS-B on log-scale variables with annealed Soft-Max
    def neg_obj_soft(x, tau):
        h = np.exp(x)
        h = h / np.sum(h)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        sm = _soft_max(h_conv, tau=tau)
        if sm == 0:
            return 0.0
        return -(l2_sq / sm)

    def neg_obj_exact(x):
        h = np.exp(x)
        h = h / np.sum(h)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0: return 0.0
        return -(l2_sq / max_conv)

    # Multi-stage optimization with annealing
    h_start = best_h.copy()
    h_start[h_start < 1e-10] = 1e-10
    x0 = np.log(h_start)
    
    # Stage 1: Coarse optimization with large tau
    tau_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    x_current = x0.copy()
    
    for tau in tau_schedule:
        if time.time() > deadline - 5:
            break
        try:
            def obj_coarse(x):
                return neg_obj_soft(x, tau)
            
            res = minimize(obj_coarse, x_current, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < 0:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                    x_current = res.x.copy()
        except Exception:
            pass

    # Stage 2: Fine optimization with exact objective
    if time.time() < deadline - 10:
        h_start = best_h.copy()
        h_start[h_start < 1e-10] = 1e-10
        x0 = np.log(h_start)
        
        try:
            res = minimize(neg_obj_exact, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success or res.fun < 0:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Stage 3: Multiple restarts around best solution
    if time.time() < deadline - 15:
        h_base = best_h.copy()
        h_base[h_base < 1e-10] = 1e-10
        
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution
            noise_scale = 0.1 / (restart + 1)
            h_pert = h_base * (1 + noise_scale * (np.random.rand(N) - 0.5))
            h_pert = np.maximum(h_pert, 1e-10)
            h_pert /= np.sum(h_pert)
            
            x_pert = np.log(h_pert)
            
            try:
                res = minimize(neg_obj_exact, x_pert, method='L-BFGS-B', 
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.success or res.fun < 0:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
            except Exception:
                pass

    # Stage 4: Coordinate descent polish
    if time.time() < deadline - 3:
        h_cd = best_h.copy()
        h_cd /= np.sum(h_cd)
        
        improved = True
        max_iters = 100
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if not improved:
                break
            improved = False
            
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                
                # Try adjusting h[i] while keeping sum=1
                h_old = h_cd[i]
                if h_old < 1e-15:
                    continue
                    
                # Binary search for optimal h[i]
                lo, hi = 1e-15, h_old * 10 + 1e-3
                for _ in range(30):
                    mid = (lo + hi) / 2
                    h_try = h_cd.copy()
                    h_try[i] = mid
                    # Renormalize
                    h_try = h_try / np.sum(h_try)
                    s_try = _score_raw(h_try)
                    s_old = _score_raw(h_cd)
                    if s_try > s_old:
                        lo = mid
                        if s_try > best_score:
                            best_score = s_try
                            best_h = h_try.copy()
                        h_cd = h_try.copy()
                        improved = True
                    else:
                        hi = mid
                        break
                
                # Reset to best found
                h_cd = best_h.copy() if improved else h_cd

    # Final exact polish
    if time.time() < deadline - 1:
        h_final = best_h.copy()
        h_final[h_final < 1e-12] = 1e-12
        x_final = np.log(h_final)
        
        try:
            res = minimize(neg_obj_exact, x_final, method='L-BFGS-B', 
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.success or res.fun < 0:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass

    return list(best_h)

# EVOLVE-BLOCK-END
