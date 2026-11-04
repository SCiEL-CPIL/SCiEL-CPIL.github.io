import numpy as np
import time
import scipy.optimize as opt
from scipy.optimize import differential_evolution

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

# Precompute fine grid for fast convolution ratio
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / (integral f)^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    return float(np.max(conv) / integral_sq)

def gaussians_to_h(params):
    """
    Convert Gaussian superposition parameters to step heights.
    params = [mu1, sigma1, amp1, mu2, sigma2, amp2, ..., base]
    h(x) = base + sum_i amp_i * exp(-0.5 * ((x - mu_i)/sigma_i)^2)
    """
    n_gauss = (len(params) - 1) // 3
    base = params[-1]
    h = np.full(N, base, dtype=np.float64)
    for i in range(n_gauss):
        mu = params[3*i]
        sigma = params[3*i + 1]
        amp = params[3*i + 2]
        sigma = abs(sigma) + 1e-6
        h += amp * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
    return h

def rects_to_h(params):
    """
    Convert rectangular pulse parameters to step heights.
    params = [x1, w1, a1, x2, w2, a2, ..., base]
    h(x) = base + sum_i a_i * 1_{x in [x_i - w_i/2, x_i + w_i/2]}
    """
    n_rects = (len(params) - 1) // 3
    base = params[-1]
    h = np.full(N, base, dtype=np.float64)
    for i in range(n_rects):
        x_c = params[3*i]
        w = abs(params[3*i + 1]) + 1e-4
        a = params[3*i + 2]
        # Map to bin indices
        left = (x_c - w/2 + 0.25) / W
        right = (x_c + w/2 + 0.25) / W
        i_left = max(0, int(np.floor(left)))
        i_right = min(N - 1, int(np.ceil(right)))
        if i_right >= i_left:
            h[i_left:i_right+1] += a
    return h

def objective_gauss(params):
    h = gaussians_to_h(params)
    neg = h[h < 0]
    if len(neg) > 0:
        penalty = 100.0 * np.sum(neg**2)
    else:
        penalty = 0.0
    r = ratio_from_h(h)
    return r + penalty

def objective_rects(params):
    h = rects_to_h(params)
    neg = h[h < 0]
    if len(neg) > 0:
        penalty = 100.0 * np.sum(neg**2)
    else:
        penalty = 0.0
    r = ratio_from_h(h)
    return r + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Start with uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Gaussian superposition optimization (from current program)
    for n_gauss in [3, 4, 5]:
        if time.time() > deadline - 40.0:
            break
        
        n_params = 3 * n_gauss + 1
        bounds = []
        for _ in range(n_gauss):
            bounds.append((-0.25, 0.25))
            bounds.append((0.01, 0.15))
            bounds.append((-1.0, 5.0))
        bounds.append((0.0, 2.0))
        
        n_restarts = 10
        for i in range(n_restarts):
            if time.time() > deadline - 35.0:
                break
            x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            try:
                result = opt.minimize(
                    objective_gauss, x0, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_opt = gaussians_to_h(result.x)
                h_clipped = np.clip(h_opt, 0, None)
                true_val = ratio_from_h(h_clipped)
                update_best(h_clipped, true_val)
            except Exception:
                pass
    
    # Phase 2: Rectangular pulse optimization with differential evolution (new strategy)
    if time.time() < deadline - 30.0:
        for n_rects in [3, 4, 5]:
            if time.time() > deadline - 25.0:
                break
            
            n_params = 3 * n_rects + 1
            bounds = []
            for _ in range(n_rects):
                bounds.append((-0.25, 0.25))  # center
                bounds.append((0.005, 0.25))   # width
                bounds.append((0.0, 10.0))     # amplitude
            bounds.append((0.0, 2.0))          # base
            
            try:
                result = differential_evolution(
                    objective_rects, bounds, maxiter=50, popsize=15,
                    tol=1e-8, seed=42, polish=True, init='sobol',
                    maxfevals=2000
                )
                h_opt = rects_to_h(result.x)
                h_clipped = np.clip(h_opt, 0, None)
                true_val = ratio_from_h(h_clipped)
                update_best(h_clipped, true_val)
            except Exception:
                pass
    
    # Phase 3: Direct optimization with soft-max surrogate (from inspiration 1)
    if time.time() < deadline - 15.0:
        beta = 200.0
        
        def obj_direct_smooth(h):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * W) ** 2
            f_fine = h[CELL]
            F = np.fft.rfft(f_fine, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return soft_max / integral_sq
        
        bounds = [(0.0, None)] * N
        try:
            res_direct = opt.minimize(
                obj_direct_smooth, best_h, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = ratio_from_h(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5.0:
        current_h = best_h.copy()
        current_val = best_val
        step_size = 0.05
        
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            improved = False
            indices = rng.choice(N, size=min(100, N), replace=False)
            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                original = current_h[idx]
                current_h[idx] = max(0, original + step_size)
                val_plus = ratio_from_h(current_h)
                if val_plus < current_val:
                    current_val = val_plus
                    improved = True
                else:
                    current_h[idx] = max(0, original - step_size)
                    val_minus = ratio_from_h(current_h)
                    if val_minus < current_val:
                        current_val = val_minus
                        improved = True
                    else:
                        current_h[idx] = original
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
            update_best(current_h, current_val)
    
    return list(best_h)
