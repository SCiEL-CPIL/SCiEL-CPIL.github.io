import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
W = 1.0 / 1200.0

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute bin centers for parametric model evaluation
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Normalization: max((f*f)(t)) / (integral f)^2
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Map step heights to fine grid
    f_fine = h[CELL]
    
    # FFT-based autoconvolution
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral
    
    max_conv = np.max(conv)
    
    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def params_to_h(params):
    """
    Convert Gaussian superposition parameters to step function heights.
    params: [a1, x1, s1, a2, x2, s2, ..., aK, xK, sK]
    """
    K = len(params) // 3
    h = np.zeros(N)
    
    for i in range(K):
        a_raw = params[3 * i]
        x_c = params[3 * i + 1]
        s_raw = params[3 * i + 2]
        
        # Softplus for positive amplitude
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        # Exp for positive width
        s = np.exp(np.clip(s_raw, -10, 10))
        s = max(s, 1e-4)
        # Clip center to domain
        x_c = np.clip(x_c, -0.25, 0.25)
        
        h += a * np.exp(-0.5 * ((BIN_CENTERS - x_c) / s) ** 2)
    
    return h

def objective(params):
    h = params_to_h(params)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Number of Gaussian basis functions
    K = 8
    
    np.random.seed(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform-ish spread
    p0 = np.zeros(K * 3)
    for i in range(K):
        p0[3 * i] = np.log(1.0 / K)
        p0[3 * i + 1] = np.random.uniform(-0.2, 0.2)
        p0[3 * i + 2] = np.log(0.1 + np.random.rand() * 0.2)
    inits.append(p0)
    
    # 2. Single dominant bump at various positions
    for shift in [-0.15, -0.075, 0.0, 0.075, 0.15]:
        p0 = np.zeros(K * 3)
        p0[0] = np.log(1.0)
        p0[1] = shift
        p0[2] = np.log(0.15)
        for i in range(1, K):
            p0[3 * i] = np.log(0.01)
            p0[3 * i + 1] = np.random.uniform(-0.2, 0.2)
            p0[3 * i + 2] = np.log(0.05)
        inits.append(p0)
    
    # 3. Random smooth shapes
    for _ in range(15):
        p0 = np.zeros(K * 3)
        for i in range(K):
            p0[3 * i] = np.random.uniform(-2, 1)
            p0[3 * i + 1] = np.random.uniform(-0.25, 0.25)
            p0[3 * i + 2] = np.random.uniform(np.log(0.05), np.log(0.3))
        inits.append(p0)
    
    # 4. Asymmetric pairs
    for _ in range(8):
        p0 = np.zeros(K * 3)
        x1 = np.random.uniform(-0.2, -0.05)
        x2 = np.random.uniform(0.05, 0.2)
        p0[0] = np.random.uniform(0, 1)
        p0[1] = x1
        p0[2] = np.random.uniform(np.log(0.1), np.log(0.2))
        p0[3] = np.random.uniform(0, 1)
        p0[4] = x2
        p0[5] = np.random.uniform(np.log(0.1), np.log(0.2))
        for i in range(2, K):
            p0[3 * i] = np.random.uniform(-3, 0)
            p0[3 * i + 1] = np.random.uniform(-0.25, 0.25)
            p0[3 * i + 2] = np.random.uniform(np.log(0.05), np.log(0.2))
        inits.append(p0)
    
    # 5. Symmetric patterns (potential for lower autoconv peak)
    for _ in range(5):
        p0 = np.zeros(K * 3)
        # Create symmetric pairs
        for i in range(0, K, 2):
            if i < K - 1:
                x_c = np.random.uniform(0.05, 0.2)
                p0[3 * i] = np.random.uniform(0, 1)
                p0[3 * i + 1] = -x_c
                p0[3 * i + 2] = np.random.uniform(np.log(0.1), np.log(0.2))
                p0[3 * (i + 1)] = p0[3 * i]
                p0[3 * (i + 1) + 1] = x_c
                p0[3 * (i + 1) + 2] = p0[3 * i + 2]
        inits.append(p0)
    
    # 6. Broad flat-top shapes
    for _ in range(3):
        p0 = np.zeros(K * 3)
        for i in range(K):
            p0[3 * i] = np.random.uniform(0, 0.5)
            p0[3 * i + 1] = np.random.uniform(-0.15, 0.15)
            p0[3 * i + 2] = np.random.uniform(np.log(0.15), np.log(0.3))
        inits.append(p0)
    
    # Optimize from each initialization
    for p0 in inits:
        if time.time() > deadline - 35.0:
            break
            
        try:
            res = minimize(
                objective,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_val:
                h_candidate = params_to_h(res.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Direct Optimization with Soft-Max Surrogate (from current program)
    if time.time() < deadline - 20.0:
        # We optimize h directly
        # Objective: SoftMax(conv) / (sum(h)*W)^2
        beta = 300.0 # Increased smoothness parameter for better gradient
        
        def obj_direct_smooth(h):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * W) ** 2
            
            f_fine = h[CELL]
            F = np.fft.rfft(f_fine, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
            conv *= DX
            
            # Soft Max
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            # log(sum(exp(conv - m))) + m
            soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return soft_max / integral_sq
            
        bounds = [(0.0, None)] * N
        try:
            res_direct = minimize(
                obj_direct_smooth,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = ratio_from_h(h_candidate) # Use true max for scoring
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
                
        except Exception:
            pass
            
    # SLSQP polish for high-precision local refinement
    if time.time() < deadline - 10.0:
        try:
            def obj_slsqp(h):
                h = np.clip(h, 0.0, None)
                s = h.sum()
                if s < 1e-15:
                    return 1e10
                integral_sq = (s * W) ** 2
                
                f_fine = h[CELL]
                F = np.fft.rfft(f_fine, N_FFT)
                conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
                conv *= DX
                
                # Use soft-max with higher beta for smoother landscape
                beta_slsqp = 500.0
                m = np.max(conv)
                if m < 1e-15:
                    return 1e10
                soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta_slsqp
                return soft_max / integral_sq
            
            # Equality constraint: fix integral to prevent trivial scaling
            integral_target = best_h.sum() * W
            constraints = [
                {'type': 'eq', 'fun': lambda h: h.sum() * W - integral_target}
            ]
            
            bounds_slsqp = [(0.0, None)] * N
            
            res_slsqp = minimize(
                obj_slsqp,
                best_h,
                method='SLSQP',
                bounds=bounds_slsqp,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            h_candidate = np.clip(res_slsqp.x, 0.0, None)
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
                
        except Exception:
            pass
    
    # Final local search polish (from inspiration 1)
    if time.time() < deadline - 5.0:
        # Gradient-free local search: random perturbations
        for it in range(3000):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb a random subset of bins
            n_perturb = np.random.randint(3, 15)
            idx = np.random.choice(N, n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(n_perturb) * 0.01
            h_try = np.clip(h_try, 0, None)
            
            val = ratio_from_h(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
    
    return list(best_h)
