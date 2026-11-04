import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * DELTA

# Two grids: coarse for fast optimization, fine for final scoring
COARSE_N = 3000
FINE_N = 12000

X_COARSE = np.linspace(-0.25, 0.25, COARSE_N, endpoint=False)
DX_COARSE = 0.5 / COARSE_N
CELL_COARSE = np.clip(((X_COARSE + 0.25) / DELTA).astype(np.int64), 0, N - 1)
N_FFT_COARSE = 1 << (2 * COARSE_N - 1).bit_length()

X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX_FINE = 0.5 / FINE_N
CELL_FINE = np.clip(((X_FINE + 0.25) / DELTA).astype(np.int64), 0, N - 1)
N_FFT_FINE = 1 << (2 * FINE_N - 1).bit_length()


def compute_autoconv(h, use_fine=True):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    if use_fine:
        f_grid = h[CELL_FINE]
        n_fft = N_FFT_FINE
        dx = DX_FINE
        n_conv = 2 * FINE_N - 1
    else:
        f_grid = h[CELL_COARSE]
        n_fft = N_FFT_COARSE
        dx = DX_COARSE
        n_conv = 2 * COARSE_N - 1
    
    F = np.fft.rfft(f_grid, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:n_conv]
    conv *= dx
    return conv


def compute_score(h, use_fine=True):
    """Compute the normalized max autoconvolution score (L1-based normalization)."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    
    autoconv = compute_autoconv(h, use_fine=use_fine)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac / (integral ** 2)
    return score


def params_to_h_gaussian(params):
    """Convert Gaussian superposition parameters to step heights."""
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


def params_to_h_fourier(params):
    """
    Convert Fourier-mode parameters to step heights.
    params: [a0, b0, a1, b1, ..., aK, bK, bias]
    h[j] = bias + sum_k (a_k * cos(2*pi*k*x_j/L) + b_k * sin(2*pi*k*x_j/L))
    where x_j are bin centers, L = 0.5
    Use softplus on final result to ensure nonnegativity.
    """
    n_params = len(params)
    # Last param is bias
    bias_raw = params[-1]
    bias = np.log1p(np.exp(np.clip(bias_raw, -50, 50))) * 0.1
    
    K = (n_params - 1) // 2
    x = BIN_CENTERS
    L = 0.5
    h = np.zeros(N)
    for k in range(1, K + 1):
        a_k = params[2 * (k - 1)]
        b_k = params[2 * (k - 1) + 1]
        h += a_k * np.cos(2 * np.pi * k * x / L) + b_k * np.sin(2 * np.pi * k * x / L)
    
    h += bias
    # Softplus to ensure nonnegativity
    h = np.log1p(np.exp(np.clip(h, -50, 50)))
    return h


def params_to_h_block(params):
    """
    Convert block structure parameters to step heights.
    params: [b1_start, b1_end, w1, b2_start, b2_end, w2, ...]
    Each block is an indicator function on [start, end] with weight w.
    Blocks are disjoint and symmetric pairs are encouraged by initialization.
    """
    K = len(params) // 3
    h = np.zeros(N)
    for i in range(K):
        s_raw = params[3 * i]
        e_raw = params[3 * i + 1]
        w_raw = params[3 * i + 2]
        
        # Convert raw params to bin indices [0, N-1]
        s_idx = int(np.clip(np.round(s_raw), 0, N - 1))
        e_idx = int(np.clip(np.round(e_raw), 0, N - 1))
        
        if e_idx >= s_idx:
            # Weight is non-negative via softplus
            w = np.log1p(np.exp(np.clip(w_raw, -50, 50)))
            h[s_idx:e_idx + 1] += w
    
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = compute_score(best_h)
    
    def update_best(h, val, use_fine=True):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(42)
    
    # === Phase 1: Gaussian superposition with L-BFGS-B (coarse grid) ===
    K_options = [3, 5, 7, 9]
    
    for K in K_options:
        if time.time() > deadline - 60.0:
            break
        
        n_starts = 12
        for start in range(n_starts):
            if time.time() > deadline - 55.0:
                break
            
            # Generate diverse initializations
            p0 = np.zeros(K * 3)
            if start == 0:
                # Uniform-ish spread
                for i in range(K):
                    p0[3 * i] = np.log(1.0 / K)
                    p0[3 * i + 1] = float(rng.uniform(-0.2, 0.2))
                    p0[3 * i + 2] = np.log(0.1 + float(rng.uniform(0, 0.2)))
            elif start == 1:
                # Single dominant bump
                p0[0] = np.log(1.0)
                p0[1] = 0.0
                p0[2] = np.log(0.15)
                for i in range(1, K):
                    p0[3 * i] = np.log(0.01)
                    p0[3 * i + 1] = float(rng.uniform(-0.2, 0.2))
                    p0[3 * i + 2] = np.log(0.05)
            elif start == 2:
                # Asymmetric pair
                x1 = float(rng.uniform(-0.2, -0.05))
                x2 = float(rng.uniform(0.05, 0.2))
                p0[0] = float(rng.uniform(0, 1))
                p0[1] = x1
                p0[2] = float(rng.uniform(np.log(0.1), np.log(0.2)))
                if K >= 2:
                    p0[3] = float(rng.uniform(0, 1))
                    p0[4] = x2
                    p0[5] = float(rng.uniform(np.log(0.1), np.log(0.2)))
                for i in range(2, K):
                    p0[3 * i] = float(rng.uniform(-3, 0))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.2)))
            elif start == 3:
                # Symmetric pair
                x_off = float(rng.uniform(0.05, 0.2))
                p0[0] = float(rng.uniform(0, 1))
                p0[1] = -x_off
                p0[2] = float(rng.uniform(np.log(0.1), np.log(0.2)))
                if K >= 2:
                    p0[3] = float(rng.uniform(0, 1))
                    p0[4] = x_off
                    p0[5] = float(rng.uniform(np.log(0.1), np.log(0.2)))
                for i in range(2, K):
                    p0[3 * i] = float(rng.uniform(-3, 0))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.2)))
            else:
                # Random smooth shapes
                for i in range(K):
                    p0[3 * i] = float(rng.uniform(-2, 1))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.3)))
            
            def objective_gauss_coarse(params):
                h = params_to_h_gaussian(params)
                val = compute_score(h, use_fine=False)
                if not np.isfinite(val):
                    return 1e10
                return val
            
            try:
                res = minimize(
                    objective_gauss_coarse,
                    p0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.success or res.fun < best_val:
                    h_candidate = params_to_h_gaussian(res.x)
                    val = compute_score(h_candidate, use_fine=True)
                    update_best(h_candidate, val)
            except Exception:
                pass
    
    # === Phase 1b: Fourier-mode parameterization with L-BFGS-B (coarse grid) ===
    if time.time() < deadline - 50.0:
        K_fourier_options = [5, 8, 12]
        
        for K_f in K_fourier_options:
            if time.time() > deadline - 45.0:
                break
            
            n_params = 2 * K_f + 1  # a_k, b_k for k=1..K_f, plus bias
            n_starts = 8
            
            for start in range(n_starts):
                if time.time() > deadline - 40.0:
                    break
                
                p0 = np.zeros(n_params)
                if start == 0:
                    # Low-frequency dominant
                    for k in range(1, K_f + 1):
                        amp = 0.5 / k
                        p0[2 * (k - 1)] = amp
                        p0[2 * (k - 1) + 1] = float(rng.uniform(-amp, amp))
                    p0[-1] = np.log(0.5)  # bias
                else:
                    # Random Fourier coefficients
                    for k in range(1, K_f + 1):
                        amp = float(rng.uniform(0.1, 1.0)) / np.sqrt(k)
                        p0[2 * (k - 1)] = float(rng.uniform(-amp, amp))
                        p0[2 * (k - 1) + 1] = float(rng.uniform(-amp, amp))
                    p0[-1] = float(rng.uniform(np.log(0.1), np.log(1.0)))
                
                def objective_fourier_coarse(params):
                    h = params_to_h_fourier(params)
                    val = compute_score(h, use_fine=False)
                    if not np.isfinite(val):
                        return 1e10
                    return val
                
                try:
                    res = minimize(
                        objective_fourier_coarse,
                        p0,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    if res.success or res.fun < best_val:
                        h_candidate = params_to_h_fourier(res.x)
                        val = compute_score(h_candidate, use_fine=True)
                        update_best(h_candidate, val)
                except Exception:
                    pass
    
    # === Phase 2: Block structure optimization with Differential Evolution (coarse grid) ===
    n_blocks_options = [2, 3, 4]
    
    for n_blocks in n_blocks_options:
        if time.time() > deadline - 35.0:
            break
        
        n_params = 3 * n_blocks
        bounds = []
        for _ in range(n_blocks):
            bounds.append((0.0, float(N - 1)))   # start bin
            bounds.append((0.0, float(N - 1)))   # end bin
            bounds.append((-5.0, 5.0))           # weight raw
        
        def objective_block_coarse(params):
            h = params_to_h_block(params)
            val = compute_score(h, use_fine=False)
            if not np.isfinite(val):
                return 1e10
            return val
        
        # Initial population with symmetric block structures
        init_pop = []
        for _ in range(15):
            p = np.zeros(n_params)
            for i in range(n_blocks):
                center = N // 2
                offset = int(rng.uniform(50, N // 2 - 50))
                width = int(rng.uniform(20, 100))
                
                if i % 2 == 0:
                    s = max(0, center - offset - width // 2)
                    e = min(N - 1, center - offset + width // 2)
                else:
                    s = max(0, center + offset - width // 2)
                    e = min(N - 1, center + offset + width // 2)
                
                p[3 * i] = float(s)
                p[3 * i + 1] = float(e)
                p[3 * i + 2] = float(rng.uniform(-1, 1))
            
            init_pop.append(p)
        
        try:
            result = differential_evolution(
                objective_block_coarse,
                bounds,
                maxiter=30,
                popsize=12,
                seed=42,
                polish=True,
                tol=1e-4,
                maxfev=800,
                init=np.array(init_pop)
            )
            
            if result.fun < best_val:
                h_candidate = params_to_h_block(result.x)
                val = compute_score(h_candidate, use_fine=True)
                update_best(h_candidate, val)
                
        except Exception:
            pass
    
    # === Phase 3: Direct bin-level optimization with L-BFGS-B (coarse grid, soft-max) ===
    if time.time() < deadline - 25.0:
        bounds = [(0.0, None)] * N
        
        def smooth_objective_coarse(h, beta=100.0):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * DELTA) ** 2
            
            f_grid = h[CELL_COARSE]
            F = np.fft.rfft(f_grid, N_FFT_COARSE)
            conv = np.fft.irfft(F * F, N_FFT_COARSE)[:2 * COARSE_N - 1]
            conv *= DX_COARSE
            
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            # Soft-max for smoothness
            soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return soft_max / integral_sq
        
        try:
            res_direct = minimize(
                smooth_objective_coarse,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = compute_score(h_candidate, use_fine=True)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # === Phase 4: Fine-grid local polish with L-BFGS-B ===
    if time.time() < deadline - 10.0:
        bounds = [(0.0, None)] * N
        
        def smooth_objective_fine(h, beta=500.0):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * DELTA) ** 2
            
            f_grid = h[CELL_FINE]
            F = np.fft.rfft(f_grid, N_FFT_FINE)
            conv = np.fft.irfft(F * F, N_FFT_FINE)[:2 * FINE_N - 1]
            conv *= DX_FINE
            
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return soft_max / integral_sq
        
        try:
            res_fine = minimize(
                smooth_objective_fine,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_candidate = np.clip(res_fine.x, 0.0, None)
            val = compute_score(h_candidate, use_fine=True)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # === Phase 5: Intensive local search polish (coarse grid for speed) ===
    if time.time() < deadline - 3.0:
        rng2 = np.random.default_rng(777)
        n_iterations = 5000
        for it in range(n_iterations):
            if time.time() > deadline - 1.0:
                break
            
            # Adaptive perturbation size
            scale = 0.05 * (1.0 - it / n_iterations) + 0.005
            n_perturb = int(rng2.integers(3, 15))
            idx = rng2.choice(N, size=n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += rng2.normal(0, scale, size=n_perturb)
            h_try = np.clip(h_try, 0, None)
            
            val = compute_score(h_try, use_fine=False)
            if val < best_val * 1.001:  # Only accept if close or better
                # Verify with fine grid
                val_fine = compute_score(h_try, use_fine=True)
                update_best(h_try, val_fine)
    
    return list(best_h)
