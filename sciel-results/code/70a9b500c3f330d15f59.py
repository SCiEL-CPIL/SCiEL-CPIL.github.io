import numpy as np
import time
from scipy.optimize import minimize

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * DELTA

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / DELTA).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()


def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals using fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    return conv


def compute_score(h):
    """Compute the normalized max autoconvolution score (L1-based normalization)."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    
    autoconv = compute_autoconv(h)
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


def params_to_h_phase(params):
    """
    Convert phase-coded signal parameters to step heights.
    The step function is |sum_k a_k * exp(i*(phi_k + f_k * x))|^2, smoothed.
    params: for each component k: [a_k_raw, f_k, phi_k_raw]
    a_k_raw -> softplus amplitude, phi_k_raw -> phase in [0, 2pi)
    """
    K = len(params) // 3
    h = np.zeros(N)
    for i in range(K):
        a_raw = params[3 * i]
        f_k = params[3 * i + 1]
        phi_raw = params[3 * i + 2]
        
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        phi = np.arctan2(np.sin(phi_raw), np.cos(phi_raw))  # wrap to [-pi, pi]
        
        h += a * np.cos(BIN_CENTERS * f_k + phi)
    
    # Square to get non-negative envelope
    h = h ** 2
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = compute_score(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(42)
    
    # === Phase 1: Gaussian superposition with L-BFGS-B ===
    K_options = [3, 5, 7]
    
    for K in K_options:
        if time.time() > deadline - 40.0:
            break
        
        n_starts = 15
        for start in range(n_starts):
            if time.time() > deadline - 30.0:
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
            else:
                # Random smooth shapes
                for i in range(K):
                    p0[3 * i] = float(rng.uniform(-2, 1))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.3)))
            
            def objective_gauss(params):
                h = params_to_h_gaussian(params)
                val = compute_score(h)
                if not np.isfinite(val):
                    return 1e10
                return val
            
            try:
                res = minimize(
                    objective_gauss,
                    p0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.success or res.fun < best_val:
                    h_candidate = params_to_h_gaussian(res.x)
                    val = compute_score(h_candidate)
                    update_best(h_candidate, val)
            except Exception:
                pass
    
    # === Phase 2: Phase-coded signal optimization ===
    # Represent h(x) = |sum_k a_k * exp(i*(phi_k + f_k * x))|^2
    # This exploits low autocorrelation properties of pseudo-random sequences
    
    K_phase_options = [2, 3, 4]
    
    for K_phase in K_phase_options:
        if time.time() > deadline - 35.0:
            break
        
        n_starts = 10
        for start in range(n_starts):
            if time.time() > deadline - 25.0:
                break
            
            p0 = np.zeros(K_phase * 3)
            for i in range(K_phase):
                p0[3 * i] = float(rng.uniform(-1, 1))  # amplitude raw
                p0[3 * i + 1] = float(rng.uniform(5, 50))  # frequency
                p0[3 * i + 2] = float(rng.uniform(0, 2 * np.pi))  # phase
            
            def objective_phase(params):
                h = params_to_h_phase(params)
                val = compute_score(h)
                if not np.isfinite(val):
                    return 1e10
                return val
            
            try:
                res = minimize(
                    objective_phase,
                    p0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.success or res.fun < best_val:
                    h_candidate = params_to_h_phase(res.x)
                    val = compute_score(h_candidate)
                    update_best(h_candidate, val)
            except Exception:
                pass
    
    # === Phase 3: Direct bin-level optimization with smooth surrogate ===
    if time.time() < deadline - 25.0:
        bounds = [(0.0, None)] * N
        
        def smooth_objective(h, beta=300.0):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * DELTA) ** 2
            
            f_fine = h[CELL]
            F = np.fft.rfft(f_fine, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
            conv *= DX
            
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            # Soft max via log-sum-exp
            soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return soft_max / integral_sq
        
        try:
            res_direct = minimize(
                smooth_objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = compute_score(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # === Phase 4: Simulated annealing on bin-level perturbations ===
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(777)
        T = 0.05
        T_min = 0.001
        cooling = 0.999
        
        for it in range(5000):
            if time.time() > deadline - 2.0:
                break
            
            # Random perturbation
            n_perturb = int(rng2.integers(3, 15))
            idx = rng2.choice(N, size=n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += rng2.normal(0, 0.03, size=n_perturb)
            h_try = np.clip(h_try, 0, None)
            
            val_try = compute_score(h_try)
            
            # Accept if better or with probability exp(-(val_try - best_val)/T)
            if val_try < best_val:
                update_best(h_try, val_try)
            else:
                delta_val = val_try - best_val
                if delta_val < 0.1:  # Only consider small increases
                    accept_prob = np.exp(-delta_val / T)
                    if rng2.random() < accept_prob:
                        update_best(h_try, val_try)
            
            T = max(T_min, T * cooling)
    
    # === Phase 5: Final high-precision polish ===
    if time.time() < deadline - 3.0:
        bounds = [(0.0, None)] * N
        
        def tight_objective(h):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * DELTA) ** 2
            
            f_fine = h[CELL]
            F = np.fft.rfft(f_fine, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
            conv *= DX
            
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            return m / integral_sq
        
        try:
            res_final = minimize(
                tight_objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_candidate = np.clip(res_final.x, 0.0, None)
            val = compute_score(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    return list(best_h)
