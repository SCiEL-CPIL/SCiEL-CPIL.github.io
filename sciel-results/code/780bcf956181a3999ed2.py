import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # len(g) = 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    def _build_y(g):
        y = np.zeros(M + 2)
        y[1:-1] = g
        return y

    def compute_value(h):
        """Exact evaluator metric for given heights."""
        h = np.asarray(h, dtype=np.float64)
        if len(h) != N:
            return 0.0
        if not np.all(np.isfinite(h)):
            return 0.0
        h = np.maximum(0.0, h)
        g = np.convolve(h, h)
        m = len(g)
        y = _build_y(g)
        a = y[:m + 1]
        b = y[1:m + 2]
        l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        if norm_1 * norm_inf == 0:
            return 0.0
        return l2 / (norm_1 * norm_inf)

    def _smooth_value(h, beta=50.0):
        """
        Smooth surrogate for the value function.
        Uses Log-Sum-Exp for norm_inf to provide differentiability.
        Returns negative value for minimization.
        """
        h = np.maximum(0.0, h)
        g = np.convolve(h, h)
        m = len(g)
        y = _build_y(g)
        a = y[:m + 1]
        b = y[1:m + 2]
        l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
        
        # Smooth max: (1/beta) * log(sum(exp(beta * g)))
        g_max = np.max(g)
        if g_max <= 0:
            return 0.0
        lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
        
        if norm_1 <= 0 or lse <= 0:
            return 0.0
            
        val = l2 / (norm_1 * lse)
        return -val

    def _cosine_basis_coeffs_to_heights(coeffs, amplitudes):
        """
        Convert low-rank cosine basis representation to heights.
        coeffs: array of shape (K,) - frequencies (integers)
        amplitudes: array of shape (K,) - amplitudes
        Returns: h of length N
        """
        x = np.arange(N)
        h = np.zeros(N)
        for c, a in zip(coeffs, amplitudes):
            h += a * np.cos(2 * np.pi * c * x / N)
        return np.maximum(0.0, h)

    def _optimize_low_rank(start_coeffs, start_amplitudes, beta, maxiter=200):
        """
        Optimize the low-rank cosine representation.
        Returns (best_h, best_val).
        """
        K = len(start_coeffs)
        
        def objective(params):
            # params = [c_1, ..., c_K, a_1, ..., a_K]
            coeffs = params[:K]
            amps = params[K:]
            
            # Ensure frequencies are positive integers (clamp to [1, 20])
            coeffs_int = np.clip(np.round(coeffs), 1, 20).astype(int)
            h = _cosine_basis_coeffs_to_heights(coeffs_int, amps)
            
            return _smooth_value(h, beta)
        
        x0 = np.concatenate([start_coeffs, start_amplitudes])
        
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            coeffs_opt = np.clip(np.round(res.x[:K]), 1, 20).astype(int)
            amps_opt = res.x[K:]
            h_opt = _cosine_basis_coeffs_to_heights(coeffs_opt, amps_opt)
            return h_opt, compute_value(h_opt)
        except Exception:
            return None, 0.0

    def _generate_structured_starts():
        """Generate diverse structured starting profiles."""
        x = np.linspace(0, 1, N)
        starts = []
        
        # Uniform
        starts.append(np.ones(N))
        
        # Triangular
        starts.append(1 - 2 * np.abs(x - 0.5))
        
        # Parabolic
        starts.append(np.maximum(0, 1 - 4 * (x - 0.5)**2))
        
        # Cosine bump
        starts.append(0.5 * (1 + np.cos(2 * np.pi * x)))
        
        # Gaussian center
        starts.append(np.exp(-((x - 0.5)**2) / (2 * 0.15**2)))
        
        # Two peaks
        p1 = np.exp(-((x - 0.25)**2) / (2 * 0.1**2))
        p2 = np.exp(-((x - 0.75)**2) / (2 * 0.1**2))
        starts.append(p1 + p2)
        
        # Semicircle
        starts.append(np.sqrt(np.maximum(0, 1 - (2*x - 1)**2)))
        
        # Power law
        starts.append(np.maximum(0, 1 - x)**0.5)
        
        # Random smooth profiles using low-frequency cosine
        rng = np.random.RandomState(42)
        for _ in range(10):
            K = rng.randint(2, 5)
            coeffs = rng.randint(1, 6, size=K)
            amps = rng.uniform(0.5, 1.5, size=K)
            h = _cosine_basis_coeffs_to_heights(coeffs, amps)
            if np.max(h) > 0:
                h = h / np.max(h)
            starts.append(h)
        
        return starts

    # Initialize best
    best_h = np.ones(N)
    best_val = compute_value(best_h)

    def track(h):
        nonlocal best_h, best_val
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Generate starts
    starts = _generate_structured_starts()
    for s in starts:
        track(s)

    # Phase 1: Low-rank cosine optimization
    # Try different numbers of basis functions
    rng = np.random.RandomState(123)
    
    for K in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Generate random initializations for low-rank representation
        for trial in range(5):
            if time.time() > deadline:
                break
            
            start_coeffs = rng.uniform(1, 10, size=K)
            start_amplitudes = rng.uniform(0.5, 2.0, size=K)
            
            for beta in [20.0, 50.0, 100.0]:
                if time.time() > deadline:
                    break
                
                h_opt, val_opt = _optimize_low_rank(start_coeffs, start_amplitudes, beta, maxiter=150)
                if h_opt is not None:
                    track(h_opt)

    # Phase 2: Full-space L-BFGS-B optimization from best found so far
    bounds = [(0, None)] * N
    
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline:
            break
        
        # Try from best_h and some perturbed versions
        perturbations = [best_h.copy()]
        for _ in range(5):
            pert = best_h + rng.normal(0, 0.1, N)
            perturbations.append(np.maximum(0, pert))
        
        for x0 in perturbations:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    lambda x: _smooth_value(x, beta),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                track(res.x)
            except Exception:
                pass

    # Phase 3: Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_value(x, 200.0),
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            track(res.x)
        except Exception:
            pass

    # Final cleanup
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    return [float(h) for h in best_h]
# EVOLVE-BLOCK-END
