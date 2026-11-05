"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, tau=0.01):
    return tau * np.log(np.sum(np.exp(x / tau)))

def _smooth_value(h, tau=0.01):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h)
    # Use soft max to smooth the max
    sm = tau * np.log(np.sum(np.exp(conv / tau)))
    return abs(2.0 * N * sm / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Try multiple initializations
    initializations = []
    
    # 1. Cosine-shaped window (Tukey-like)
    x = np.linspace(0, 1, N)
    initializations.append(0.5 + 0.5 * np.cos(2 * np.pi * x))
    
    # 2. Raised cosine
    initializations.append(0.5 * (1 + np.cos(2 * np.pi * x)))
    
    # 3. Sine-based oscillatory with positive mean
    initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * x))
    initializations.append(1.0 + 0.3 * np.sin(4 * np.pi * x))
    
    # 4. Two-lobe pattern
    initializations.append(1.0 + 0.5 * np.sin(np.pi * x))
    
    # 5. Uniform with small noise
    initializations.append(np.ones(N) + 0.01 * np.random.randn(N))
    
    # 6. Triangular
    tri = np.minimum(x, 1 - x) * 2
    initializations.append(0.5 + 0.5 * tri)
    
    # 7. Gaussian-like
    center = N / 2.0
    sigma = N / 4.0
    initializations.append(np.exp(-0.5 * ((x - 0.5) * N / sigma) ** 2))
    
    best_overall = best_h.copy()
    best_overall_val = best_val
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        # Ensure sum is not zero
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 1.0
        
        h0 = h0 / s0 * N  # normalize so sum = N
        
        # Try L-BFGS with smooth objective
        for tau in [0.05, 0.02, 0.01]:
            if time.time() > deadline:
                break
            
            def objective(h):
                return _smooth_value(h, tau)
            
            try:
                result = minimize(
                    objective,
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                continue
            
            h_opt = result.x
            val = _compute_value(h_opt)
            
            if val < best_overall_val:
                best_overall_val = val
                best_overall = h_opt.copy()
        
        # Also try direct evaluation of the initial point
        val0 = _compute_value(h0)
        if val0 < best_overall_val:
            best_overall_val = val0
            best_overall = h0.copy()
    
    # Final: try a few more random perturbations around best
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline:
                break
            h_pert = best_overall + 0.01 * np.random.randn(N)
            s_p = np.sum(h_pert)
            if abs(s_p) < 1e-10:
                continue
            h_pert = h_pert / s_p * N
            val_p = _compute_value(h_pert)
            if val_p < best_overall_val:
                best_overall_val = val_p
                best_overall = h_pert.copy()
    
    # Safety: ensure valid output
    s_final = np.sum(best_overall)
    if abs(s_final) < 1e-10:
        return np.ones(N)
    
    return best_overall
# EVOLVE-BLOCK-END
