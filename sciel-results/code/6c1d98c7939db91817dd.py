"""Optimized Erdos minimum overlap via spectral shaping with smooth optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlap_max(h):
    """Compute max overlap of h with shifted 1-h."""
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _soft_max(x, beta=50.0):
    """Smooth max approximation via log-sum-exp."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m))) / beta * 0  # placeholder
    # Actually: soft_max = (1/beta) * log(sum(exp(beta*x)))
    # Let me redo:
def _soft_max_v2(x, beta=50.0):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _overlaps(h):
    """Compute all N overlaps."""
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    ovs = np.zeros(N)
    for s in range(N):
        ovs[s] = np.dot(h, np.roll(one_minus_h, s))
    return ovs

def _objective(params):
    """Objective: soft max of overlaps, with penalty for constraint violations."""
    u = params  # zero-mean component, h = 0.5 + u
    h = 0.5 + u
    # Constraint penalties
    # h in [0, 1]: penalty for h < 0 or h > 1
    viol_low = np.sum(np.maximum(0, -h) ** 2)
    viol_high = np.sum(np.maximum(0, h - 1) ** 2)
    # Sum constraint: sum(h) = N/2 => sum(u) = 0
    viol_sum = (np.sum(u)) ** 2
    
    ovs = _overlaps(h)
    soft_max_val = _soft_max_v2(ovs, beta=30.0)
    
    return soft_max_val + 100.0 * (viol_low + viol_high) + 1000.0 * viol_sum

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Start from a good initialization: use spectral shaping
    # Initialize u with a pattern that has high minimum autocorrelation
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap_max(best_h)
    
    # Try several initializations
    initializations = []
    
    # 1. Zero (constant)
    initializations.append(np.zeros(N))
    
    # 2. Small random perturbation
    rng = np.random.RandomState(42)
    for trial in range(5):
        u_init = rng.randn(N) * 0.1
        u_init -= np.mean(u_init)
        initializations.append(u_init)
    
    # 3. Spectral initialization: try to create u with flat-ish power spectrum
    # but with some structure to maximize min autocorrelation
    for freq_pattern in range(3):
        U = np.zeros(N, dtype=complex)
        if freq_pattern == 0:
            # Low frequency dominant
            for k in range(1, N//4):
                U[k] = 1.0
                U[N-k] = 1.0
        elif freq_pattern == 1:
            # Spread across frequencies
            for k in range(1, N//2):
                U[k] = 1.0 / np.sqrt(k)
                U[N-k] = 1.0 / np.sqrt(k)
        else:
            # Mid frequencies
            for k in range(N//6, N//3):
                U[k] = 1.0
                U[N-k] = 1.0
        
        u_init = np.real(np.fft.ifft(U))
        u_init -= np.mean(u_init)
        # Scale to be within bounds
        max_u = np.max(np.abs(u_init))
        if max_u > 0:
            u_init = u_init / max_u * 0.45  # keep h in [0.05, 0.95]
        initializations.append(u_init)
    
    # Optimize each initialization
    for u_init in initializations:
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                _objective,
                u_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                u_opt = result.x
                h_opt = 0.5 + u_opt
                # Project to feasible region
                h_opt = np.clip(h_opt, 0, 1)
                # Adjust to satisfy sum constraint exactly
                h_opt = h_opt - (np.sum(h_opt) - N/2) / N
                h_opt = np.clip(h_opt, 0, 1)
                
                score = _compute_overlap_max(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    # Final polish: try local refinement with tighter tolerance
    if time.time() < deadline - 15.0:
        try:
            u_current = best_h - 0.5
            result = minimize(
                _objective,
                u_current,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            u_opt = result.x
            h_opt = 0.5 + u_opt
            h_opt = np.clip(h_opt, 0, 1)
            h_opt = h_opt - (np.sum(h_opt) - N/2) / N
            h_opt = np.clip(h_opt, 0, 1)
            
            score = _compute_overlap_max(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h - (np.sum(best_h) - N/2) / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
