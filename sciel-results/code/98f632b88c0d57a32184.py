"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac


def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))


def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta


def _tanh_relaxation_objective(z, beta=50.0):
    """
    Objective for tanh relaxation: minimize soft_max_sidelobe(tanh(z))
    Returns (value, gradient_wrt_z)
    """
    h = np.tanh(z)
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. h
    weights = beta * exp_shifted / sum_exp
    grad_h = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        if c_k == 0:
            continue
        grad_h[0:N-k] += c_k * h[k:N]
        grad_h[k:N] += c_k * h[0:N-k]
    
    # Chain rule: d/dz = d/dh * dh/dz = grad_h * (1 - tanh(z)^2)
    sech2 = 1.0 - h**2
    grad_z = grad_h * sech2
    
    return obj, grad_z


def _fast_flip_eval(h, i):
    """
    Compute new max sidelobe after flipping h[i] without full FFT.
    Uses incremental update of autocorrelation.
    """
    # Precompute ac once per call is not possible here, so we do full FFT
    # But we can try a smarter approach: only recompute affected lags
    # Actually, for simplicity and speed, let's just do full FFT
    # The key optimization is in the search strategy, not the eval
    ac = _compute_autocorr(h)
    return _max_sidelobe(ac)


def _multi_flip_local_search(h, max_iterations=5000, time_limit=5.0, n_flips=1):
    """
    Local search that tries flipping n_flips bits at a time.
    Uses FFT for evaluation.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping n_flips random bits
        indices = np.random.choice(N, size=n_flips, replace=False)
        old_vals = h[indices].copy()
        h[indices] = -h[indices]
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            h[indices] = old_vals
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Tanh Relaxation with L-BFGS-B
    # Use multiple random initializations in z-space (before tanh)
    num_starts = 8
    betas = [20.0, 50.0, 100.0]
    
    for i in range(num_starts):
        if time.time() >= deadline - 40.0:
            break
        
        rng = np.random.default_rng(1000 + i)
        # Initialize z so that tanh(z) is near ±1 but not exactly
        # This allows gradient flow
        z0 = rng.normal(0, 1, size=N) * 2.0
        
        beta = betas[i % len(betas)]
        
        def obj_grad(z, beta=beta):
            return _tanh_relaxation_objective(z, beta)
        
        try:
            res = minimize(obj_grad, z0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30})
        except Exception:
            continue
        
        # Convert back to binary
        h_bin = np.sign(np.tanh(res.x))
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = rng.choice([-1.0, 1.0], size=np.sum(zero_mask))
        
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
        
        # Quick 1-bit local search
        if time.time() < deadline - 30.0:
            h_ref, val_ref = _multi_flip_local_search(h_bin.copy(), max_iterations=2000, time_limit=2.0, n_flips=1)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 2: 2-bit flip local search for better escape
    if best_h is not None and time.time() < deadline - 20.0:
        h_ref, val_ref = _multi_flip_local_search(best_h.copy(), max_iterations=5000, time_limit=8.0, n_flips=2)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 3: 3-bit flip local search
    if best_h is not None and time.time() < deadline - 10.0:
        h_ref, val_ref = _multi_flip_local_search(best_h.copy(), max_iterations=3000, time_limit=5.0, n_flips=3)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 4: Final 1-bit polish
    if best_h is not None and time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 1.0:
            h_ref, val_ref = _multi_flip_local_search(best_h.copy(), max_iterations=10000, time_limit=remaining, n_flips=1)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
