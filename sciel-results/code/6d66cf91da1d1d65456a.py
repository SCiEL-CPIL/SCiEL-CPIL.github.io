import numpy as np
import time
from scipy.optimize import minimize

N = 400

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
    # Stable log-sum-exp: max_val + log(sum(exp(x - max_val)))
    # We want a smooth max, so we use:
    # soft_max(x) = (1/beta) * log(sum(exp(beta * x)))
    # = (1/beta) * log(sum(exp(beta * (x - max_val) + beta * max_val)))
    # = max_val + (1/beta) * log(sum(exp(beta * (x - max_val))))
    
    shifted = beta * (sidelobes - max_val)
    # Clip to avoid overflow in exp (though shifted <= 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe.
    Gradient: Analytic.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft max value
    max_val = np.max(sidelobes_abs)
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient of soft max w.r.t. sidelobes
    # d/dx_i (soft_max) = sum_k w_k * d|ac[k]|/dx_i
    # where w_k = exp(beta * (ac[k] - max)) / sum_exp = beta * exp_shifted[k] / sum_exp
    # Note: d|ac[k]|/dx_i = sign(ac[k]) * d(ac[k])/dx_i
    # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k == 0:
            continue
        
        # ac[k] = sum_j h[j] h[j+k]
        # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
        # sign(ac[k]) * d(ac[k])/dx_i
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        # Vectorized update
        # Indices i such that i+k < N => i < N-k
        # grad[0:N-k] += c_k * h[k:N]
        # Indices i such that i-k >= 0 => i >= k
        # grad[k:N] += c_k * h[0:N-k]
        
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _spectral_init(seed):
    """
    Initialize from spectral phase retrieval: create a flat magnitude spectrum,
    take inverse FFT, and threshold to ±1.
    """
    rng = np.random.RandomState(seed)
    
    # Target: flat magnitude spectrum (all sidelobes zero in ideal case)
    # ac[0] = N (DC), ac[k] = 0 for k>0
    # |F|^2 = N * delta[k]
    # F[k] = sqrt(N) * exp(i * phi[k]) for all k
    
    # Create random phases
    n_fft = 2 * N
    phases = 2 * np.pi * rng.rand(n_fft // 2 + 1)
    
    # Build complex spectrum with flat magnitude
    F = np.sqrt(N) * np.exp(1j * phases)
    
    # Ensure Hermitian symmetry for real signal
    F_full = np.zeros(n_fft, dtype=complex)
    F_full[:n_fft//2+1] = F
    F_full[n_fft//2+1:] = np.conj(F_full[1:n_fft//2][::-1])
    
    # Inverse FFT
    h_continuous = np.fft.ifft(F_full).real
    
    # Threshold to ±1
    h = np.sign(h_continuous)
    # Handle zeros
    zero_mask = (h == 0)
    if np.any(zero_mask):
        h[zero_mask] = rng.choice([-1.0, 1.0], size=np.sum(zero_mask))
    
    return h

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using FFT for evaluation with targeted bit-flip heuristic.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_k = np.argmax(np.abs(ac[1:N])) + 1  # Track worst lag
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Strategy: try flipping bits that affect the worst lag most
        # ac[k] = sum_j h[j] h[j+k]
        # Flipping h[i] changes ac[k] by -2*h[i]*h[i+k] (if i+k < N) and -2*h[i]*h[i-k] (if i-k >= 0)
        
        # Target the worst lag
        k = best_k
        sign_k = np.sign(ac[k])
        
        # Find indices i where flipping would most reduce |ac[k]|
        # Delta = -2 * h[i] * (h[i+k] + h[i-k]) for valid i
        # We want sign(ac[k]) * delta < 0, i.e., sign_k * (-2 * h[i] * (h[i+k] + h[i-k])) < 0
        # => sign_k * h[i] * (h[i+k] + h[i-k]) > 0
        
        candidates = []
        
        # Indices where both i+k < N and i-k >= 0
        valid_start = k
        valid_end = N - k
        if valid_end > valid_start:
            i_range = np.arange(valid_start, valid_end)
            deltas = -2.0 * h[i_range] * (h[i_range + k] + h[i_range - k])
            # We want sign_k * delta < 0
            good_mask = (sign_k * deltas) < 0
            if np.any(good_mask):
                good_indices = i_range[good_mask]
                good_deltas = deltas[good_mask]
                # Sort by magnitude of improvement
                order = np.argsort(-np.abs(good_deltas))
                candidates = list(good_indices[order[:10]])
        
        # Also add some random candidates for diversity
        for _ in range(5):
            candidates.append(np.random.randint(N))
        
        # Try each candidate
        found_improvement = False
        for i in candidates:
            old_h_i = h[i]
            h[i] = -old_h_i
            
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            
            if new_val < current_val:
                current_val = new_val
                ac = ac_new
                best_k = np.argmax(np.abs(ac[1:N])) + 1
                found_improvement = True
                break
            else:
                # Revert
                h[i] = old_h_i
        
        if not found_improvement:
            # Stagnation: try a random flip
            i = np.random.randint(N)
            old_h_i = h[i]
            h[i] = -old_h_i
            
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            
            if new_val < current_val:
                current_val = new_val
                ac = ac_new
                best_k = np.argmax(np.abs(ac[1:N])) + 1
            else:
                h[i] = old_h_i
    
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Spectral phase retrieval initialization + local search
    num_spectral_starts = 20
    
    for i in range(num_spectral_starts):
        if time.time() >= deadline - 30.0:  # Reserve 30s for continuous + final polish
            break
        
        # Initialize from spectral phase retrieval
        h0 = _spectral_init(seed=42 + i * 137)
        
        # Quick local search
        h_ref, val_ref = _local_search(h0.copy(), max_iterations=2000, time_limit=2.0)
        
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref.copy()
    
    # Phase 2: Continuous Relaxation with Smooth Objective
    num_continuous_starts = 8
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 20.0:  # Reserve 20s for final polish
            break
            
        # Initialize x in [-1, 1]
        # Start near binary to help rounding
        if best_h is not None and i % 2 == 0:
            # Perturb best solution
            x0 = best_h.copy()
            noise_mask = np.random.rand(N) < 0.1
            x0[noise_mask] = -x0[noise_mask]
        else:
            x0 = np.random.choice([-1.0, 1.0], size=N)
        
        # Use a beta that increases over iterations or pick one
        beta = betas[i % len(betas)]
        
        # Define objective with fixed beta
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        # Use L-BFGS-B
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
        except Exception:
            continue
        
        # Round to binary
        h_bin = np.sign(res.x)
        # Handle zeros
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
    
    # Phase 3: Refine best solution with local search
    if best_h is not None and time.time() < deadline - 10.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=5000, time_limit=5.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 4: Final Discrete Search
    remaining_time = deadline - time.time()
    if remaining_time > 2.0 and best_h is not None:
        h_final, val_final = _local_search(best_h.copy(), max_iterations=50000, time_limit=remaining_time - 1.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
