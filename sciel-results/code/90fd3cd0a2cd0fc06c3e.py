"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard: 150s wall-clock limit, use 140s as safety margin
_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_exceeded():
    return _elapsed() > _DEADLINE

def _compute_overlap(h, N):
    """
    Compute the maximum overlap of h with shifted 1-h.
    
    The overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    
    Note: shift by 0 gives sum h[i](1-h[i]) = sum h[i] - sum h[i]^2 = N/2 - sum h[i]^2.
    """
    # h has length N, values in [0,1], sum = N/2
    # For each shift k in 0..N-1, compute sum_i h[i] * (1 - h[(i+k) % N])
    # Use FFT for efficiency
    
    # overlap[k] = sum_i h[i] * (1 - h[(i+k) % N])
    #            = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    #            = N/2 - circular_cross_correlation(h, h)[k]
    
    # circular cross-correlation of h with itself
    # corr[k] = sum_i h[i] * h[(i+k) % N]
    
    # Using FFT: corr = ifft(fft(h) * conj(fft(h)))
    H = np.fft.fft(h)
    psd = np.abs(H)**2
    corr = np.fft.ifft(psd).real
    # corr[k] = sum_i h[i] * h[(i+k) % N]
    
    overlaps = N/2.0 - corr
    # The maximum overlap is max(overlaps)
    return np.max(overlaps)

def _score(h):
    """Score is the upper bound on C5, which is the max overlap."""
    return _compute_overlap(h, N)

def construct():
    """
    Construct a sequence of N step heights in [0, 1] whose sum is exactly N/2,
    minimizing the maximum overlap of h with shifted 1-h.
    """
    global _start_time
    
    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    # The objective: maximize the minimum of (N/2 - corr[k]) over k
    # Equivalently, minimize the maximum of corr[k] over k
    # Or: minimize max_k (N/2 - corr[k])
    
    # Let's think about this more carefully.
    # overlap[k] = N/2 - corr[k]
    # We want to minimize max_k overlap[k]
    # This is equivalent to minimizing max_k (N/2 - corr[k])
    # Which is equivalent to maximizing min_k corr[k]
    
    # So we want to maximize the minimum circular autocorrelation.
    
    # Alternative formulation: introduce t such that t >= overlap[k] for all k, minimize t.
    # t >= N/2 - corr[k] for all k
    # t + corr[k] >= N/2 for all k
    # corr[k] >= N/2 - t for all k
    
    # Let's use a smooth surrogate: soft-min of corr[k] values
    # maximize min_k corr[k] ≈ maximize log-sum-exp(-corr[k]/tau) with small tau... 
    # Actually, min_k corr[k] can be approximated by -log-sum-exp(-corr[k]/tau)/N * tau... 
    
    # Better: use epigraph with smooth constraints.
    # Let f(t, h) = sum_k max(0, N/2 - corr_k(h) - t)^2 + lambda * t
    # Minimize over t and h.
    
    # Or simpler: minimize max_k overlap[k] directly using a smooth max approximation.
    # smooth_max(x, tau) = tau * log(sum exp(x_i / tau))
    
    def objective(params):
        if _time_exceeded():
            return 1e6
        h = params[:N]
        t = params[N]
        
        # Clip h to [0, 1] softly for gradient computation
        # Actually, let's handle bounds via the optimizer
        
        H = np.fft.fft(h)
        psd = np.abs(H)**2
        corr = np.fft.ifft(psd).real
        overlaps = N/2.0 - corr
        
        # Smooth max of overlaps
        tau = 0.1
        # For numerical stability
        max_ov = np.max(overlaps)
        exp_terms = np.exp((overlaps - max_ov) / tau)
        smooth_max = max_ov + tau * np.log(np.sum(exp_terms))
        
        # We want to minimize t, with constraint t >= smooth_max
        # Penalty if t < smooth_max
        penalty = 0.0
        if t < smooth_max:
            penalty = 1000.0 * (smooth_max - t)**2
        
        return t + penalty
    
    def gradients(params):
        if _time_exceeded():
            return np.zeros_like(params)
        h = params[:N]
        t = params[N]
        
        H = np.fft.fft(h)
        psd = np.abs(H)**2
        corr = np.fft.ifft(psd).real
        overlaps = N/2.0 - corr
        
        tau = 0.1
        max_ov = np.max(overlaps)
        exp_terms = np.exp((overlaps - max_ov) / tau)
        smooth_max = max_ov + tau * np.log(np.sum(exp_terms))
        
        # Gradient of smooth_max w.r.t. h
        # d smooth_max / dh = sum_k (exp(overlaps[k]/tau) / sum_j exp(overlaps[j]/tau)) * d overlaps[k] / dh
        # overlaps[k] = N/2 - corr[k]
        # d overlaps[k] / dh = -d corr[k] / dh
        
        # corr[k] = sum_i h[i] h[(i+k)%N]
        # d corr[k] / dh[j] = h[(j+k)%N] + h[(j-k)%N] (circular)
        
        # So d overlaps[k] / dh[j] = -h[(j+k)%N] - h[(j-k)%N]
        
        weights = exp_terms / np.sum(exp_terms)
        
        # grad_h[j] = sum_k weights[k] * (-h[(j+k)%N] - h[(j-k)%N])
        # = -sum_k weights[k] * h[(j+k)%N] - sum_k weights[k] * h[(j-k)%N]
        # = -(convolution of weights with h)[j] - (convolution of weights with h reversed)[j]
        
        # Using FFT:
        W = np.fft.fft(weights)
        Hh = np.fft.fft(h)
        # First term: sum_k weights[k] h[(j+k)%N] = ifft(fft(weights) * fft(h))
        conv1 = np.fft.ifft(W * Hh).real
        # Second term: sum_k weights[k] h[(j-k)%N] = ifft(fft(weights) * conj(fft(h)))
        conv2 = np.fft.ifft(W * np.conj(Hh)).real
        
        grad_h = -(conv1 + conv2)
        
        # Gradient w.r.t. t
        grad_t = 1.0
        if t < smooth_max:
            grad_t += 2000.0 * (t - smooth_max)
        
        return np.concatenate([grad_h, [grad_t]])
    
    # Initial parameters: h = 0.5, t = initial overlap estimate
    h_init = np.full(N, 0.5)
    t_init = _score(h_init)
    params0 = np.concatenate([h_init, [t_init]])
    
    # Bounds for h: [0, 1], t: [0, 1]
    bounds = [(0, 1)] * N + [(0, 1)]
    
    best_params = params0.copy()
    best_score = _score(h_init)
    
    # Use L-BFGS-B with gradient
    try:
        result = minimize(
            objective,
            params0,
            jac=gradients,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if result.success or result.fun < best_score:
            h_candidate = result.x[:N]
            # Ensure sum is exactly N/2
            h_candidate = np.clip(h_candidate, 0, 1)
            h_candidate = h_candidate / np.sum(h_candidate) * (N/2.0)
            # Re-clip and renormalize
            h_candidate = np.clip(h_candidate, 0, 1)
            h_candidate = h_candidate / np.sum(h_candidate) * (N/2.0)
            
            score_candidate = _score(h_candidate)
            if score_candidate < best_score:
                best_score = score_candidate
                best_params = np.concatenate([h_candidate, [score_candidate]])
    except Exception:
        pass
    
    if _time_exceeded():
        h_out = best_params[:N]
        h_out = np.clip(h_out, 0, 1)
        h_out = h_out / np.sum(h_out) * (N/2.0)
        return h_out
    
    # Try a few different initializations
    for trial in range(5):
        if _time_exceeded():
            break
            
        # Random initialization with sum constraint
        h_rand = np.random.dirichlet(np.ones(N)) * N
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = h_rand / np.sum(h_rand) * (N/2.0)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = h_rand / np.sum(h_rand) * (N/2.0)
        
        t_rand = _score(h_rand)
        params_rand = np.concatenate([h_rand, [t_rand]])
        
        try:
            result = minimize(
                objective,
                params_rand,
                jac=gradients,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.fun < best_score:
                h_candidate = result.x[:N]
                h_candidate = np.clip(h_candidate, 0, 1)
                h_candidate = h_candidate / np.sum(h_candidate) * (N/2.0)
                h_candidate = np.clip(h_candidate, 0, 1)
                h_candidate = h_candidate / np.sum(h_candidate) * (N/2.0)
                
                score_candidate = _score(h_candidate)
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_params = np.concatenate([h_candidate, [score_candidate]])
        except Exception:
            pass
    
    # Final polish: local optimization with tighter tolerance
    if not _time_exceeded():
        try:
            h_polish = best_params[:N].copy()
            t_polish = best_params[N]
            params_polish = np.concatenate([h_polish, [t_polish]])
            
            result = minimize(
                objective,
                params_polish,
                jac=gradients,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            
            if result.fun < best_score:
                h_candidate = result.x[:N]
                h_candidate = np.clip(h_candidate, 0, 1)
                h_candidate = h_candidate / np.sum(h_candidate) * (N/2.0)
                h_candidate = np.clip(h_candidate, 0, 1)
                h_candidate = h_candidate / np.sum(h_candidate) * (N/2.0)
                
                score_candidate = _score(h_candidate)
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_params = np.concatenate([h_candidate, [score_candidate]])
        except Exception:
            pass
    
    h_out = best_params[:N]
    h_out = np.clip(h_out, 0, 1)
    # Ensure exact sum
    h_out = h_out / np.sum(h_out) * (N/2.0)
    h_out = np.clip(h_out, 0, 1)
    # Final adjustment for exact sum
    diff = N/2.0 - np.sum(h_out)
    if abs(diff) > 1e-15:
        # Distribute the difference
        h_out += diff / N
        h_out = np.clip(h_out, 0, 1)
        h_out = h_out / np.sum(h_out) * (N/2.0)
    
    return h_out

# EVOLVE-BLOCK-END
