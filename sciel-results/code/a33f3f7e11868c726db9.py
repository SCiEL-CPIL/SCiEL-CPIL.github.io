"""Optimized signed step function minimizing max absolute autoconvolution."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill


def _autoconv(x):
    """Compute autoconvolution using FFT."""
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * np.conj(f), n=2*n)[:n]
    return ac


def _objective(params, x0, t_scale):
    """
    Objective: minimize max|autocorr| / autocorr[0]
    Using log-sum-exp for smooth max approximation.
    params: [x..., t] where t is the log-sum-exp temperature
    """
    x = np.tanh(params[:-1])  # Map to [-1, 1]
    t = max(params[-1], 1e-6)
    
    ac = _autoconv(x)
    ac0 = ac[0]
    if ac0 <= 1e-12:
        return 1e6
    
    # Ratio for each lag
    ratios = np.abs(ac) / ac0
    # Exclude lag 0 (always 1.0)
    ratios_nonzero = np.concatenate([ratios[1:], ratios[:n-1][::-1]]) if False else ratios[1:]
    # Actually autoconv is symmetric, just use lags 1..N-1
    ratios_lags = ratios[1:]
    
    # Log-sum-exp for smooth max
    m = np.max(ratios_lags)
    if m < 1e-10:
        lse = m
    else:
        lse = m + np.log(np.sum(np.exp((ratios_lags - m) / t)) * t)
    
    # Add small penalty to encourage |x| close to 1
    penalty = 0.01 * np.mean((1 - x**2)**2)
    
    return lse + penalty


def construct():
    start_time = time.time()
    
    best_x = np.ones(N)
    best_score = np.inf
    
    # Initial: try alternating pattern
    x_init = np.ones(N)
    x_init[1::2] = -1
    ac_init = _autoconv(x_init)
    score_init = np.max(np.abs(ac_init[1:])) / ac_init[0]
    best_score = score_init
    best_x = x_init.copy()
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() - start_time > DEADLINE - 30:
            break
        x_rand = rng.choice([-1, 1], size=N)
        ac_rand = _autoconv(x_rand)
        score_rand = np.max(np.abs(ac_rand[1:])) / ac_rand[0]
        if score_rand < best_score:
            best_score = score_rand
            best_x = x_rand.copy()
    
    # Gradient-based refinement using smooth surrogate
    # Use continuous relaxation: x in [-1,1], then round
    def _smooth_obj_and_grad(theta):
        x = np.tanh(theta)
        ac = _autoconv(x)
        ac0 = ac[0]
        if ac0 <= 1e-12:
            return 1e6, np.zeros_like(theta)
        
        ratios = np.abs(ac) / ac0
        ratios_lags = ratios[1:]
        
        # Smooth max via log-sum-exp
        t = 0.1
        m = np.max(ratios_lags)
        if m < 1e-10:
            lse = m
            weights = np.ones_like(ratios_lags) / len(ratios_lags)
        else:
            exp_terms = np.exp((ratios_lags - m) / t)
            sum_exp = np.sum(exp_terms)
            lse = m + t * np.log(sum_exp)
            weights = exp_terms / sum_exp
        
        # Gradient of lse w.r.t. ac
        grad_ac = np.zeros_like(ac)
        grad_ac[1:] = weights / ac0
        # d|ac[i]|/dac[i] = sign(ac[i])
        signs = np.sign(ac)
        signs[0] = 0  # Don't penalize lag 0
        grad_ac = grad_ac * signs
        # d(ac[i]/ac0)/dac = grad_ac[i]/ac0 - ac[i]*grad_ac[i]/ac0^2 for i>=1
        # Simplified: just use the weight-based gradient
        # Actually, let's compute more carefully
        # f = max_i |ac[i]|/ac0, smoothed
        # df/dac[i] = (1/ac0) * sign(ac[i]) * w_i for i>=1
        # df/dac[0] = -sum_i w_i * |ac[i]| / ac0^2
        
        grad_ac = np.zeros_like(ac)
        for i in range(1, N):
            grad_ac[i] = weights[i-1] * np.sign(ac[i]) / ac0
        grad_ac[0] = -np.sum(weights * np.abs(ac[1:])) / (ac0 * ac0)
        
        # Gradient of ac w.r.t. x via FFT
        # ac = ifft(fft(x) * conj(fft(x)))[:N]
        # dac/dx = 2 * ifft(fft(x) * conj(fft(x))) ... need careful derivation
        # Use numerical or analytical: ac[i] = sum_j x[j]x[j+i]
        # dac[i]/dx[k] = x[k-i] + x[k+i] (with appropriate bounds)
        
        # Simpler: use finite difference for gradient, but that's slow
        # Let's use the analytical gradient via FFT
        # ac = ifft(|X|^2) where X = fft(x)
        # d ac / d x = 2 * ifft(X * d|X|^2/dx) ... this is getting complex
        # 
        # Alternative: use the fact that ac[i] = sum_k x[k]*x[k+i]
        # d ac[i] / d x[k] = x[k-i] + x[k+i] (with x[j]=0 for j out of [0,N-1])
        
        # This is O(N^2) which is too slow for N=400 in Python
        # Use FFT-based gradient
        # 
        # Let's just use a simple approach: perturb and compute
        # But for speed, let's use a coarser method
        
        # Actually, let's just do coordinate descent or a simpler optimization
        return lse, None
    
    # Given time constraints, let's use a simpler iterative approach
    # Local search: flip individual bits and check improvement
    
    def _score(x):
        ac = _autoconv(x)
        ac0 = ac[0]
        if ac0 <= 1e-12:
            return 1e6
        return np.max(np.abs(ac[1:])) / ac0
    
    # Hill climbing with random restarts
    for restart in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        if restart == 0:
            x = best_x.copy()
        else:
            x = rng.choice([-1, 1], size=N)
        
        current_score = _score(x)
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
        
        # Try flipping pairs of bits
        improved = True
        max_iters = 500
        iters = 0
        while improved and iters < max_iters:
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            iters += 1
            
            # Try random flips
            for _ in range(50):
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue
                
                x[i] *= -1
                x[j] *= -1
                new_score = _score(x)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                    improved = True
                else:
                    x[i] *= -1
                    x[j] *= -1
        
        # Also try single bit flips
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            for i in range(N):
                if time.time() - start_time > DEADLINE - 5:
                    break
                x[i] *= -1
                new_score = _score(x)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                    improved = True
                else:
                    x[i] *= -1
    
    return best_x
