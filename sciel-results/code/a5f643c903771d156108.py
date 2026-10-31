import numpy as np
from scipy import optimize
import time

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if np.sum(h) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, 'full')
    total = np.sum(h)
    return np.max(conv) / (total * total)

def soft_objective(x, T=0.01):
    """
    Smooth surrogate for the autoconvolution peak using log-sum-exp.
    Minimizing this encourages spreading out the convolution mass,
    which lowers the peak more effectively than direct peak smoothing.
    """
    h = np.abs(x)
    conv = np.convolve(h, h, 'full')
    total = np.sum(h)
    if total < 1e-10:
        return 1e10
    conv_max = np.max(conv)
    # Numerically stable log-sum-exp
    shifted = (conv - conv_max) / T
    shifted = np.clip(shifted, -100, 100)
    soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
    return soft_max / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # Initial best: uniform distribution
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)
    
    np.random.seed(42)
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Uniform
    initial_guesses.append(np.ones(N))
    
    # 2. Center-enhanced (Gaussian-like)
    t = np.linspace(-1, 1, N)
    h2 = 1.0 + 0.3 * np.exp(-t**2 * 5)
    initial_guesses.append(np.maximum(h2, 0.01))
    
    # 3. Edge-enhanced
    h3 = 1.0 + 0.25 * np.abs(np.linspace(-1, 1, N))
    initial_guesses.append(np.maximum(h3, 0.01))
    
    # 4. Cosine perturbation
    t4 = np.linspace(0, 1, N, endpoint=False)
    h4 = 1.0 + 0.15 * np.cos(2 * np.pi * t4)
    initial_guesses.append(np.maximum(h4, 0.01))
    
    # 5. Sine perturbation
    h5 = 1.0 + 0.15 * np.sin(2 * np.pi * t4)
    initial_guesses.append(np.maximum(h5, 0.01))
    
    # 6. Random perturbation
    h6 = np.ones(N) + 0.1 * np.random.randn(N)
    initial_guesses.append(np.maximum(h6, 0.01))
    
    # 7. Parabolic
    h7 = 1.0 + 0.3 * (1 - t**2)
    initial_guesses.append(np.maximum(h7, 0.01))
    
    # 8. Random with different seed
    np.random.seed(123)
    h8 = np.random.rand(N) + 0.1
    h8 = h8 / np.mean(h8)
    initial_guesses.append(h8)
    
    # 9. Two-frequency perturbation
    h9 = 1.0 + 0.08 * np.cos(2 * np.pi * t4) + 0.08 * np.cos(4 * np.pi * t4)
    initial_guesses.append(np.maximum(h9, 0.01))
    
    # 10. Three-frequency perturbation
    h10 = 1.0 + 0.05 * np.cos(2 * np.pi * t4) + 0.05 * np.cos(4 * np.pi * t4) + 0.05 * np.cos(6 * np.pi * t4)
    initial_guesses.append(np.maximum(h10, 0.01))
    
    # Optimize each initial guess with two-stage approach
    for idx, h0 in enumerate(initial_guesses):
        if time.time() > deadline - 15:
            break
        
        h0 = np.abs(h0)
        
        # Stage 1: Coarse optimization with higher temperature
        try:
            result1 = optimize.minimize(
                lambda x: soft_objective(x, T=0.05), h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if time.time() > deadline - 10:
                break
            
            # Evaluate stage 1 result
            h_new = np.abs(result1.x)
            score1 = compute_autoconv_max(h_new)
            
            if score1 < best_score:
                best_score = score1
                best_h = h_new.copy()
            
            # Stage 2: Fine optimization with lower temperature
            if time.time() > deadline - 8:
                break
                
            # Use the better of h0 and result1.x as starting point
            start_h = h_new if score1 < compute_autoconv_max(h0) else h0
            
            result2 = optimize.minimize(
                lambda x: soft_objective(x, T=0.01), start_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Evaluate stage 2 result
            h_final = np.abs(result2.x)
            score2 = compute_autoconv_max(h_final)
            
            if score2 < best_score:
                best_score = score2
                best_h = h_final.copy()
                
        except Exception:
            pass
    
    # Final polish: try to improve the best solution with very low temperature
    if time.time() < deadline - 5:
        try:
            for _ in range(5):
                if time.time() > deadline - 3:
                    break
                h_perturbed = best_h + 0.005 * np.random.randn(N)
                h_perturbed = np.maximum(h_perturbed, 0)
                
                result = optimize.minimize(
                    lambda x: soft_objective(x, T=0.005), h_perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_new = np.abs(result.x)
                score = compute_autoconv_max(h_new)
                if score < best_score:
                    best_score = score
                    best_h = h_new.copy()
                    
        except Exception:
            pass
    
    return list(best_h)
