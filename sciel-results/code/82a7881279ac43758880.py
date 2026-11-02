import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: all ones (maximizes sum, simple structure)
    f0 = np.ones(N)
    
    # We optimize the flattened vector
    f = f0.copy()
    
    # Objective function
    def objective(x):
        # x is the sequence
        # Compute linear autocorrelation via FFT
        # Zero pad to 2*N to avoid circular aliasing
        F = np.fft.fft(x, n=2*N)
        R = np.fft.ifft(np.abs(F)**2).real
        
        # R contains lags 0 to N-1 and -N+1 to -1
        # R[0] is lag 0. R[1:N] are lags 1..N-1. R[N:2N] are lags -N+1..-1.
        
        # Extract side lobes (exclude lag 0)
        # Lags 1 to N-1 are R[1:N]
        # Lags -N+1 to -1 are R[N:2N]
        side_lobes = np.concatenate([R[1:N], R[N:2*N]])
        
        abs_side = np.abs(side_lobes)
        
        # Soft max approximation
        # Use alpha to control sharpness. Higher alpha -> closer to max.
        # Start with moderate alpha.
        alpha = 20.0
        # Avoid overflow in exp
        max_val = np.max(abs_side)
        if max_val == 0:
            soft_max = 0.0
        else:
            # Shift by max_val for numerical stability
            exp_terms = np.exp(alpha * (abs_side - max_val))
            soft_max = max_val + (1.0/alpha) * np.log(np.sum(exp_terms))
            
        S = np.sum(x)
        S2 = S * S
        
        # If sum is too small, penalize heavily
        if S2 < 1e-4:
            return 1e10 + soft_max
        
        # Minimize soft_max / S2
        # Equivalent to minimizing log(soft_max) - 2*log(S2)
        # But soft_max can be 0.
        # Use ratio directly? Gradient of ratio is messy.
        # Let's minimize: soft_max + lambda / S2
        # Or just soft_max - 2*log(S2)
        
        # To keep scale reasonable, we can normalize x?
        # The objective is scale invariant.
        # L-BFGS might wander in scale.
        # We can add a constraint or regularization to keep norm fixed.
        # ||x||^2 = 1?
        # If we fix norm, S is bounded.
        # Let's add a penalty if ||x||^2 deviates from 1.
        norm2 = np.dot(x, x)
        norm_penalty = 100.0 * (norm2 - 1.0)**2
        
        return soft_max - 2.0 * np.log(S2 + 1e-9) + norm_penalty

    def gradient(x):
        # Numerical gradient is safe but slow.
        # Analytical gradient is better.
        # Let's try numerical first for correctness, then optimize if needed.
        # N=400, numerical grad is 800 evals.
        # 100 L-BFGS steps * 800 evals * (2N FFT) = 80000 FFTs.
        # FFT 800 length is fast.
        # Should be fine.
        
        eps = 1e-7
        grad = np.zeros_like(x)
        f0_val = objective(x)
        for i in range(N):
            x[i] += eps
            f_plus = objective(x)
            x[i] -= 2*eps
            f_minus = objective(x)
            x[i] += eps # restore
            
            grad[i] = (f_plus - f_minus) / (2*eps)
            
        return grad

    # Optimization loop
    best_score = np.inf
    best_f = f0.copy()
    
    # Compute initial score
    def compute_score(x):
        F = np.fft.fft(x, n=2*N)
        R = np.fft.ifft(np.abs(F)**2).real
        side_lobes = np.concatenate([R[1:N], R[N:2*N]])
        max_side = np.max(np.abs(side_lobes))
        S = np.sum(x)
        if S == 0: return np.inf
        return max_side / (S*S)

    current_score = compute_score(f0)
    if current_score < best_score:
        best_score = current_score
        best_f = f0.copy()

    # Run L-BFGS
    # We might need to restart or use multiple alphas
    # Let's run a few iterations
    
    for alpha_val in [10.0, 20.0, 50.0]:
        if time.time() > deadline:
            break
            
        # Redefine objective with current alpha
        def obj_alpha(x):
            F = np.fft.fft(x, n=2*N)
            R = np.fft.ifft(np.abs(F)**2).real
            side_lobes = np.concatenate([R[1:N], R[N:2*N]])
            abs_side = np.abs(side_lobes)
            
            max_val = np.max(abs_side)
            if max_val == 0:
                soft_max = 0.0
            else:
                exp_terms = np.exp(alpha_val * (abs_side - max_val))
                soft_max = max_val + (1.0/alpha_val) * np.log(np.sum(exp_terms))
                
            S = np.sum(x)
            S2 = S * S
            if S2 < 1e-4:
                return 1e10 + soft_max
            
            norm2 = np.dot(x, x)
            norm_penalty = 10.0 * (norm2 - 1.0)**2
            
            return soft_max - 2.0 * np.log(S2 + 1e-9) + norm_penalty

        def grad_alpha(x):
            # Numerical gradient for this specific alpha
            eps = 1e-6
            grad = np.zeros_like(x)
            f0_val = obj_alpha(x)
            for i in range(N):
                x[i] += eps
                f_plus = obj_alpha(x)
                x[i] -= 2*eps
                f_minus = obj_alpha(x)
                x[i] += eps
                grad[i] = (f_plus - f_minus) / (2*eps)
            return grad

        try:
            res = minimize(obj_alpha, f, method='L-BFGS-B', jac=grad_alpha, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            f = res.x
            current_score = compute_score(f)
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
        except Exception as e:
            pass

    # Final polish: Peak targeted
    # Identify top peaks and try to reduce them specifically
    if time.time() < deadline - 10:
        # Simple local search on the best_f
        # Try small perturbations
        for _ in range(50):
            if time.time() > deadline: break
            # Perturb a few random elements
            idxs = np.random.choice(N, size=5, replace=False)
            f_pert = best_f.copy()
            for idx in idxs:
                f_pert[idx] += np.random.normal(0, 0.1)
            
            score_pert = compute_score(f_pert)
            if score_pert < best_score:
                best_score = score_pert
                best_f = f_pert

    return best_f
# EVOLVE-BLOCK-END
