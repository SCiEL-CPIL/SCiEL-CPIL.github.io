"""Rewritten program for the third autocorrelation inequality task using DCT spectral shaping."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.fftpack import dct

_DEADLINE = 145.0  # wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _build_from_dct(coeffs):
    """
    Build h from DCT-II coefficients.
    coeffs[k] are the DCT-II coefficients; h[n] = sum_k coeffs[k] * dct_basis[k][n]
    Using scipy's dct with type=2: coeffs = dct(h, type=2, norm='ortho')
    Inverse: h = idct(coeffs, type=2, norm='ortho')
    """
    return dct(coeffs, type=2, norm='ortho', axis=0)


def _dct_objective(coeffs, K):
    """
    Objective: minimize the task value given DCT coefficients.
    coeffs has length K (first K DCT modes used, rest zero).
    """
    full_coeffs = np.zeros(N)
    full_coeffs[:K] = coeffs
    h = _build_from_dct(full_coeffs)
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val


def _softmax_objective(coeffs, K, tau=0.05):
    """
    Smooth surrogate: replace np.max(conv) with soft-max via log-sum-exp.
    This makes the objective differentiable almost everywhere.
    """
    full_coeffs = np.zeros(N)
    full_coeffs[:K] = coeffs
    h = _build_from_dct(full_coeffs)
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    # Soft max: tau * log(sum(exp(conv/tau)))
    c_max = np.max(conv)
    soft_max = c_max + tau * np.log(np.sum(np.exp((conv - c_max) / tau)))
    val = abs(2.0 * N * soft_max / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val


def _projected_gradient_descent(coeffs_init, K, deadline, lr=0.01, n_iter=5000, use_softmax=True, tau=0.05):
    """
    Projected gradient descent on DCT coefficients.
    The 'projection' here is just clipping coefficients to a reasonable range
    to maintain numerical stability.
    """
    best_coeffs = coeffs_init.copy()
    best_val = (_softmax_objective if use_softmax else _dct_objective)(best_coeffs, K, **({'tau': tau} if use_softmax else {}))
    
    lr_current = lr
    patience = 0
    max_patience = 200
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        
        # Numerical gradient via finite differences
        grad = np.zeros(K)
        eps = 1e-5
        obj_fn = _softmax_objective if use_softmax else _dct_objective
        obj_kwargs = {'tau': tau} if use_softmax else {}
        
        for j in range(K):
            cp = best_coeffs.copy()
            cm = best_coeffs.copy()
            cp[j] += eps
            cm[j] -= eps
            fp = obj_fn(cp, K, **obj_kwargs)
            fm = obj_fn(cm, K, **obj_kwargs)
            grad[j] = (fp - fm) / (2 * eps)
        
        # Gradient step
        new_coeffs = best_coeffs - lr_current * grad
        
        # Clip to reasonable range for stability
        new_coeffs = np.clip(new_coeffs, -5.0, 5.0)
        
        new_val = obj_fn(new_coeffs, K, **obj_kwargs)
        
        if np.isfinite(new_val) and new_val < best_val - 1e-12:
            best_coeffs = new_coeffs
            best_val = new_val
            patience = 0
            # Adaptive learning rate
            lr_current = min(lr_current * 1.01, 0.5)
        else:
            patience += 1
            lr_current *= 0.95
            if patience > max_patience:
                break
    
    # Final evaluation with exact (non-soft) objective
    final_val = _dct_objective(best_coeffs, K)
    return best_coeffs, min(best_val, final_val)


def _local_refine(h, deadline, max_iter=20000):
    """
    Local refinement: try small random perturbations and keep improvements.
    """
    best_val = _evaluate(h)
    best_h = h.copy()
    rng = np.random.default_rng(42)
    
    step_size = 0.1
    for step in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try perturbing 1-5 random indices
        n_perturb = rng.integers(1, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _evaluate(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-10:
            best_val = v_new
            best_h = h_new
            # Reduce step size on success
            step_size *= 0.98
        elif step % 1000 == 0:
            # Gradually reduce step size if no recent improvement
            step_size *= 0.999
    
    return best_h, best_val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    Uses DCT spectral shaping: parameterize h via first K DCT-II coefficients
    and optimize them via projected gradient descent with soft-max surrogate.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(123)
    
    # Track best solution
    best_val = 2.0  # seed value
    best_h = np.ones(N)
    
    # Phase 1: DCT spectral shaping with varying K
    # Try different numbers of DCT modes
    for K in [2, 4, 6, 8, 12, 16, 24, 32]:
        if time.time() > deadline - 10:
            break
        
        # Multiple random initializations
        n_inits = max(3, 10 - K // 4)
        for init_idx in range(n_inits):
            if time.time() > deadline - 8:
                break
            
            # Random initial DCT coefficients
            coeffs_init = rng.normal(0, 0.5, K)
            # Ensure nonzero sum: DCT mode 0 controls the mean
            coeffs_init[0] = rng.uniform(0.5, 2.0)
            
            # Projected gradient descent with soft-max
            opt_coeffs, opt_val = _projected_gradient_descent(
                coeffs_init, K, deadline - 5,
                lr=0.01, n_iter=3000, use_softmax=True, tau=0.05
            )
            
            if np.isfinite(opt_val) and opt_val < best_val:
                best_val = opt_val
                full_coeffs = np.zeros(N)
                full_coeffs[:K] = opt_coeffs
                best_h = _build_from_dct(full_coeffs)
        
        # Also try exact (non-soft) objective for final polish on best K
        if time.time() < deadline - 8:
            coeffs_init = rng.normal(0, 0.3, K)
            coeffs_init[0] = rng.uniform(0.5, 2.0)
            
            opt_coeffs, opt_val = _projected_gradient_descent(
                coeffs_init, K, deadline - 5,
                lr=0.005, n_iter=2000, use_softmax=False
            )
            
            if np.isfinite(opt_val) and opt_val < best_val:
                best_val = opt_val
                full_coeffs = np.zeros(N)
                full_coeffs[:K] = opt_coeffs
                best_h = _build_from_dct(full_coeffs)
    
    # Phase 2: Differential evolution on DCT coefficients for global search
    if time.time() < deadline - 25:
        K_de = 8  # Moderate dimension for DE
        bounds = [(-3.0, 3.0)] * K_de
        bounds[0] = (0.3, 3.0)  # DC component must be positive enough for nonzero sum
        
        def _de_objective(coeffs):
            return _dct_objective(coeffs, K_de)
        
        try:
            de_result = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=80,
                tol=1e-8,
                popsize=15,
                polish=True
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                full_coeffs = np.zeros(N)
                full_coeffs[:K_de] = de_result.x
                best_h = _build_from_dct(full_coeffs)
        except Exception:
            pass
        
        # Try a second DE run with different seed and more modes
        if time.time() < deadline - 15:
            K_de2 = 12
            bounds2 = [(-3.0, 3.0)] * K_de2
            bounds2[0] = (0.3, 3.0)
            
            def _de_objective2(coeffs):
                return _dct_objective(coeffs, K_de2)
            
            try:
                de_result2 = differential_evolution(
                    _de_objective2,
                    bounds2,
                    seed=7,
                    maxiter=60,
                    tol=1e-8,
                    popsize=12,
                    polish=True
                )
                if de_result2.fun < best_val:
                    best_val = de_result2.fun
                    full_coeffs = np.zeros(N)
                    full_coeffs[:K_de2] = de_result2.x
                    best_h = _build_from_dct(full_coeffs)
            except Exception:
                pass
    
    # Phase 3: Local refinement of the best candidate
    if time.time() < deadline - 10:
        best_h, best_val = _local_refine(best_h, deadline - 5)
    
    # Phase 4: Try structured patterns that might have low autocorrelation
    if time.time() < deadline - 15:
        # Pattern 1: Low-frequency dominant via DCT
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            K_trial = rng.integers(4, 16)
            coeffs_trial = np.zeros(N)
            coeffs_trial[0] = rng.uniform(0.5, 1.5)
            for k in range(1, min(K_trial, N)):
                coeffs_trial[k] = rng.normal(0, 0.3)
            
            h_test = _build_from_dct(coeffs_trial)
            
            if np.sum(h_test) == 0 or not np.isfinite(np.sum(h_test)):
                continue
            
            h_opt, v_opt = _local_refine(h_test, deadline - 5, max_iter=5000)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
        
        # Pattern 2: Alternating block patterns
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            n_blocks = rng.integers(4, 20)
            block_size = N // n_blocks
            h_test = np.zeros(N)
            sign = 1
            for i in range(0, N, block_size):
                val = rng.uniform(0.5, 1.5)
                h_test[i:min(i + block_size, N)] = sign * val
                sign *= -1
            
            if np.sum(h_test) == 0:
                continue
            
            h_opt, v_opt = _local_refine(h_test, deadline - 5, max_iter=5000)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
    
    # Phase 5: Final high-precision local refinement
    if time.time() < deadline - 5:
        best_h, best_val = _local_refine(best_h, deadline - 2, max_iter=15000)
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        # Fall back to seed if optimization failed
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
