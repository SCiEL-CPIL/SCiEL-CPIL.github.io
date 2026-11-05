"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_chebyt

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute Chebyshev basis on [-1, 1]
_x = np.linspace(-1, 1, N)
_basis = np.array([eval_chebyt(k, _x) for k in range(12)])  # 12 coefficients
_basis_T = _basis.T  # shape (N, 12)


def _coeffs_to_h(c):
    """Convert coefficient vector to step heights."""
    h = _basis_T @ c
    # Ensure sum is not zero (it won't be for reasonable c, but guard)
    s = np.sum(h)
    if abs(s) < 1e-10:
        h[0] += 1e-6
    return h


def _value_from_h(h):
    """Compute the task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))


def _softmax_surrogate(c, beta=50.0):
    """
    Smooth surrogate for the objective using log-sum-exp.
    """
    h = _coeffs_to_h(c)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    # Soft max via log-sum-exp
    m = np.max(conv)
    shifted = conv - m
    # For numerical stability, use a moderate beta
    logsumexp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    # Actually standard log-sum-exp: m + (1/beta)*log(sum(exp(beta*(x-m))))
    # This approximates max as beta->inf
    val = 2.0 * N * logsumexp / (s * s)
    # We want abs, but since we're minimizing and the seed gives 2.0 (positive max),
    # let's just use the value directly. If max(conv) is negative, abs would flip it.
    # For now, minimize the soft-max value; if it goes negative we'll take abs at the end.
    return abs(val)


def _hard_objective(c):
    """Hard objective for final evaluation."""
    h = _coeffs_to_h(c)
    return _value_from_h(h)


def _spectral_guided_init():
    """
    Generate initial coefficients guided by spectral considerations.
    We want to minimize the peak of the autoconvolution relative to the sum squared.
    In the frequency domain, the autoconvolution is related to the power spectrum.
    To minimize the peak, we want a flat power spectrum (white noise).
    """
    # Start with random coefficients
    rng = np.random.default_rng(42)
    c = rng.normal(0, 1, len(_basis_T[0]))
    return c


def construct():
    t0 = time.time()
    
    # Start with seed: all ones
    # In Chebyshev basis, what coefficients give h = ones?
    # h = _basis_T @ c = ones
    # c = pinv(_basis_T) @ ones
    c_seed = np.linalg.pinv(_basis_T) @ np.ones(N)
    
    best_c = c_seed.copy()
    best_val = _hard_objective(best_c)
    
    # Phase 1: Differential Evolution over coefficients
    if time.time() - t0 < DEADLINE - 30:
        # Bounds for each coefficient
        # The seed coefficients are small; let's use reasonable bounds
        bounds = [(-5.0, 5.0) for _ in range(len(c_seed))]
        
        def de_obj(c):
            if time.time() - t0 > DEADLINE - 20:
                return best_val
            h = _coeffs_to_h(c)
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            return abs(2.0 * N * np.max(conv) / (s * s))
        
        try:
            de_result = differential_evolution(
                de_obj, bounds,
                maxiter=200, popsize=15,
                tol=1e-6, seed=42,
                polish=False  # We'll do our own polish
            )
            if de_result.fun < best_val:
                best_c = de_result.x.copy()
                best_val = de_result.fun
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B using soft-max surrogate
    if time.time() - t0 < DEADLINE - 10:
        # Try multiple beta values for soft-max
        for beta in [20.0, 50.0, 100.0]:
            if time.time() - t0 > DEADLINE - 5:
                break
            try:
                res = minimize(
                    _softmax_surrogate,
                    best_c,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    # Verify with hard objective
                    hard_val = _hard_objective(res.x)
                    if hard_val < best_val:
                        best_c = res.x.copy()
                        best_val = hard_val
            except Exception:
                pass
    
    # Phase 3: Additional local refinement with tighter tolerance
    if time.time() - t0 < DEADLINE - 5:
        try:
            res = minimize(
                _softmax_surrogate,
                best_c,
                args=(200.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            hard_val = _hard_objective(res.x)
            if hard_val < best_val:
                best_c = res.x.copy()
                best_val = hard_val
        except Exception:
            pass
    
    # Final: return the best h found
    h_best = _coeffs_to_h(best_c)
    
    # Safety check
    if not np.all(np.isfinite(h_best)) or abs(np.sum(h_best)) < 1e-15:
        return np.ones(N)
    
    return h_best


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
