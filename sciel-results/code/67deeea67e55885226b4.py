"""Rewritten program for the third autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_chebyt

N = 400
DEADLINE = 135.0  # safety margin before 150s kill

# Precompute Chebyshev basis on [-1, 1]
_x = np.linspace(-1, 1, N)
_basis = np.array([eval_chebyt(k, _x) for k in range(16)])  # 16 coefficients
_basis_T = _basis.T  # shape (N, 16)
_basis_sum = _basis_T.sum(axis=0)  # sum of each basis function over the grid


def _coeffs_to_h(c):
    """Convert coefficient vector to step heights."""
    h = _basis_T @ c
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


def _hard_objective(c):
    """Hard objective for final evaluation."""
    h = _coeffs_to_h(c)
    return _value_from_h(h)


def _smooth_objective(c, beta=50.0):
    """
    Smooth surrogate for the objective using log-sum-exp.
    """
    h = _coeffs_to_h(c)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = conv - m
    # log-sum-exp for numerical stability
    logsumexp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = 2.0 * N * logsumexp / (s * s)
    return abs(val)


def _smooth_gradient(c, beta=50.0):
    """
    Numerical gradient of the smooth objective (central differences).
    """
    eps = 1e-7
    grad = np.zeros_like(c)
    f0 = _smooth_objective(c, beta)
    for i in range(len(c)):
        c_plus = c.copy()
        c_plus[i] += eps
        c_minus = c.copy()
        c_minus[i] -= eps
        grad[i] = (_smooth_objective(c_plus, beta) - _smooth_objective(c_minus, beta)) / (2 * eps)
    return grad


def _generate_diverse_init(rng, n_init=20):
    """
    Generate diverse initial coefficient vectors.
    Mix of:
    - Small random perturbations of the seed solution
    - Larger random vectors
    - Vectors emphasizing low-frequency Chebyshev modes
    """
    inits = []
    
    # Seed: all ones in coefficient space
    c_seed = np.linalg.pinv(_basis_T) @ np.ones(N)
    inits.append(c_seed.copy())
    
    for _ in range(n_init - 1):
        strategy = rng.random()
        if strategy < 0.3:
            # Small perturbation of seed
            c = c_seed + rng.normal(0, 0.5, len(c_seed))
        elif strategy < 0.6:
            # Moderate random
            c = rng.normal(0, 2.0, len(c_seed))
        else:
            # Emphasize low-frequency: large first few coefficients
            c = np.zeros(len(c_seed))
            for k in range(min(6, len(c_seed))):
                c[k] = rng.normal(0, 3.0)
            for k in range(6, len(c_seed)):
                c[k] = rng.normal(0, 0.5)
        inits.append(c)
    
    return inits


def construct():
    t0 = time.time()
    
    # Start with seed: all ones
    c_seed = np.linalg.pinv(_basis_T) @ np.ones(N)
    best_c = c_seed.copy()
    best_val = _hard_objective(best_c)
    
    # Phase 1: Differential Evolution over coefficients
    if time.time() - t0 < DEADLINE - 40:
        bounds = [(-8.0, 8.0) for _ in range(len(c_seed))]
        
        def de_obj(c):
            if time.time() - t0 > DEADLINE - 25:
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
                maxiter=150, popsize=20,
                tol=1e-7, seed=42,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_c = de_result.x.copy()
                best_val = de_result.fun
        except Exception:
            pass
    
    # Phase 2: Multi-start local refinement with smooth surrogate
    if time.time() - t0 < DEADLINE - 15:
        rng = np.random.default_rng(123)
        inits = _generate_diverse_init(rng, n_init=12)
        
        for beta in [30.0, 80.0, 150.0]:
            if time.time() - t0 > DEADLINE - 8:
                break
            for c0 in inits:
                if time.time() - t0 > DEADLINE - 5:
                    break
                try:
                    res = minimize(
                        _smooth_objective,
                        c0,
                        args=(beta,),
                        method='L-BFGS-B',
                        jac=_smooth_gradient,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
                    )
                    hard_val = _hard_objective(res.x)
                    if hard_val < best_val:
                        best_c = res.x.copy()
                        best_val = hard_val
                except Exception:
                    pass
    
    # Phase 3: Additional local refinement with tighter tolerance from best so far
    if time.time() - t0 < DEADLINE - 5:
        for beta in [100.0, 200.0, 400.0]:
            if time.time() - t0 > DEADLINE - 3:
                break
            try:
                res = minimize(
                    _smooth_objective,
                    best_c,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_smooth_gradient,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                hard_val = _hard_objective(res.x)
                if hard_val < best_val:
                    best_c = res.x.copy()
                    best_val = hard_val
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if time.time() - t0 < DEADLINE - 2:
        try:
            res = minimize(
                _smooth_objective,
                best_c,
                args=(500.0,),
                method='L-BFGS-B',
                jac=_smooth_gradient,
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            hard_val = _hard_objective(res.x)
            if hard_val < best_val:
                best_c = res.x.copy()
                best_val = hard_val
        except Exception:
            pass
    
    # Final: return the best h found
    h_best = _coeffs_to_h(best_c)
    
    if not np.all(np.isfinite(h_best)) or abs(np.sum(h_best)) < 1e-15:
        return np.ones(N)
    
    return h_best


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
