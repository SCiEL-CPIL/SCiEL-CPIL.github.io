"""Optimized program: numerical minimization of the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial


def _hermite_prob(n, x):
    """Probabilists' Hermite polynomial Hn(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0 = np.ones_like(x)
    h1 = x.copy()
    for k in range(2, n + 1):
        h2 = x * h1 - (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_c4_bound_normalized(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(g) * A(ĝ) for g normalized to ||g||_2 = 1.
    Since g is composed of even Hermite-Gaussians, ĝ(ξ) = g(ξ) (up to a sign/phase
    which is 1 for even degrees in the unitary transform), so A(ĝ) = A(g).
    Thus the bound is A(g)^2.
    """
    a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Use a sufficiently fine grid
    x = np.linspace(-12, 12, 24001)
    dx = x[1] - x[0]
    
    h0 = _hermite_prob(0, x)
    h4 = _hermite_prob(4, x)
    h8 = _hermite_prob(8, x)
    h12 = _hermite_prob(12, x)
    
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    g = poly * np.exp(-x ** 2 / 2.0)
    
    # Calculate L2 norm
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-12:
        return 1e10
        
    norm = np.sqrt(norm_sq)
    g_norm = g / norm
    
    # Sup norm
    A_g = np.max(np.abs(g_norm))
    
    return A_g * A_g


def construct():
    """
    Numerically optimize the three free Hermite coefficients (a0, a4, a8)
    to minimize the upper bound on C4 = A(g) * A(ĝ), with the constraint
    g(0) = 0 enforced by solving for a12.
    """
    def objective(coeffs):
        a0, a4, a8 = coeffs
        try:
            return _compute_c4_bound_normalized(a0, a4, a8)
        except Exception:
            return 1e10
    
    # Seed coefficients: a0 = -1.13, a4 = 0.04, a8 = 0.0003
    seed_coeffs = np.array([-1.13, 0.04, 0.0003])
    
    # Bounds for each coefficient
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-0.1, 0.1)]
    
    # Global optimization using differential evolution
    best_val = np.inf
    best_x = seed_coeffs.copy()
    
    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=300,
            popsize=20,
            tol=1e-6,
            atol=1e-8,
            init='sobol',
            polish=True
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_x = de_result.x
    except Exception:
        pass
    
    # Local refinement from the best point found
    try:
        local_result = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'gtol': 1e-12, 'ftol': 1e-12}
        )
        if local_result.fun < best_val:
            best_val = local_result.fun
            best_x = local_result.x
    except Exception:
        pass
    
    # Additional random restarts for robustness
    np.random.seed(123)
    for _ in range(15):
        start = np.array([
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(-0.05, 0.05)
        ])
        try:
            res = minimize(
                objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final refinement
    try:
        final_result = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'gtol': 1e-12, 'ftol': 1e-12}
        )
        if final_result.fun < best_val:
            best_val = final_result.fun
            best_x = final_result.x
    except Exception:
        pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
