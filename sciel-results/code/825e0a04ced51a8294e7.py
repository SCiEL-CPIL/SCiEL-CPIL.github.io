"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Combined strategy: 1D alpha search + multi-start DE + high-precision L-BFGS-B polish.
Key improvements: efficient gradient via log-derivatives, better initial guesses, more robust polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _build_p_coeffs(roots):
    """Build P(x) = prod (x-x_i)^2 in ascending coeff order. Returns P_coeffs (degree 24)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)  # degree 12, ascending
    P = polymul(B, B)          # degree 24
    return P

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2/I1^2 where I1=int P e^{-x}, I2=int P^2 e^{-x}, P=prod(x-x_i)^2."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P = _build_p_coeffs(roots)
    I1 = _poly_moment(P)
    if I1 <= 0 or not np.isfinite(I1):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # P^2 = P * P
    from numpy.polynomial.polynomial import polymul
    P2 = polymul(P, P)
    I2 = _poly_moment(P2)
    if not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Efficient gradient: precompute P and P2, then for each i compute
    # S_i = P/(x-x_i) via synthetic division (faster than polydiv)
    # dI1/dx_i = -2 * int S_i * e^{-x} dx
    # dI2/dx_i = -4 * int P * S_i * e^{-x} dx
    # C4 = I2/I1^2, so dC4/dx_i = (dI2*I1^2 - I2*2*I1*dI1)/I1^4 = dI2/I1^2 - 2*I2*dI1/I1^3

    grad = np.zeros(n)
    inv_I1 = 1.0 / I1
    inv_I1_sq = inv_I1 * inv_I1
    inv_I1_cu = inv_I1_sq * inv_I1

    # Precompute P^2 coefficients once
    # For each i, we need S_i = P/(x-x_i) and T_i = P*S_i = P^2/(x-x_i)
    # Use synthetic division for speed

    from numpy.polynomial.polynomial import polymul

    for i in range(n):
        # Synthetic division of P by (x - r_i)
        # P is in ascending order: P[0] + P[1]*x + ... + P[24]*x^24
        # Divide by (x - r): quotient has degree 23
        r = roots[i]
        P_arr = P
        deg = len(P_arr) - 1
        S_i = np.zeros(deg)  # degree 23
        # Synthetic division: start from highest degree
        carry = P_arr[deg]
        for k in range(deg - 1, -1, -1):
            S_i[k] = carry
            carry = P_arr[k] + carry * r
        # carry should be ~0 (remainder) since r_i is a root
        
        dI1_i = -2.0 * _poly_moment(S_i)
        
        # T_i = P * S_i (degree 47)
        T_i = polymul(P, S_i)
        dI2_i = -4.0 * _poly_moment(T_i)
        
        grad[i] = dI2_i * inv_I1_sq - 2.0 * I2 * dI1_i * inv_I1_cu

    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _c4_from_alpha(alpha):
    """Compute C4 for generalized Laguerre roots with parameter alpha."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if np.all(xg > 0) and np.all(np.isfinite(xg)):
            return _compute_c4(xg)
    except Exception:
        pass
    return np.inf

def _dyson_init(n=12, beta=2.0):
    """Initialize roots using Dyson gas equilibrium: sum_{j!=i} 2/(r_i - r_j) = 1 (confining potential V'(x)=1).
    Use iterative relaxation to find equilibrium configuration."""
    # Start from Laguerre roots
    x0, _ = roots_laguerre(n)
    x = x0.copy()
    
    # Iterate: adjust each root to satisfy the Dyson balance
    for iteration in range(200):
        converged = True
        for i in range(n):
            # Compute force from other roots: sum_{j!=i} 2/(x_i - x_j)
            force = 0.0
            for j in range(n):
                if j != i:
                    force += 2.0 / (x[i] - x[j])
            # Equilibrium: force = 1 (confining potential derivative)
            # Newton step: dx = (1 - force) / (sum_{j!=i} 4/(x_i - x_j)^2)
            curvature = 0.0
            for j in range(n):
                if j != i:
                    curvature += 4.0 / (x[i] - x[j])**2
            if curvature > 1e-12:
                dx = (1.0 - force) / curvature
                # Limit step size
                dx = np.clip(dx, -0.5, 0.5)
                x[i] += dx
                if abs(dx) > 1e-8:
                    converged = False
            else:
                x[i] += 0.1 * (1.0 - force)
                if abs(1.0 - force) > 1e-8:
                    converged = False
        if not np.all(x > 0):
            x = np.maximum(x, 1e-3)
        if converged:
            break
    
    return np.sort(x)

def construct():
    start = time.time()
    DEADLINE = 600.0
    SAFETY = 30.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < deadline

    # Phase 1: 1D search over generalized Laguerre parameter alpha
    # This gives a good initial guess based on root density profile
    best_roots = None
    best_c4 = np.inf

    # Grid search over alpha
    for alpha in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        if not time_ok():
            break
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if np.all(xg > 0) and np.all(np.isfinite(xg)):
            c4g = _compute_c4(xg)
            if c4g < best_c4:
                best_c4 = c4g
                best_roots = xg.copy()

    # Standard Laguerre as fallback
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Dyson gas initialization
    try:
        x_d = _dyson_init(12)
        if np.all(x_d > 0) and np.all(np.isfinite(x_d)):
            c4_d = _compute_c4(x_d)
            if c4_d < best_c4:
                best_c4 = c4_d
                best_roots = x_d.copy()
    except Exception:
        pass

    # Phase 2: Differential evolution for global search
    # Use the best alpha-derived roots as a starting point in the population
    if time_ok():
        bounds = [(1e-4, 200.0)] * 12
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=250,
                popsize=18,
                tol=1e-10,
                polish=False,
                mutation=(0.4, 1.0),
                recombination=0.7,
                update_interval=8
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Second DE run with different seed if time permits
    if time_ok():
        try:
            de_result2 = differential_evolution(
                _objective,
                bounds=bounds,
                seed=137,
                maxiter=150,
                popsize=15,
                tol=1e-10,
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.6,
                update_interval=10
            )
            if np.all(de_result2.x > 0) and np.isfinite(de_result2.fun):
                candidate = np.sort(de_result2.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B refinement from multiple starting points
    candidates_to_refine = [best_roots.copy()]
    
    # Perturbed versions
    rng = np.random.default_rng(42)
    for scale in [0.05, 0.1, 0.2, 0.35]:
        for _ in range(2):
            pert = best_roots * (1.0 + scale * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            candidates_to_refine.append(pert)

    # Add best alpha-derived roots
    for alpha in [0.5, 1.0, 2.0]:
        try:
            xg, _ = roots_genlaguerre(12, alpha)
            xg = np.sort(xg)
            if np.all(xg > 0):
                candidates_to_refine.append(xg.copy())
        except Exception:
            pass

    for init in candidates_to_refine:
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: High-precision polish passes with multiple strategies
    for polish_iter in range(6):
        if not time_ok():
            break
        try:
            # L-BFGS-B with very tight tolerances
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-12, None)] * 12,
                options={'maxiter': 3000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    # Try BFGS as alternative
                    try:
                        res2 = minimize(
                            _objective_grad,
                            best_roots,
                            jac=True,
                            method='BFGS',
                            options={'maxiter': 1000, 'gtol': 1e-16}
                        )
                        if np.all(res2.x > 0) and np.isfinite(res2.fun):
                            candidate2 = np.sort(res2.x)
                            c4_2 = _compute_c4(candidate2)
                            if c4_2 < best_c4:
                                best_c4 = c4_2
                                best_roots = candidate2.copy()
                    except Exception:
                        pass
                    break
        except Exception:
            break

    # Phase 5: Coordinate descent refinement
    if time_ok():
        for cd_iter in range(200):
            if not time_ok():
                break
            improved = False
            for i in range(12):
                # Try small perturbations of root i
                base = best_roots.copy()
                for delta in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3]:
                    test = base.copy()
                    test[i] += delta
                    if test[i] > 1e-8 and np.all(test > 0):
                        test = np.sort(test)
                        c4_test = _compute_c4(test)
                        if c4_test < best_c4:
                            best_c4 = c4_test
                            best_roots = test
                            improved = True
                            break
            if not improved:
                break

    # Final safety check
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
