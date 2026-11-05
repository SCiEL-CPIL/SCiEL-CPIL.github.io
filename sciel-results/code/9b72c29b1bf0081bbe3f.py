"""Gas-equilibrium representation: 12 double roots as critical points of a 1D gas
with repulsive 1/(x_i-x_j)^2 potential and logarithmic confinement V(x)=a*ln(x).
Optimize single scalar parameter a to minimize C4."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar, newton
from scipy.special import roots_genlaguerre, roots_laguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

# Precompute factorials
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _compute_c4(roots):
    """Compute C4 = I2 / I1^2 where P = prod (x-x_i)^2."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf
    return float(I2 / (I1 * I1))

def _compute_c4_grad(roots):
    """Compute C4 and its gradient with respect to roots."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, np.zeros_like(roots)
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf, np.zeros_like(roots)
    c4 = I2 / (I1 * I1)
    
    B3 = polymul(polymul(B, B), B)
    grad = np.zeros(n)
    for i in range(n):
        R_i = np.array([1.0])
        for j in range(n):
            if j != i:
                R_i = polymul(R_i, np.array([-roots[j], 1.0]))
        BR_i = polymul(B, R_i)
        dI1_i = -2.0 * _poly_moment(BR_i)
        B3R_i = polymul(B3, R_i)
        dI2_i = -4.0 * _poly_moment(B3R_i)
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    return float(c4), grad

def _gas_equilibrium(roots, a):
    """Compute equilibrium forces for the gas model.
    Force on particle i: F_i = -dU/dx_i
    U = sum_{i<j} 1/(x_i-x_j)^2 + a*sum_i ln(x_i)
    dU/dx_i = -2*sum_{j!=i} 1/(x_i-x_j)^3 + a/x_i
    Equilibrium: F_i = 2*sum_{j!=i} 1/(x_i-x_j)^3 - a/x_i = 0
    """
    n = len(roots)
    F = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j != i:
                d = roots[i] - roots[j]
                s += 1.0 / (d * d * d)
        F[i] = 2.0 * s - a / roots[i]
    return F

def _gas_equilibrium_jac(roots, a):
    """Jacobian of equilibrium forces w.r.t. roots."""
    n = len(roots)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                # dF_i/dx_i = 2*sum_{k!=i} 6/(x_i-x_k)^4 + a/x_i^2
                s = 0.0
                for k in range(n):
                    if k != i:
                        d = roots[i] - roots[k]
                        s += 6.0 / (d ** 4)
                J[i, i] = 2.0 * s + a / (roots[i] * roots[i])
            else:
                # dF_i/dx_j = 2 * d/dx_j [1/(x_i-x_j)^3] = 2 * 3/(x_i-x_j)^4
                d = roots[i] - roots[j]
                J[i, j] = 6.0 / (d ** 4)
    return J

def _solve_gas_equilibrium(a, x0, max_iter=200, tol=1e-14):
    """Solve for roots satisfying gas equilibrium conditions using Newton-Raphson."""
    roots = np.array(x0, dtype=np.float64).copy()
    n = len(roots)
    
    for it in range(max_iter):
        F = _gas_equilibrium(roots, a)
        if np.max(np.abs(F)) < tol:
            return roots, True
        
        J = _gas_equilibrium_jac(roots, a)
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # Regularize
            delta = np.linalg.solve(J + 1e-10 * np.eye(n), -F)
        
        roots_new = roots + delta
        # Ensure positivity
        if np.any(roots_new <= 1e-10):
            # Damped step
            for damp in [0.5, 0.25, 0.1, 0.05]:
                roots_try = roots + damp * delta
                if np.all(roots_try > 1e-10):
                    roots_new = roots_try
                    break
            else:
                roots_new = np.maximum(roots + delta, 1e-10)
        
        # Sort to maintain ordering
        roots_new = np.sort(roots_new)
        roots = roots_new
    
    return roots, False

def _c4_from_gas(a, x0):
    """Compute C4 for gas equilibrium roots at parameter a."""
    try:
        roots, converged = _solve_gas_equilibrium(a, x0)
        if not np.all(roots > 0) or not np.all(np.isfinite(roots)):
            return np.inf
        return _compute_c4(roots)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 30.0
    
    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Initial guess: standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    best_roots = x0.copy()
    best_c4 = _compute_c4(best_roots)
    
    # Phase 1: 1D search over gas parameter a
    # The gas parameter a controls the density profile
    # Search over a range of a values
    best_a = 1.0
    best_c4_gas = np.inf
    
    # Log-spaced grid search
    log_ais = np.linspace(math.log(0.01), math.log(1000.0), 200)
    for la in log_ais:
        if not time_ok():
            break
        a_val = np.exp(la)
        c4_val = _c4_from_gas(a_val, x0)
        if c4_val < best_c4_gas:
            best_c4_gas = c4_val
            best_a = a_val
    
    # Refine with scalar minimization
    if time_ok():
        try:
            res = minimize_scalar(
                lambda la: _c4_from_gas(np.exp(la), x0),
                bounds=(math.log(max(best_a * 0.5, 0.001)), math.log(min(best_a * 2.0, 5000.0))),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 300}
            )
            if res.success and np.isfinite(res.fun) and res.fun < best_c4_gas:
                best_a = np.exp(res.x)
                best_c4_gas = res.fun
        except Exception:
            pass
    
    # Get gas equilibrium roots at best a
    gas_roots, _ = _solve_gas_equilibrium(best_a, x0)
    if np.all(gas_roots > 0) and np.all(np.isfinite(gas_roots)):
        c4_gas = _compute_c4(gas_roots)
        if c4_gas < best_c4:
            best_c4 = c4_gas
            best_roots = gas_roots.copy()
    
    # Phase 2: Full 12-DOF L-BFGS-B refinement from gas equilibrium point
    if time_ok():
        try:
            res = minimize(
                lambda r: _compute_c4(r),
                best_roots,
                jac=lambda r: _compute_c4_grad(r)[1],
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Try multiple gas parameter initializations
    if time_ok():
        for a_try in [best_a * 0.1, best_a * 0.3, best_a * 3.0, best_a * 10.0,
                      0.05, 0.5, 5.0, 50.0, 500.0]:
            if not time_ok():
                break
            try:
                roots_try, _ = _solve_gas_equilibrium(a_try, x0)
                if np.all(roots_try > 0) and np.all(np.isfinite(roots_try)):
                    c4_try = _compute_c4(roots_try)
                    if c4_try < best_c4:
                        best_c4 = c4_try
                        best_roots = roots_try.copy()
                    # Also refine with L-BFGS-B
                    res = minimize(
                        lambda r: _compute_c4(r),
                        roots_try,
                        jac=lambda r: _compute_c4_grad(r)[1],
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
                    )
                    if np.all(res.x > 0) and np.isfinite(res.fun):
                        candidate = np.sort(res.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
            except Exception:
                pass
    
    # Phase 4: Multi-start L-BFGS-B from perturbed points
    if time_ok():
        rng = np.random.default_rng(42)
        for scale in [0.05, 0.1, 0.2, 0.3]:
            for _ in range(3):
                if not time_ok():
                    break
                pert = best_roots * (1.0 + scale * rng.standard_normal(12))
                pert = np.maximum(pert, 1e-3)
                pert = np.sort(pert)
                try:
                    res = minimize(
                        lambda r: _compute_c4(r),
                        pert,
                        jac=lambda r: _compute_c4_grad(r)[1],
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
                    )
                    if np.all(res.x > 0) and np.isfinite(res.fun):
                        candidate = np.sort(res.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass
    
    # Phase 5: Final high-precision polish
    for polish_iter in range(5):
        if not time_ok():
            break
        try:
            res = minimize(
                lambda r: _compute_c4(r),
                best_roots,
                jac=lambda r: _compute_c4_grad(r)[1],
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break
    
    # Final safety check
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
