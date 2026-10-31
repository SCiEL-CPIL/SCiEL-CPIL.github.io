"""Analytic Hermite-Gaussian uncertainty bound with generalized eigenvalue + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

_DEADLINE = time.time() + 195  # 15s safety margin before the 210s kill


def _bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product and its gradient with respect to (a0, a4, a8).
    a12 is determined by the constraint g(0) = 0: a12 = -(a0 + 3*a4 + 105*a8)/945.
    Key insight: vF = n2/n0 = 7/3 (constant), so bound = sqrt(7/3 * m2/m0).
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    g2 = 2.0 * a0 + 210.0 * a8
    sqrt_pi = np.sqrt(np.pi)

    m0_inner = a0 * a0 + 3.0 * a4 * a4 + 105.0 * a8 * a8 + 945.0 * a12 * a12
    m0 = sqrt_pi * m0_inner

    m2_inner = (a0 * (3.0 * a0 + 12.0 * a4)
                + a4 * (12.0 * a0 + 105.0 * a4 + 840.0 * a8)
                + a8 * (840.0 * a4 + 2940.0 * a8)
                + 11340.0 * a8 * a12)
    m2 = sqrt_pi * m2_inner

    n0 = 18.0 * np.pi * g2 * g2
    n2 = 42.0 * np.pi * g2 * g2

    if m0 <= 0.0 or n0 <= 0.0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([0.0, 0.0, 0.0])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0.0 or vF <= 0.0:
        return 1e18, np.array([0.0, 0.0, 0.0])

    bound = np.sqrt(vf * vF)

    # Gradient using quotient rule
    # d(m2/m0)/da_i = (dm2_da_i * m0 - m2 * dm0_da_i) / m0^2
    # d(bound)/da_i = 0.5/bound * (vF * d(m2/m0)/da_i + (m2/m0) * d(n2/n0)/da_i)
    # But vF = 7/3 is constant, so d(vF)/da_i = 0
    # Thus: grad = 0.5/bound * vF * d(m2/m0)/da_i
    
    dm0_da0 = sqrt_pi * (2.0 * a0 + 2.0 * 945.0 * a12 * (-1.0 / 945.0))
    dm0_da4 = sqrt_pi * (6.0 * a4 + 2.0 * 945.0 * a12 * (-3.0 / 945.0))
    dm0_da8 = sqrt_pi * (210.0 * a8 + 2.0 * 945.0 * a12 * (-105.0 / 945.0))

    dm2_inner_da0 = 6.0 * a0 + 24.0 * a4
    dm2_inner_da4 = 24.0 * a0 + 210.0 * a4 + 1680.0 * a8
    dm2_inner_da8 = 1680.0 * a4 + 4620.0 * a8 + 11340.0 * a12

    dm2_da0 = sqrt_pi * dm2_inner_da0
    dm2_da4 = sqrt_pi * dm2_inner_da4
    dm2_da8 = sqrt_pi * dm2_inner_da8

    # Since vF is constant, only the m2/m0 part contributes to the gradient
    dvf_da0 = (dm2_da0 * m0 - m2 * dm0_da0) / (m0 * m0)
    dvf_da4 = (dm2_da4 * m0 - m2 * dm0_da4) / (m0 * m0)
    dvf_da8 = (dm2_da8 * m0 - m2 * dm0_da8) / (m0 * m0)

    grad_da0 = 0.5 / bound * vF * dvf_da0
    grad_da4 = 0.5 / bound * vF * dvf_da4
    grad_da8 = 0.5 / bound * vF * dvf_da8

    return bound, np.array([grad_da0, grad_da4, grad_da8])


def _objective(params):
    val, _ = _bound_and_grad(params[0], params[1], params[2])
    return val if np.isfinite(val) else 1e18


def _objective_grad(params):
    val, grad = _bound_and_grad(params[0], params[1], params[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    return val, grad


def construct():
    """
    Key insight: vF = n2/n0 = 7/3 is constant (given g(0)=0 constraint).
    So bound = sqrt(7/3 * m2/m0). Minimizing m2/m0 is a Rayleigh quotient problem.
    Set up symmetric matrices A, B such that m0_inner = x^T A x, m2_inner = x^T B x.
    Solve B v = lambda A v; smallest eigenvalue gives the minimum ratio.
    
    The problem is scale-invariant, so we can fix the scale and optimize over the 2D sphere.
    """
    # Build symmetric matrices A and B
    # m0_inner = a0^2 + 3*a4^2 + 105*a8^2 + 945*a12^2
    # where a12 = -(a0+3*a4+105*a8)/945
    # 945*a12^2 = (a0+3*a4+105*a8)^2 / 945
    # Expanding: (a0^2 + 9*a4^2 + 11025*a8^2 + 6*a0*a4 + 210*a0*a8 + 630*a4*a8)/945

    A = np.array([
        [1.0 + 1.0/945.0,       3.0/945.0,       105.0/945.0],
        [3.0/945.0,              3.0 + 9.0/945.0, 315.0/945.0],
        [105.0/945.0,            315.0/945.0,     105.0 + 11025.0/945.0]
    ])

    # m2_inner = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1680*a4*a8 + 2940*a8^2 + 11340*a8*a12
    # 11340*a8*a12 = -12*a8*a0 - 36*a8*a4 - 1260*a8^2
    # m2_inner = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1644*a4*a8 + 1680*a8^2 - 12*a0*a8

    B = np.array([
        [3.0,     12.0,   -6.0],
        [12.0,    105.0,  822.0],
        [-6.0,    822.0,  1680.0]
    ])

    seed = np.array([-113.0/100, 1.0/25, 1.0/3240.0])

    # Solve generalized eigenvalue problem B v = lambda A v
    best = seed.copy()
    best_val, _ = _bound_and_grad(*seed)

    # Get all eigenvalue solutions
    eigenvalue_solutions = []
    try:
        eigenvalues, eigenvectors = eigh(B, A)
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            g2_v = 2.0 * v[0] + 210.0 * v[2]
            if abs(g2_v) > 1e-12:
                # Scale to unit norm
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-15:
                    v_scaled = v / norm_v
                    eigenvalue_solutions.append(v_scaled)
                    val_check, _ = _bound_and_grad(*v_scaled)
                    if np.isfinite(val_check) and val_check < best_val:
                        best_val = val_check
                        best = v_scaled.copy()
    except Exception:
        pass

    # Also try the seed scaled to unit norm
    norm_seed = np.linalg.norm(seed)
    if norm_seed > 1e-15:
        seed_unit = seed / norm_seed
        val_seed_unit, _ = _bound_and_grad(*seed_unit)
        if np.isfinite(val_seed_unit) and val_seed_unit < best_val:
            best_val = val_seed_unit
            best = seed_unit.copy()

    # Multi-start L-BFGS-B in direct coefficient space
    # Since the problem is scale-invariant, we can fix one coefficient or use unit norm
    rng = np.random.default_rng(42)
    
    # Generate multi-start points on the unit sphere in 3D
    direct_starts = [best.copy()]
    
    # Add eigenvalue solutions
    for v in eigenvalue_solutions:
        direct_starts.append(v.copy())
    
    # Add random points on the unit sphere
    for i in range(80):
        # Random direction in 3D
        rand_dir = rng.normal(0.0, 1.0, 3)
        norm_rd = np.linalg.norm(rand_dir)
        if norm_rd > 1e-15:
            rand_unit = rand_dir / norm_rd
            # Check g2 != 0
            g2_rand = 2.0 * rand_unit[0] + 210.0 * rand_unit[2]
            if abs(g2_rand) > 1e-12:
                direct_starts.append(rand_unit)
    
    # Add perturbations around the best solution
    for i in range(50):
        scale = 0.3 / (i + 1)
        perturbed = best + rng.normal(0.0, scale, 3)
        norm_p = np.linalg.norm(perturbed)
        if norm_p > 1e-15:
            perturbed_unit = perturbed / norm_p
            g2_p = 2.0 * perturbed_unit[0] + 210.0 * perturbed_unit[2]
            if abs(g2_p) > 1e-12:
                direct_starts.append(perturbed_unit)

    # Optimize in direct coefficient space with unit norm constraint handled by scaling
    for s in direct_starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                           options={'maxiter': 10000, 'gtol': 1e-15, 'ftol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # High-precision polish with tighter tolerance
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                           options={'maxiter': 50000, 'gtol': 1e-18, 'ftol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # Even tighter polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                           options={'maxiter': 100000, 'gtol': 1e-20, 'ftol': 1e-20})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # Final verification
    if time.time() < _DEADLINE - 2:
        val_final, _ = _bound_and_grad(*best)
        if np.isfinite(val_final) and val_final < best_val:
            best_val = val_final

    return np.array(best)
# EVOLVE-BLOCK-END
