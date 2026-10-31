"""Analytic Hermite-Gaussian uncertainty bound via secular equation (cubic) + L-BFGS-B polish."""
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

    dm0_da0 = sqrt_pi * (2.0 * a0 + 2.0 * 945.0 * a12 * (-1.0 / 945.0))
    dm0_da4 = sqrt_pi * (6.0 * a4 + 2.0 * 945.0 * a12 * (-3.0 / 945.0))
    dm0_da8 = sqrt_pi * (210.0 * a8 + 2.0 * 945.0 * a12 * (-105.0 / 945.0))

    dm2_inner_da0 = 6.0 * a0 + 24.0 * a4
    dm2_inner_da4 = 24.0 * a0 + 210.0 * a4 + 1680.0 * a8
    dm2_inner_da8 = 1680.0 * a4 + 4620.0 * a8 + 11340.0 * a12

    dm2_da0 = sqrt_pi * dm2_inner_da0
    dm2_da4 = sqrt_pi * dm2_inner_da4
    dm2_da8 = sqrt_pi * dm2_inner_da8

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


def _solve_cubic(a, b, c, d):
    """
    Solve a*x^3 + b*x^2 + c*x + d = 0 for real roots.
    Uses numpy.roots for robustness.
    """
    coeffs = np.array([a, b, c, d], dtype=np.float64)
    roots = np.roots(coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 * max(1.0, abs(r.real)):
            real_roots.append(r.real)
    return real_roots


def construct():
    """
    Key insight: vF = n2/n0 = 7/3 is constant (given g(0)=0 constraint).
    So bound = sqrt(7/3 * m2/m0). Minimizing m2/m0 is a Rayleigh quotient problem.
    
    The secular equation is det(B - lambda*A) = 0, a cubic in lambda.
    We solve it analytically, then for each real positive root lambda, we find
    the corresponding eigenvector v (null space of B - lambda*A), verify g2 != 0,
    and compute the actual bound. The minimum bound across all valid roots is optimal.
    
    Since the problem is scale-invariant, we return the coefficients directly.
    """
    # Build symmetric matrices A and B
    # m0_inner = a0^2 + 3*a4^2 + 105*a8^2 + 945*a12^2
    # where a12 = -(a0+3*a4+105*a8)/945
    # 945*a12^2 = (a0+3*a4+105*a8)^2 / 945

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
    best = seed.copy()
    best_val, _ = _bound_and_grad(*seed)

    # === Secular equation approach: det(B - lambda*A) = 0 ===
    # This is a cubic in lambda. Compute coefficients.
    # det(B - lambda*A) = det(B) - lambda*tr(adj(B)*A) + lambda^2*tr(adj(A)*B) - lambda^3*det(A)
    # More carefully, for 3x3: det(M) where M = B - lambda*A
    # Use the formula: det(B - lambda*A) = sum over all minors
    
    # Let's compute it directly by expanding
    # det(B - lambda*A) for 3x3:
    # = (B00-lA00)*[(B11-lA11)*(B22-lA22) - (B12-lA12)*(B21-lA21)]
    #   - (B01-lA01)*[(B10-lA10)*(B22-lA22) - (B12-lA12)*(B20-lA20)]
    #   + (B02-lA02)*[(B10-lA10)*(B21-lA21) - (B11-lA11)*(B20-lA20)]
    
    # Coefficients of lambda^3, lambda^2, lambda^1, lambda^0
    # lambda^3: -det(A)
    # lambda^2: sum of 2x2 minors of A with one row/col from B... 
    # Actually let me just compute numerically at several points and fit, or use the direct formula.
    
    # Direct computation of det(B - lambda*A) as polynomial in lambda:
    # For 3x3, det(B - lambda*A) is cubic. Let me compute coefficients directly.
    
    a00, a01, a02 = A[0,0], A[0,1], A[0,2]
    a10, a11, a12 = A[1,0], A[1,1], A[1,2]
    a20, a21, a22 = A[2,0], A[2,1], A[2,2]
    b00, b01, b02 = B[0,0], B[0,1], B[0,2]
    b10, b11, b12 = B[1,0], B[1,1], B[1,2]
    b20, b21, b22 = B[2,0], B[2,1], B[2,2]
    
    # M = B - lambda*A
    # M[0,0] = b00 - lambda*a00, etc.
    
    # det(M) = M00*(M11*M22 - M12*M21) - M01*(M10*M22 - M12*M20) + M02*(M10*M21 - M11*M20)
    
    # Expand each term:
    # Term1 = (b00 - l*a00) * [(b11-l*a11)(b22-l*a22) - (b12-l*a12)(b21-l*a21)]
    # Term2 = (b01 - l*a01) * [(b10-l*a10)(b22-l*a22) - (b12-l*a12)(b20-l*a20)]
    # Term3 = (b02 - l*a02) * [(b10-l*a10)(b21-l*a21) - (b11-l*a11)(b20-l*a20)]
    
    # Each (bi - l*ai)(bj - l*aj) = bi*bj - l*(bi*aj + ai*bj) + l^2*ai*aj
    
    # Let me compute the polynomial coefficients by evaluating at enough points and interpolating,
    # or just use np.poly on the symbolic expansion.
    
    # Simpler: evaluate det(B - lambda*A) at lambda = 0, 1, 2, 3 and use Lagrange interpolation
    # to get the cubic coefficients.
    
    def _det_at(lam):
        M = B - lam * A
        return np.linalg.det(M)
    
    # Evaluate at 4 points to determine the cubic
    lam_pts = np.array([0.0, 1.0, 2.0, 3.0])
    det_vals = np.array([_det_at(l) for l in lam_pts])
    
    # Fit cubic: det(lam) = c3*lam^3 + c2*lam^2 + c1*lam + c0
    # Using Vandermonde:
    V = np.vander(lam_pts, 4)  # columns: lam^3, lam^2, lam^1, lam^0
    try:
        poly_coeffs = np.linalg.solve(V, det_vals)  # [c3, c2, c1, c0]
        c3, c2, c1, c0 = poly_coeffs
    except np.linalg.LinAlgError:
        # Fallback to numpy.roots on the matrix
        poly_coeffs = np.array([0.0, 0.0, 0.0, 0.0])
        c3, c2, c1, c0 = 0.0, 0.0, 0.0, 0.0
    
    # Find real roots of c3*lam^3 + c2*lam^2 + c1*lam + c0 = 0
    eigenvalue_candidates = []
    try:
        real_roots = _solve_cubic(c3, c2, c1, c0)
        eigenvalue_candidates = [r for r in real_roots if r > 1e-15]
    except Exception:
        pass
    
    # Also use scipy.linalg.eigh as a cross-check / backup
    try:
        eigvals_scipy, eigvecs_scipy = eigh(B, A)
        for i in range(len(eigvals_scipy)):
            if eigvals_scipy[i] > 1e-15:
                eigenvalue_candidates.append(eigvals_scipy[i])
    except Exception:
        pass
    
    # For each candidate eigenvalue, find the eigenvector and compute the bound
    candidate_solutions = []
    for lam in eigenvalue_candidates:
        # Find null space of (B - lam*A)
        M = B - lam * A
        try:
            # SVD to find null space
            U, s, Vt = np.linalg.svd(M)
            # The right singular vector corresponding to smallest singular value
            v = Vt[-1, :]
            # Check g2 != 0
            g2_v = 2.0 * v[0] + 210.0 * v[2]
            if abs(g2_v) > 1e-12:
                # Verify this is actually an eigenpair
                residual = np.linalg.norm(M @ v)
                if residual < 1e-8 * (np.linalg.norm(M) * np.linalg.norm(v) + 1e-15):
                    # Normalize
                    norm_v = np.linalg.norm(v)
                    if norm_v > 1e-15:
                        v_unit = v / norm_v
                        candidate_solutions.append((lam, v_unit))
        except Exception:
            continue
    
    # Also try scipy eigenvectors directly
    try:
        eigvals_scipy, eigvecs_scipy = eigh(B, A)
        for i in range(len(eigvals_scipy)):
            v = eigvecs_scipy[:, i]
            g2_v = 2.0 * v[0] + 210.0 * v[2]
            if abs(g2_v) > 1e-12:
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-15:
                    v_unit = v / norm_v
                    candidate_solutions.append((eigvals_scipy[i], v_unit))
    except Exception:
        pass
    
    # Evaluate bound for each candidate
    for lam, v in candidate_solutions:
        val_check, _ = _bound_and_grad(*v)
        if np.isfinite(val_check) and val_check < best_val:
            best_val = val_check
            best = v.copy()
    
    # Also try the seed
    norm_seed = np.linalg.norm(seed)
    if norm_seed > 1e-15:
        seed_unit = seed / norm_seed
        val_seed_unit, _ = _bound_and_grad(*seed_unit)
        if np.isfinite(val_seed_unit) and val_seed_unit < best_val:
            best_val = val_seed_unit
            best = seed_unit.copy()
    
    # L-BFGS-B polish from the best candidate
    if time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                           options={'maxiter': 50000, 'gtol': 1e-16, 'ftol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass
    
    # Second polish pass with even tighter tolerance
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                           options={'maxiter': 100000, 'gtol': 1e-18, 'ftol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass
    
    # Multi-start from multiple eigenvalue solutions to ensure global optimum
    if time.time() < _DEADLINE - 20:
        for lam, v in candidate_solutions[:10]:
            if time.time() > _DEADLINE - 20:
                break
            try:
                res = minimize(_objective_grad, v, method='L-BFGS-B',
                               options={'maxiter': 20000, 'gtol': 1e-16, 'ftol': 1e-16})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best = res.x.copy()
            except Exception:
                continue
    
    # Final verification
    if time.time() < _DEADLINE - 5:
        val_final, _ = _bound_and_grad(*best)
        if np.isfinite(val_final) and val_final < best_val:
            best_val = val_final

    return np.array(best)
# EVOLVE-BLOCK-END
