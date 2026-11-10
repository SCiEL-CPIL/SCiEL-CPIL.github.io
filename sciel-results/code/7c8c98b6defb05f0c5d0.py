import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    x, w = np.polynomial.legendre.leggauss(70)
    phi = np.linspace(0, 2*np.pi, 130, endpoint=False)
    
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    dphi = 2*np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)
    
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0/(4*np.pi) * factorial(10-m)/factorial(10+m))
    
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    return Y, weights_2d

def _objective(coeffs, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    return norm4 / norm2

def _objective_and_grad(coeffs, Y, weights_2d):
    """Compute objective and correct analytic gradient."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    
    f3 = f**3
    f3Y = np.einsum('ij,mij->mij', f3, Y)
    fY = np.einsum('ij,mij->mij', f, Y)
    
    dnorm4_dc = np.einsum('mij,ij->m', f3Y, w) / (norm4**3)
    dnorm2_dc = np.einsum('mij,ij->m', fY, w) / norm2
    
    grad = dnorm4_dc / norm2 - (norm4 / norm2**2) * dnorm2_dc
    
    return norm4 / norm2, grad

def _fixed_point_iteration(c0, Y, weights_2d, n_iter=200, over_relax=1.0, tol=1e-12):
    """
    Solve the Euler-Lagrange self-consistency equation via fixed-point iteration.
    
    For L2-normalized f (||f||_2 = 1), the optimality condition is:
        f^3 = lambda * f  (pointwise on the sphere)
    which implies f takes values in {0, ±lambda^(1/2)}.
    
    More generally, for the L4/L2 ratio minimization, the condition is:
        f^3 = mu * f / ||f||_4^2  (up to a constant)
    
    We use the projection: given current f, compute new coefficients as
    c_new = Y^T * (f^3) / ||Y^T * f^3||_2 * ||c_old||_2
    with over-relaxation.
    """
    c = c0.copy()
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return c0
    
    for it in range(n_iter):
        f = np.tensordot(c, Y, axes=(0, 0))
        w = weights_2d
        
        # Compute f^3 weighted by quadrature weights
        f3 = f**3
        
        # Project f^3 onto the harmonic space: c_new_unnorm = Y^T * f3
        # c_new_unnorm[m] = sum_ij f3[i,j] * Y[m,i,j] * w[i,j]
        c_new_unnorm = np.einsum('ij,mij,ij->m', f3, Y, w)
        
        c_new_norm = np.linalg.norm(c_new_unnorm)
        if c_new_norm < 1e-15:
            break
        
        # Normalize to same norm as c
        c_new = c_new_unnorm / c_new_norm * c_norm
        
        # Over-relaxation
        c_next = (1 - over_relax) * c + over_relax * c_new
        
        # Check convergence
        delta = np.linalg.norm(c_next - c)
        c = c_next
        
        if delta < tol:
            break
    
    return c

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses fixed-point iteration on the Euler-Lagrange condition plus L-BFGS polish.
    """
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s limit
    
    Y, weights_2d = _build_harmonics()
    
    best_coeffs = np.ones(21) / np.sqrt(21)
    best_score = _objective(best_coeffs, Y, weights_2d)
    
    # Structured initial guesses
    structured_guesses = []
    
    # m=0 only
    c = np.zeros(21)
    c[10] = 1.0
    structured_guesses.append(c)
    
    # Even m cos only
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10+m] = 1.0
    structured_guesses.append(c)
    
    # All cos terms
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
    structured_guesses.append(c)
    
    # Symmetric: c[m] = c[-m] for all m
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    # Alternating signs on cos terms
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = (-1)**m
    structured_guesses.append(c)
    
    # m=0 dominant with small others
    c = np.zeros(21)
    c[10] = 1.0
    for m in [1, 2, 3, 4, 5]:
        c[10+m] = 0.1
        c[10-m] = 0.1
    structured_guesses.append(c)
    
    for c in structured_guesses:
        s = _objective(c, Y, weights_2d)
        if s < best_score and np.isfinite(s):
            best_score = s
            best_coeffs = c.copy()
    
    # Phase 1: Fixed-point iteration from multiple starts
    np.random.seed(42)
    n_fp_starts = 20
    
    for i in range(n_fp_starts):
        if time.time() - start_time > deadline - 120:
            break
        
        # Generate diverse starting points
        if i == 0:
            x0 = best_coeffs.copy()
        elif i < 5:
            x0 = np.random.randn(21)
        elif i < 10:
            # Sparse random
            x0 = np.random.randn(21)
            mask = np.random.random(21) < 0.3
            x0 *= mask
            if np.linalg.norm(x0) < 1e-10:
                x0 = np.random.randn(21)
        elif i < 15:
            # Perturb best solution
            x0 = best_coeffs + 0.3 * np.random.randn(21)
        else:
            # Structured random
            x0 = np.zeros(21)
            for m in range(11):
                if np.random.random() < 0.5:
                    x0[10+m] = np.random.randn()
                if m > 0 and np.random.random() < 0.5:
                    x0[10-m] = np.random.randn()
        
        x0 = x0 / np.linalg.norm(x0)
        
        # Try fixed-point with different over-relaxation parameters
        for omega in [0.5, 1.0, 1.5, 2.0]:
            if time.time() - start_time > deadline - 120:
                break
            try:
                c_fp = _fixed_point_iteration(x0, Y, weights_2d, n_iter=300, over_relax=omega, tol=1e-13)
                s = _objective(c_fp, Y, weights_2d)
                if s < best_score and np.isfinite(s):
                    best_score = s
                    best_coeffs = c_fp.copy()
            except Exception:
                pass
    
    # Phase 2: L-BFGS polish from best fixed-point result
    if time.time() - start_time < deadline - 60:
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                best_coeffs,
                jac=lambda c: _objective_and_grad(c, Y, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: More L-BFGS restarts from perturbed best
    n_lbfgs = 15
    for i in range(n_lbfgs):
        if time.time() - start_time > deadline - 30:
            break
        
        x0 = best_coeffs + 0.1 * np.random.randn(21)
        
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                x0,
                jac=lambda c: _objective_and_grad(c, Y, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                best_coeffs,
                jac=lambda c: _objective_and_grad(c, Y, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Validate
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
