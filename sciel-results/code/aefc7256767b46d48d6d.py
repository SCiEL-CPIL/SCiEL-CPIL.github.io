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

def _compute_f(c, Y):
    return np.tensordot(c, Y, axes=(0, 0))

def _objective(c, Y, w):
    f = _compute_f(c, Y)
    n2sq = np.sum(f**2 * w)
    if n2sq < 1e-30:
        return 1e10
    n2 = np.sqrt(n2sq)
    n4 = np.sum(f**4 * w)**0.25
    return n4 / n2

def _objective_grad(c, Y, w):
    f = _compute_f(c, Y)
    n2sq = np.sum(f**2 * w)
    if n2sq < 1e-30:
        return 1e10, np.zeros_like(c)
    n2 = np.sqrt(n2sq)
    n4 = np.sum(f**4 * w)**0.25
    f3 = f**3
    dnorm4 = np.einsum('ij,mij,ij->m', f3, Y, w) / (n4**3)
    dnorm2 = np.einsum('ij,mij,ij->m', f, Y, w) / n2
    grad = dnorm4 / n2 - (n4 / n2**2) * dnorm2
    return n4/n2, grad

def _sphere_project(c, Y, w, max_iter=5000, tol=1e-14, deadline=None, start_time=None):
    """
    Sphere-constrained gradient descent: minimize R(c) subject to ||c||=const.
    Uses backtracking line search on the sphere.
    """
    c = c.copy()
    cnorm = np.linalg.norm(c)
    if cnorm < 1e-15:
        return c
    c = c / cnorm
    
    best_f = _objective(c, Y, w)
    best_c = c.copy()
    
    for it in range(max_iter):
        if deadline is not None:
            if time.time() > deadline:
                break
        
        f = _objective(c, Y, w)
        if f < best_f:
            best_f = f
            best_c = c.copy()
        
        f, grad = _objective_grad(c, Y, w)
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        
        # Project gradient to tangent space
        proj = grad - np.dot(grad, c) * c
        pnorm = np.linalg.norm(proj)
        if pnorm < tol * 0.1:
            break
        p = proj / pnorm
        
        # Backtracking line search
        alpha = 0.01
        c_cur = c
        f_cur = f
        improved = False
        for bt in range(30):
            # Move on sphere
            c_new = c_cur + alpha * p
            c_new = c_new / np.linalg.norm(c_new)
            f_new = _objective(c_new, Y, w)
            if f_new < f_cur - 1e-12:
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
        
        c = c_new
    
    # Restore original norm
    result = best_c * cnorm
    return result

def _fixed_point(c0, Y, w, n_iter=500, omega=1.0, tol=1e-13):
    c = c0.copy()
    cnorm = np.linalg.norm(c)
    if cnorm < 1e-15:
        return c0
    
    for it in range(n_iter):
        f = _compute_f(c, Y)
        f3 = f**3
        c_new_un = np.einsum('ij,mij,ij->m', f3, Y, w)
        cn = np.linalg.norm(c_new_un)
        if cn < 1e-15:
            break
        c_new = c_new_un / cn * cnorm
        c_next = (1 - omega) * c + omega * c_new
        delta = np.linalg.norm(c_next - c)
        c = c_next
        if delta < tol:
            break
    return c

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0
    
    Y, w = _build_harmonics()
    
    best_c = np.ones(21) / np.sqrt(21)
    best_score = _objective(best_c, Y, w)
    
    # Structured starts
    starts = []
    c = np.zeros(21); c[10] = 1.0; starts.append(c)
    c = np.zeros(21)
    for m in [0,2,4,6,8,10]: c[10+m] = 1.0
    starts.append(c)
    c = np.zeros(21)
    for m in range(11): c[10+m] = 1.0
    starts.append(c)
    c = np.zeros(21)
    for m in range(11):
        c[10+m] = 1.0
        if m > 0: c[10-m] = 1.0
    starts.append(c)
    c = np.zeros(21)
    for m in range(11): c[10+m] = (-1)**m
    starts.append(c)
    c = np.zeros(21); c[10] = 1.0
    for m in [1,2,3,4,5]:
        c[10+m] = 0.1; c[10-m] = 0.1
    starts.append(c)
    
    np.random.seed(123)
    for i in range(10):
        x = np.random.randn(21)
        starts.append(x / np.linalg.norm(x))
    for i in range(10):
        x = np.random.randn(21)
        mask = np.random.random(21) < 0.4
        x *= mask
        if np.linalg.norm(x) < 1e-10:
            x = np.random.randn(21)
        starts.append(x / np.linalg.norm(x))
    
    for s in starts:
        sc = _objective(s, Y, w)
        if sc < best_score and np.isfinite(sc):
            best_score = sc
            best_c = s.copy()
    
    # Phase 1: Fixed-point + sphere polish
    for i in range(25):
        if time.time() - start_time > deadline - 180:
            break
        
        if i == 0:
            x0 = best_c.copy()
        elif i < 5:
            x0 = np.random.randn(21)
        elif i < 10:
            x0 = np.random.randn(21)
            mask = np.random.random(21) < 0.3
            x0 *= mask
            if np.linalg.norm(x0) < 1e-10:
                x0 = np.random.randn(21)
        elif i < 15:
            x0 = best_c + 0.3 * np.random.randn(21)
        else:
            x0 = np.zeros(21)
            for m in range(11):
                if np.random.random() < 0.5:
                    x0[10+m] = np.random.randn()
                if m > 0 and np.random.random() < 0.5:
                    x0[10-m] = np.random.randn()
        
        x0 = x0 / np.linalg.norm(x0)
        
        for om in [0.5, 1.0, 1.5, 2.0]:
            if time.time() - start_time > deadline - 180:
                break
            try:
                c_fp = _fixed_point(x0, Y, w, n_iter=300, omega=om)
                sc = _objective(c_fp, Y, w)
                if sc < best_score and np.isfinite(sc):
                    best_score = sc
                    best_c = c_fp.copy()
            except Exception:
                pass
    
    # Phase 2: L-BFGS polish
    if time.time() - start_time < deadline - 120:
        try:
            res = minimize(
                lambda c: _objective(c, Y, w),
                best_c,
                jac=lambda c: _objective_grad(c, Y, w)[1],
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Sphere-constrained polish
    if time.time() - start_time < deadline - 60:
        try:
            polished = _sphere_project(best_c, Y, w, max_iter=10000, tol=1e-14,
                                        deadline=deadline-55, start_time=start_time)
            sc = _objective(polished, Y, w)
            if sc < best_score and np.isfinite(sc):
                best_score = sc
                best_c = polished.copy()
        except Exception:
            pass
    
    # Phase 4: More L-BFGS restarts
    for i in range(15):
        if time.time() - start_time > deadline - 30:
            break
        x0 = best_c + 0.1 * np.random.randn(21)
        try:
            res = minimize(
                lambda c: _objective(c, Y, w),
                x0,
                jac=lambda c: _objective_grad(c, Y, w)[1],
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final sphere polish
    if time.time() - start_time < deadline - 15:
        try:
            polished = _sphere_project(best_c, Y, w, max_iter=20000, tol=1e-15,
                                        deadline=deadline-10, start_time=start_time)
            sc = _objective(polished, Y, w)
            if sc < best_score and np.isfinite(sc):
                best_score = sc
                best_c = polished.copy()
        except Exception:
            pass
    
    # Final L-BFGS high precision
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                lambda c: _objective(c, Y, w),
                best_c,
                jac=lambda c: _objective_grad(c, Y, w)[1],
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Validate
    if len(best_c) != 21 or not np.all(np.isfinite(best_c)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs < 1e-3 and max_abs > 0:
            best_c *= 1.0 / max_abs
        elif max_abs > 1e5:
            best_c *= 1e5 / max_abs
        if not np.all(np.isfinite(best_c)):
            return np.ones(21)
    
    return best_c.astype(np.float64)
