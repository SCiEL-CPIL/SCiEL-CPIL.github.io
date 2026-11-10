"""Direct geometric construction for logarithmic energy on sphere."""
import numpy as np
from math import fsum, log
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0
SAFETY = 0.85


def _energy(pts):
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return 1e18
            terms.append(-log(d))
    return fsum(terms)


def _cartesian_energy_and_grad(X):
    """Energy and gradient for points on sphere (Cartesian coords, projected)."""
    # Normalize to ensure we're on the sphere
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X = X / norms
    n = len(X)
    diff = X[:, None, :] - X[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1e-30)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    E = -0.5 * np.sum(np.log(r))
    # dE/dX_i = sum_{j!=i} (X_i - X_j) / r_ij^2
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    dE_dX = np.sum(diff * inv_r2[:, :, None], axis=1)
    # Project gradient to tangent space: g_tan = g - (g·x)x
    g_tan = dE_dX - np.sum(dE_dX * X, axis=1, keepdims=True) * X
    return E, g_tan


def _opt_cartesian(X0, maxiter=1000, ftol=1e-14, gtol=1e-12):
    """Optimize in Cartesian coords with projection."""
    try:
        res = minimize(_cartesian_energy_and_grad, X0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                'maxls': 50})
        return res.x
    except Exception:
        return X0


def _riemannian_cg(X, maxiter=500, tol=1e-14, time_limit=None):
    """
    Riemannian conjugate gradient on the sphere.
    Uses geodesic line search (retraction via normalization).
    """
    X = X.copy()
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(X)
    
    # Compute initial energy and gradient
    E, g = _cartesian_energy_and_grad(X)
    g = g / np.linalg.norm(g, axis=1, keepdims=True)  # normalize for stability
    
    # Riemannian conjugate gradient
    d = -g  # initial direction
    beta = 0.0
    
    for it in range(maxiter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Line search: minimize E(retract(X, alpha*d))
        # Use backtracking
        alpha = 1.0
        E_new = 1e18
        X_new = X.copy()
        
        # Try a few step sizes
        for _ in range(30):
            X_trial = X + alpha * d
            X_trial = X_trial / np.linalg.norm(X_trial, axis=1, keepdims=True)
            E_trial, _ = _cartesian_energy_and_grad(X_trial)
            if E_trial < E:
                E_new = E_trial
                X_new = X_trial
                break
            alpha *= 0.5
        
        if E_new >= E:
            # Couldn't find a better step, break
            break
        
        # Check convergence
        E, g = _cartesian_energy_and_grad(X_new)
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            X = X_new
            break
        
        # Compute beta (Riemannian Fletcher-Reeves)
        d_old = d.copy()
        d = -g
        beta = np.sum(g * g) / max(np.sum(d_old * d_old), 1e-30)
        d = d + beta * d_old
        
        # Project d to tangent space
        d = d - np.sum(d * X, axis=1, keepdims=True) * X
        
        X = X_new
    
    return X


def construct():
    start = time.time()
    dl = start + DEADLINE * SAFETY

    def _fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1 - z ** 2))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    def _random_init():
        X = np.random.randn(N, 3)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    best = None
    best_E = 1e18

    # Try multiple initializations
    np.random.seed(42)
    initializations = [_fibonacci_spiral()]
    for _ in range(8):
        initializations.append(_random_init())

    # Phase 1: L-BFGS-B from each initialization
    for X0 in initializations:
        if time.time() > dl:
            break
        Xopt = _opt_cartesian(X0, maxiter=800, ftol=1e-14, gtol=1e-12)
        Xopt /= np.linalg.norm(Xopt, axis=1, keepdims=True)
        E = _energy(Xopt)
        if E < best_E:
            best_E = E
            best = Xopt.copy()

    if best is None:
        best = _fibonacci_spiral()

    # Phase 2: Riemannian CG refinement from the best solution
    if time.time() < dl - 10.0:
        Xref = _riemannian_cg(best, maxiter=300, tol=1e-14, time_limit=dl - 5.0)
        Xref /= np.linalg.norm(Xref, axis=1, keepdims=True)
        Eref = _energy(Xref)
        if Eref < best_E:
            best_E = Eref
            best = Xref

    # Phase 3: Final high-precision L-BFGS-B polish
    if time.time() < dl - 5.0:
        Xref = _opt_cartesian(best, maxiter=500, ftol=1e-16, gtol=1e-14)
        Xref /= np.linalg.norm(Xref, axis=1, keepdims=True)
        Eref = _energy(Xref)
        if Eref < best_E:
            best = Xref

    # Final normalization
    best = best / np.linalg.norm(best, axis=1, keepdims=True)

    return best
