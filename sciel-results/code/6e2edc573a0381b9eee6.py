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
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(X)
    diff = X[:, None, :] - X[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e-30)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    E = -np.sum(np.log(r)) * 0.5
    # dE/dX_i = sum_{j!=i} (X_i - X_j) / r_ij^2
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    dE_dX = np.sum(diff * inv_r2[:, :, None], axis=1)
    # Project gradient to tangent space: g_tan = g - (g·x)x
    g_tan = dE_dX - np.sum(dE_dX * X, axis=1, keepdims=True) * X
    return E, g_tan


def _riemannian_cg(X0, maxiter=2000, tol=1e-12):
    """Riemannian conjugate gradient with line search on the sphere."""
    X = X0.copy()
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    E, g = _cartesian_energy_and_grad(X)
    
    # Initial search direction: negative projected gradient
    p = -g
    
    beta = 0.0
    for it in range(maxiter):
        if np.linalg.norm(p) < tol:
            break
        
        # Line search: find step alpha that minimizes E along geodesic
        best_alpha = 1.0
        best_E = E
        alphas = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
        for a in alphas:
            # Geodesic step: rotate by angle a*||p||
            theta = a * np.linalg.norm(p, axis=1)
            # For small theta, use first-order approx; for larger, use rotation
            # Simple approach: X_new = (X + a*p) normalized (Euler step on sphere)
            X_new = X + a * p
            X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
            E_new, _ = _cartesian_energy_and_grad(X_new)
            if E_new < best_E:
                best_E = E_new
                best_alpha = a
            if E_new < E - 1e-10:  # Sufficient decrease
                break
        
        if best_E >= E - 1e-14:
            # No improvement, try smaller steps
            for a in [0.005, 0.001, 0.0001]:
                X_new = X + a * p
                X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
                E_new, _ = _cartesian_energy_and_grad(X_new)
                if E_new < E:
                    best_E = E_new
                    best_alpha = a
                    break
            if best_E >= E - 1e-14:
                break
        
        X = X + best_alpha * p
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        E = best_E
        _, g_new = _cartesian_energy_and_grad(X)
        
        # Check convergence
        if np.linalg.norm(g_new) < tol:
            p = -g_new
            beta = 0.0
            continue
        
        # Polak-Ribiere beta
        pg = np.sum(g * g_new)
        beta = max(0.0, np.sum(g_new * (g_new - g)) / (np.sum(g * g) + 1e-30))
        p = -g_new + beta * p
        
        # Project p to tangent space
        p = p - np.sum(p * X, axis=1, keepdims=True) * X
    
    return X


def _opt_cartesian(X0, maxiter=1000):
    """Optimize in Cartesian coords with projection."""
    try:
        res = minimize(_cartesian_energy_and_grad, X0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x
    except Exception:
        return X0


def _polish(X0, maxiter=5000, tol=1e-14):
    """High-precision polish using Riemannian CG with tight tolerance."""
    X = X0.copy()
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    E, g = _cartesian_energy_and_grad(X)
    
    p = -g
    beta = 0.0
    
    for it in range(maxiter):
        if np.linalg.norm(p) < tol:
            break
        
        # More careful line search
        best_alpha = 1.0
        best_E = E
        # Adaptive step sizes
        alphas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        for a in alphas:
            X_new = X + a * p
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            E_new, _ = _cartesian_energy_and_grad(X_new)
            if E_new < best_E:
                best_E = E_new
                best_alpha = a
            if best_E < E - 1e-12:
                break
        
        if best_E >= E - 1e-16:
            break
        
        X = X + best_alpha * p
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        E = best_E
        _, g_new = _cartesian_energy_and_grad(X)
        
        if np.linalg.norm(g_new) < tol:
            break
        
        beta = max(0.0, np.sum(g_new * (g_new - g)) / (np.sum(g * g) + 1e-30))
        p = -g_new + beta * p
        p = p - np.sum(p * X, axis=1, keepdims=True) * X
    
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

    def _random_points(seed):
        rng = np.random.RandomState(seed)
        X = rng.randn(N, 3)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def _icosahedron_based():
        """Generate points based on icosahedron symmetry with jitter."""
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        # Icosahedron vertices
        ico = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        
        # Create N points by subdividing and adding jitter
        rng = np.random.RandomState(123)
        pts = []
        for i in range(N):
            # Pick nearest icosahedron vertex and add random perturbation
            idx = i % 12
            p = ico[idx] + 0.1 * rng.randn(3)
            pts.append(p / np.linalg.norm(p))
        return np.array(pts)

    best = None
    best_E = 1e18

    # Try multiple initializations
    initializations = [_fibonacci_spiral()]
    for seed in [42, 123, 456, 789, 1000, 2024]:
        initializations.append(_random_points(seed))
    initializations.append(_icosahedron_based())

    for X0 in initializations:
        if time.time() > dl:
            break
        
        # First pass: L-BFGS-B
        Xopt = _opt_cartesian(X0, maxiter=800)
        Xopt = Xopt / np.linalg.norm(Xopt, axis=1, keepdims=True)
        E = _energy(Xopt)
        if E < best_E:
            best_E = E
            best = Xopt.copy()
        
        # Second pass: Riemannian CG from this result
        if time.time() < dl - 10:
            Xcg = _riemannian_cg(Xopt, maxiter=1000, tol=1e-12)
            Xcg = Xcg / np.linalg.norm(Xcg, axis=1, keepdims=True)
            Ecg = _energy(Xcg)
            if Ecg < best_E:
                best_E = Ecg
                best = Xcg.copy()

    if best is None:
        best = _fibonacci_spiral()

    # Final refinement with tighter tolerances if time permits
    if time.time() < dl - 5.0:
        Xref = _opt_cartesian(best, maxiter=500)
        Xref = Xref / np.linalg.norm(Xref, axis=1, keepdims=True)
        Eref = _energy(Xref)
        if Eref < best_E:
            best_E = Eref
            best = Xref

    # High-precision polish
    if time.time() < dl - 2.0:
        Xpol = _polish(best, maxiter=3000, tol=1e-14)
        Xpol = Xpol / np.linalg.norm(Xpol, axis=1, keepdims=True)
        Epol = _energy(Xpol)
        if Epol < best_E:
            best = Xpol

    return best
