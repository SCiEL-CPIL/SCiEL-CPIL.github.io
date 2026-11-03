"""
Strategy: Deterministic Riemannian L-BFGS-B with multiple diverse initializations.
Uses a proper retraction-based parameterization where the optimization variable is
the tangent vector at the current iterate, enabling L-BFGS-B to work on the manifold.
Combined with Fibonacci sphere and random starts, plus a final Newton-type polish.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# Precompute pair indices
_idx_i, _idx_j = np.triu_indices(N, k=1)
_n_pairs = len(_idx_i)


def _energy_and_tangent_grad(X):
    """
    Compute energy and tangent-space gradient for logarithmic energy on sphere.
    X: (N, 3) array of unit vectors
    Returns: (energy, grad_tangent) where grad_tangent is (N,3)
    """
    dot = X @ X.T
    dist_sq = 2.0 - 2.0 * dot
    d2 = dist_sq[_idx_i, _idx_j]
    d2 = np.maximum(d2, 1e-18)
    
    energy = -0.5 * math.fsum(np.log(d2))
    
    diff = X[_idx_i] - X[_idx_j]
    inv_d2 = 1.0 / d2
    
    grad_euclidean = np.zeros((N, 3))
    np.add.at(grad_euclidean, _idx_i, diff * inv_d2[:, np.newaxis])
    np.add.at(grad_euclidean, _idx_j, -diff * inv_d2[:, np.newaxis])
    
    dot_grad = np.sum(grad_euclidean * X, axis=1, keepdims=True)
    grad_tangent = grad_euclidean - dot_grad * X
    
    return energy, grad_tangent


def _exp_map(X, V, step):
    """Exponential map on sphere."""
    cos_s = math.cos(step)
    sin_s = math.sin(step)
    X_new = cos_s * X + sin_s * V
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_new = X_new / norms
    return X_new


def _retract(X, V):
    """Retraction: normalize(X + V)."""
    X_new = X + V
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X_new / norms


def _fibonacci_init(rng, offset=0.0):
    """Generate Fibonacci sphere initialization with random rotation."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    X = X @ Q.T
    return X


def _random_init(rng):
    """Generate random points on sphere."""
    X = rng.standard_normal((N, 3))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def _lbfgs_sphere(X_init, deadline, maxiter=3000):
    """
    Riemannian L-BFGS-B using a retraction-based pullback.
    The variable V represents the displacement from X_init via retraction.
    At each evaluation, we compute the point X_new = retract(X_init, V),
    get the tangent gradient, and return it as the gradient in V-space.
    """
    V0 = np.zeros(N * 3)
    
    def obj_and_grad(V_flat):
        if time.time() > deadline:
            E, _ = _energy_and_tangent_grad(X_init)
            return E, np.zeros(N * 3)
        
        V = V_flat.reshape(N, 3)
        X_new = _retract(X_init, V)
        E, grad_tan = _energy_and_tangent_grad(X_new)
        return E, grad_tan.flatten()
    
    try:
        result = minimize(
            obj_and_grad,
            V0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'maxfun': 5000,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 100,
                'iprint': -1,
            }
        )
        V_opt = result.x.reshape(N, 3)
        X_opt = _retract(X_init, V_opt)
    except Exception:
        X_opt = X_init
    
    return X_opt


def _newton_polish(X, deadline, max_iter=100):
    """
    Newton-type refinement using approximate Hessian via finite differences.
    This is a heavy method used only at the very end for precision.
    """
    X = X.copy()
    E_curr, grad = _energy_and_tangent_grad(X)
    
    # Use conjugate gradient on the linearized problem as a cheap Newton substitute
    # Actually, let's do a few steps of trust-region-like Newton with diagonal Hessian approx
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Approximate curvature: use |grad| as a rough scale
        # Step: Newton direction would be -H^{-1} g; approximate with -g / (1 + ||g||)
        step = grad / (1.0 + grad_norm)
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break
        
        # Try several step sizes
        accepted = False
        for alpha in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
            V = -step / step_norm  # Unit direction
            X_new = _exp_map(X, V, alpha * 0.1)
            E_new, grad_new = _energy_and_tangent_grad(X_new)
            
            if E_new < E_curr:
                X = X_new
                E_curr = E_new
                grad = grad_new
                accepted = True
                break
        
        if not accepted:
            break
    
    return X, E_curr


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(12345)
    
    best_X = None
    best_energy = float('inf')
    
    # Phase 1: Multiple L-BFGS-B starts
    # Diverse initializations: Fibonacci with different rotations/offsets + random
    starts = []
    
    # 4 Fibonacci-based starts with different rotations
    for i in range(4):
        starts.append(_fibonacci_init(rng, offset=float(i) * 0.1))
    
    # 2 random starts
    for i in range(2):
        starts.append(_random_init(rng))
    
    # 2 Fibonacci with small perturbations
    for i in range(2):
        X = _fibonacci_init(rng)
        pert = rng.standard_normal((N, 3)) * 0.05
        X = X + pert
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        starts.append(X / norms)
    
    for s, X_init in enumerate(starts):
        if time.time() > deadline - 15.0:
            break
        
        remaining = deadline - time.time()
        # Give more iterations to earlier starts, less to later ones
        maxiter = min(3000, int(remaining * 15))
        if maxiter < 100:
            break
        
        X_opt = _lbfgs_sphere(X_init, deadline - 10.0, maxiter=maxiter)
        E_opt, _ = _energy_and_tangent_grad(X_opt)
        
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt.copy()
    
    # Phase 2: Polish the best solution with Newton-type steps
    if best_X is not None and time.time() < deadline - 5.0:
        X_polished, E_polished = _newton_polish(best_X, deadline - 3.0, max_iter=200)
        if E_polished < best_energy:
            best_energy = E_polished
            best_X = X_polished
    
    # Phase 3: Final L-BFGS-B polish from the best solution
    if best_X is not None and time.time() < deadline - 2.0:
        X_final = _lbfgs_sphere(best_X, deadline - 1.0, maxiter=500)
        E_final, _ = _energy_and_tangent_grad(X_final)
        if E_final < best_energy:
            best_energy = E_final
            best_X = X_final
    
    # Fallback
    if best_X is None:
        best_X = _fibonacci_init(rng)
    
    # Final normalization and validation
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_init(rng)
    
    return best_X
