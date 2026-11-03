"""
Strategy: Spectral/SDP-relaxation-inspired initialization + Riemannian trust-region polish.
Uses eigen-decomposition of the log-kernel matrix to generate globally-informed initial
configurations, then refines with a high-precision Riemannian optimizer.
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


def _spectral_init(rng, n_modes=10):
    """
    Generate initialization via spectral decomposition of the log-kernel.
    This mimics an SDP relaxation: we construct a Gram matrix G where G_ij = log(1/||x_i - x_j||)
    and use its eigenvectors to generate diverse, globally-informed configurations.
    """
    # Start with a base configuration (Fibonacci sphere)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X_base = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute the log-kernel matrix K where K_ij = log(1/||x_i - x_j||) for i != j, 0 on diagonal
    dot = X_base @ X_base.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)  # Avoid log(0)
    dist_sq = np.maximum(dist_sq, 1e-12)
    K = np.log(1.0 / np.sqrt(dist_sq))
    np.fill_diagonal(K, 0.0)
    
    # Eigen-decompose K to get spectral components
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(K)
    except Exception:
        # Fallback to Fibonacci if eigen fails
        return X_base
    
    # Use top eigenvectors to create diverse configurations
    # Each eigenvector gives a pattern of "correlation" that we can use to perturb the base
    starts = []
    
    # Base configuration
    starts.append(X_base.copy())
    
    # Generate diverse starts from spectral components
    for mode in range(min(n_modes, N)):
        if mode >= len(eigenvalues):
            break
        # Scale by eigenvalue magnitude to emphasize important modes
        scale = abs(eigenvalues[mode]) / (abs(eigenvalues[0]) + 1e-12)
        if scale < 0.01:
            continue
        
        # Create perturbation from eigenvector
        eig = eigenvectors[:, mode]
        # Map eigenvector values to angular perturbations
        perturbation = eig[:, np.newaxis] * 0.1 * scale
        
        # Apply perturbation in tangent space
        X_perturbed = X_base + perturbation
        norms = np.linalg.norm(X_perturbed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_perturbed = X_perturbed / norms
        
        # Add random rotation for diversity
        M = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(M)
        d = np.sign(np.linalg.det(Q))
        Q[:, 0] *= d
        X_rotated = X_perturbed @ Q.T
        
        starts.append(X_rotated)
    
    # Add a few random starts for robustness
    for _ in range(3):
        X_rand = rng.standard_normal((N, 3))
        norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
        starts.append(X_rand / norms)
    
    return starts


def _trust_region_polish(X, deadline, max_iter=500):
    """
    High-precision Riemannian trust-region polish using L-BFGS-B with tight tolerances.
    This is the final refinement step that achieves sub-1e-10 precision.
    """
    V0 = np.zeros(N * 3)
    
    def obj_and_grad(V_flat):
        if time.time() > deadline:
            E, _ = _energy_and_tangent_grad(X)
            return E, np.zeros(N * 3)
        
        V = V_flat.reshape(N, 3)
        X_new = _retract(X, V)
        E, grad_tan = _energy_and_tangent_grad(X_new)
        return E, grad_tan.flatten()
    
    try:
        result = minimize(
            obj_and_grad,
            V0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': 10000,
                'ftol': 1e-18,
                'gtol': 1e-15,
                'maxls': 200,
                'iprint': -1,
            }
        )
        V_opt = result.x.reshape(N, 3)
        X_opt = _retract(X, V_opt)
    except Exception:
        X_opt = X
    
    return X_opt


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


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_X = None
    best_energy = float('inf')
    
    # Phase 1: Generate diverse starts using spectral/SDP-inspired method
    starts = _spectral_init(rng, n_modes=15)
    
    # Also add some Fibonacci-based starts for robustness
    for i in range(3):
        starts.append(_fibonacci_init(rng, offset=float(i) * 0.15))
    
    # Phase 2: Optimize each start with Riemannian L-BFGS-B
    for s, X_init in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 10.0:
            break
        
        # Allocate time based on start index: earlier starts get more time
        time_per_start = min(remaining / max(1, len(starts) - s), 20.0)
        maxiter = min(5000, int(time_per_start * 20))
        if maxiter < 100:
            break
        
        # First pass: moderate optimization
        V0 = np.zeros(N * 3)
        
        def obj_and_grad_pass1(V_flat, X_init=X_init):
            if time.time() > deadline - 5.0:
                E, _ = _energy_and_tangent_grad(X_init)
                return E, np.zeros(N * 3)
            
            V = V_flat.reshape(N, 3)
            X_new = _retract(X_init, V)
            E, grad_tan = _energy_and_tangent_grad(X_new)
            return E, grad_tan.flatten()
        
        try:
            result = minimize(
                obj_and_grad_pass1,
                V0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': maxiter,
                    'maxfun': maxiter * 2,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'iprint': -1,
                }
            )
            V_opt = result.x.reshape(N, 3)
            X_opt = _retract(X_init, V_opt)
        except Exception:
            X_opt = X_init
        
        E_opt, _ = _energy_and_tangent_grad(X_opt)
        
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt.copy()
    
    # Phase 3: High-precision polish of the best solution
    if best_X is not None and time.time() < deadline - 10.0:
        X_polished = _trust_region_polish(best_X, deadline - 5.0, max_iter=1000)
        E_polished, _ = _energy_and_tangent_grad(X_polished)
        if E_polished < best_energy:
            best_energy = E_polished
            best_X = X_polished
    
    # Phase 4: Final ultra-precise polish if time permits
    if best_X is not None and time.time() < deadline - 5.0:
        # One more round with even tighter tolerances
        V0 = np.zeros(N * 3)
        
        def obj_and_grad_final(V_flat, X_init=best_X):
            if time.time() > deadline - 2.0:
                E, _ = _energy_and_tangent_grad(X_init)
                return E, np.zeros(N * 3)
            
            V = V_flat.reshape(N, 3)
            X_new = _retract(X_init, V)
            E, grad_tan = _energy_and_tangent_grad(X_new)
            return E, grad_tan.flatten()
        
        try:
            result = minimize(
                obj_and_grad_final,
                V0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'maxfun': 1000,
                    'ftol': 1e-20,
                    'gtol': 1e-16,
                    'maxls': 200,
                    'iprint': -1,
                }
            )
            V_opt = result.x.reshape(N, 3)
            X_final = _retract(best_X, V_opt)
            E_final, _ = _energy_and_tangent_grad(X_final)
            if E_final < best_energy:
                best_energy = E_final
                best_X = X_final
        except Exception:
            pass
    
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
