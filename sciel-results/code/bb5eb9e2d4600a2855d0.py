"""Optimized logarithmic energy minimizer for 150 points on the unit sphere using SDP relaxation and Riemannian optimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _energy_and_grad(x):
    """
    Compute logarithmic energy and Euclidean gradient for points x of shape (N, 3).
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient is the Euclidean gradient in R^3.
    """
    # Compute pairwise differences (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # Compute squared distances (N, N)
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) and division by zero
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    # Mask for upper triangle i < j
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient computation
    # grad[i] = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||
    # Note: The term for pair (i, j) contributes to grad[i] and grad[j].
    # d/dx_i (-log ||x_i - x_j||) = -(x_i - x_j) / ||x_i - x_j||
    # d/dx_j (-log ||x_i - x_j||) = (x_i - x_j) / ||x_i - x_j|| = -(x_j - x_i) / ||x_i - x_j||
    
    grad = np.zeros_like(x)
    
    # Get distances and vectors for i < j
    d_ij = dist[iu]  # (M,)
    dvec = diff[iu]  # (M, 3) -> x_i - x_j
    
    # Contribution to grad[i] is -dvec / d_ij
    # Contribution to grad[j] is +dvec / d_ij
    
    i_idx = iu[0]
    j_idx = iu[1]
    
    contrib = dvec / d_ij[:, np.newaxis]
    
    # Accumulate gradients
    # np.add.at is slow, but for N=150 it's acceptable. 
    # Alternatively, we can use bincount or scatter, but add.at is clear.
    np.add.at(grad, i_idx, -contrib)
    np.add.at(grad, j_idx, contrib)
    
    return energy, grad


def _log_energy_fast(x):
    """Fast energy computation only (no gradient)."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))


def _fibonacci_sphere(n):
    """Generate n points using Fibonacci spiral on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere(n):
    """Generate n points using Halton sequence on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        p = [halton(i, b) for b in [2, 3, 5]]
        theta = 2 * np.pi * p[0]
        phi = np.arccos(2 * p[1] - 1)
        pt = [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        norm = np.linalg.norm(pt)
        points.append(pt / norm)
    return np.array(points)


def _normalize(x):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _riemannian_grad_descent(x0, deadline, max_iter=20000, tol=1e-10):
    """
    Riemannian Gradient Descent with backtracking line search.
    Uses projection retraction onto the sphere.
    """
    x = x0.copy()
    energy, grad = _energy_and_grad(x)
    
    # Project gradient to tangent space
    # grad_tangent = grad - (grad . x) x
    grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
    
    # Initial step size
    alpha = 0.1
    prev_energy = energy
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
            
        # Backtracking line search
        # We want to minimize f(x + alpha * grad_tangent) after retraction
        # Approximation: f(retract(x, alpha * grad_tangent))
        
        step_dir = -grad_tangent
        step_norm = np.linalg.norm(step_dir, axis=1, keepdims=True)
        
        # If gradient is very small, stop
        if np.max(step_norm) < 1e-15:
            break
            
        # Normalize direction for step size control? 
        # Usually alpha scales the raw gradient.
        
        found_alpha = False
        for _ in range(20):
            # Move in tangent direction
            x_trial = x + alpha * step_dir
            # Retract to sphere
            x_trial = _normalize(x_trial)
            
            energy_trial = _log_energy_fast(x_trial)
            
            # Armijo condition: f_new <= f_old - c * alpha * ||grad||^2
            # Simple check: energy decreased
            if energy_trial < energy:
                found_alpha = True
                break
            alpha *= 0.5
            
        if not found_alpha:
            # If we can't find a step that decreases energy, try a very small step
            x_trial = x + 1e-8 * step_dir
            x_trial = _normalize(x_trial)
            energy_trial = _log_energy_fast(x_trial)
            if energy_trial >= energy:
                break # Stuck
        
        # Update
        x = x_trial
        energy = energy_trial
        
        # Compute new gradient
        _, grad = _energy_and_grad(x)
        grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        
        # Adaptive alpha: increase if we took full step, decrease if we reduced
        # Here we just keep alpha or adjust slightly
        # Simple strategy: if we broke out early, alpha was good.
        # If we reduced alpha, maybe increase it back slowly?
        # Let's stick to a fixed alpha or slow adaptation to avoid complexity.
        # A common trick is to scale alpha by the inverse of the gradient norm magnitude.
        # But for now, fixed alpha with backtracking is robust.
        
    return x


def _sdp_gram_optimize(x0, deadline, max_iter=5000):
    """
    SDP relaxation approach: optimize the Gram matrix G = X^T X where X is (N, 3).
    Constraints: G is PSD, diag(G) = 1, trace(G) = 3 (since points are on unit sphere in R^3).
    Actually, for N points on S^2 in R^3, we have trace(G) = N (each point has norm 1).
    Wait: G_ij = x_i · x_j, so trace(G) = sum_i ||x_i||^2 = N.
    And G must be PSD with rank <= 3.
    
    Energy in terms of G: d_ij^2 = 2 - 2*G_ij, so log(1/d_ij) = -0.5*log(2-2*G_ij)
    Energy = -0.5 * sum_{i<j} log(2 - 2*G_ij)
    
    We optimize G directly with PSD constraint using projected gradient.
    """
    # Initialize G from x0
    G = x0 @ x0.T  # (N, N)
    # Ensure diag is 1
    np.fill_diagonal(G, 1.0)
    # Symmetrize
    G = (G + G.T) / 2.0
    
    def _energy_from_gram(G):
        """Compute energy from Gram matrix."""
        # d_ij^2 = 2 - 2*G_ij
        dist_sq = 2.0 - 2.0 * G
        np.fill_diagonal(dist_sq, 1.0)  # avoid log(0) on diagonal
        # Clip to avoid negative values (shouldn't happen if G is valid)
        dist_sq = np.maximum(dist_sq, 1e-10)
        iu = np.triu_indices(N, k=1)
        return -0.5 * np.sum(np.log(dist_sq[iu]))
    
    def _grad_from_gram(G):
        """Compute gradient of energy w.r.t. G."""
        dist_sq = 2.0 - 2.0 * G
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-10)
        
        # dE/dG_ij = -0.5 * d/dG_ij log(2-2*G_ij) = -0.5 * (-2)/(2-2*G_ij) = 1/(2-2*G_ij)
        # But only for i != j
        grad = np.zeros_like(G)
        iu = np.triu_indices(N, k=1)
        vals = 1.0 / dist_sq[iu]
        grad[iu] = vals
        grad[iu[1], iu[0]] = vals
        return grad
    
    def _project_psd_trace1(G):
        """
        Project G onto the set of PSD matrices with diag = 1 and trace = N.
        This is a complex projection. Simple approach:
        1. Symmetrize
        2. Clip eigenvalues to be non-negative (PSD)
        3. Adjust to enforce diag = 1
        4. This is approximate; for better results use iterative methods.
        """
        G = (G + G.T) / 2.0
        
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        
        # Clip negative eigenvalues
        eigvals_clipped = np.maximum(eigvals, 0)
        
        # Reconstruct
        G_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        G_psd = (G_psd + G_psd.T) / 2.0
        
        # Enforce diagonal = 1
        # This is tricky; a simple approach is to scale rows/columns
        # Better: use iterative diagonal adjustment
        for _ in range(50):
            diag = np.diag(G_psd)
            # Scale each row/column to make diag = 1
            # D_ii = 1/sqrt(G_ii)
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = 1.0 / np.sqrt(np.maximum(diag, 1e-15))
            # Apply scaling: G_new = D @ G @ D
            D = np.diag(scale)
            G_psd = D @ G_psd @ D
            G_psd = (G_psd + G_psd.T) / 2.0
            
            # Check convergence
            if np.max(np.abs(np.diag(G_psd) - 1.0)) < 1e-10:
                break
        
        # Final symmetrization
        G_psd = (G_psd + G_psd.T) / 2.0
        np.fill_diagonal(G_psd, 1.0)
        
        return G_psd
    
    G_current = G.copy()
    energy = _energy_from_gram(G_current)
    grad = _grad_from_gram(G_current)
    
    alpha = 0.01
    prev_energy = energy
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-12:
            break
        
        step_dir = -grad
        
        found_alpha = False
        for _ in range(30):
            G_trial = G_current + alpha * step_dir
            G_trial = _project_psd_trace1(G_trial)
            
            energy_trial = _energy_from_gram(G_trial)
            
            if energy_trial < energy:
                found_alpha = True
                break
            alpha *= 0.5
        
        if not found_alpha:
            # Try very small step
            G_trial = G_current + 1e-6 * step_dir
            G_trial = _project_psd_trace1(G_trial)
            energy_trial = _energy_from_gram(G_trial)
            if energy_trial >= energy:
                break
        
        G_current = G_trial
        energy = energy_trial
        grad = _grad_from_gram(G_current)
        
        # Slowly increase alpha
        alpha = min(alpha * 1.05, 0.1)
    
    # Extract points from Gram matrix
    # G = X^T X, so we need X such that X^T X = G
    # Use eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(G_current)
    # Keep top 3 eigenvalues (since points are in R^3)
    # Sort by eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 3
    k = 3
    eigvals_top = eigvals[:k]
    eigvecs_top = eigvecs[:, :k]
    
    # X = sqrt(eigvals) * eigvecs^T
    with np.errstate(divide='ignore', invalid='ignore'):
        X = eigvecs_top.T @ np.diag(np.sqrt(np.maximum(eigvals_top, 0)))
    
    # Normalize rows
    X = _normalize(X)
    
    return X


def _lbfgs_on_manifold(x0, deadline):
    """
    Alternative: Use L-BFGS but enforce constraints more strictly?
    Actually, standard L-BFGS on R^3 with normalization is often okay if started well,
    but Riemannian GD is more principled.
    Let's try a hybrid: L-BFGS on the tangent space? No, that's complex.
    
    Let's try running L-BFGS on the full vector but with a very tight ftol/gtol
    and frequent normalization? No, L-BFGS doesn't support custom retraction.
    
    Instead, let's use a high-precision polish using Riemannian CG or Newton.
    For now, we'll rely on Riemannian GD with many iterations.
    """
    pass


def _polish(x0, deadline):
    """
    High-precision polish pass.
    Uses Riemannian Gradient Descent with very tight tolerance and small steps.
    """
    remaining = deadline - time.time()
    if remaining < 2.0:
        return x0
    
    # Run RGD with tighter tolerance and smaller max step
    x_opt = _riemannian_grad_descent(x0, deadline, max_iter=50000, tol=1e-12)
    
    # Validate
    if not np.all(np.isfinite(x_opt)):
        return x0
    if np.any(np.linalg.norm(x_opt, axis=1) < 1e-15):
        return x0
        
    # Check minimum distance
    diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < (1e-9) ** 2:
        return x0
        
    return x_opt


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_x = None
    best_energy = float('inf')
    
    # Generate multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("halton", _halton_sphere(N)),
    ]
    
    # Add random initializations
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
    
    # Phase 1: Run Riemannian GD on all initializations
    for name, x0 in initializations:
        if time.time() > deadline - 20.0:
            break
        
        x0 = _normalize(x0)
        
        # Run Riemannian Gradient Descent
        x_opt = _riemannian_grad_descent(x0, deadline - 10.0, max_iter=10000)
        
        if np.all(np.isfinite(x_opt)) and np.min(np.linalg.norm(x_opt, axis=1)) > 1e-15:
            energy = _log_energy_fast(x_opt)
            
            # Check min distance
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            if np.min(dist_sq) >= (1e-9) ** 2:
                if energy < best_energy:
                    best_energy = energy
                    best_x = x_opt.copy()
    
    # Phase 2: Try SDP Gram optimization on the best result
    if best_x is not None and time.time() < deadline - 15.0:
        try:
            x_sdp = _sdp_gram_optimize(best_x, deadline - 5.0, max_iter=3000)
            if np.all(np.isfinite(x_sdp)) and np.min(np.linalg.norm(x_sdp, axis=1)) > 1e-15:
                energy_sdp = _log_energy_fast(x_sdp)
                
                # Check min distance
                diff = x_sdp[:, np.newaxis, :] - x_sdp[np.newaxis, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                if np.min(dist_sq) >= (1e-9) ** 2:
                    if energy_sdp < best_energy:
                        best_energy = energy_sdp
                        best_x = x_sdp.copy()
        except Exception:
            pass
    
    # Phase 3: Try SDP on a few other good initializations if time permits
    if time.time() < deadline - 20.0:
        for name, x0 in initializations[:3]:  # Try first 3
            if time.time() > deadline - 10.0:
                break
            
            x0 = _normalize(x0)
            
            # Quick RGD first
            x_quick = _riemannian_grad_descent(x0, deadline - 8.0, max_iter=3000)
            if not np.all(np.isfinite(x_quick)):
                continue
            
            energy_quick = _log_energy_fast(x_quick)
            if energy_quick >= best_energy:
                continue
            
            # Try SDP
            try:
                x_sdp = _sdp_gram_optimize(x_quick, deadline - 5.0, max_iter=2000)
                if np.all(np.isfinite(x_sdp)) and np.min(np.linalg.norm(x_sdp, axis=1)) > 1e-15:
                    energy_sdp = _log_energy_fast(x_sdp)
                    
                    diff = x_sdp[:, np.newaxis, :] - x_sdp[np.newaxis, :, :]
                    dist_sq = np.sum(diff * diff, axis=2)
                    np.fill_diagonal(dist_sq, np.inf)
                    if np.min(dist_sq) >= (1e-9) ** 2:
                        if energy_sdp < best_energy:
                            best_energy = energy_sdp
                            best_x = x_sdp.copy()
            except Exception:
                pass
    
    # Final polish on the best result
    if best_x is not None and time.time() < deadline - 3.0:
        x_polished = _polish(best_x, deadline)
        if np.all(np.isfinite(x_polished)):
            energy_polished = _log_energy_fast(x_polished)
            diff = x_polished[:, np.newaxis, :] - x_polished[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            if np.min(dist_sq) >= (1e-9) ** 2 and energy_polished < best_energy:
                best_energy = energy_polished
                best_x = x_polished.copy()
    
    if best_x is None:
        best_x = _fibonacci_sphere(N)
    
    return _normalize(best_x)
