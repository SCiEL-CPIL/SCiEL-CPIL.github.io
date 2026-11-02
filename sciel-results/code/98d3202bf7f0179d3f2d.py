import time
import math
import numpy as np
from scipy.optimize import minimize, anneal

N = 150

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i, k] = sum_j (-(pts[i,k] - pts[j,k]) / dists_sq[i,j])
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad

def _energy_and_grad_3d(x):
    """Energy and gradient for L-BFGS-B on 3D coordinates."""
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    energy = _log_energy_fast(pts)
    grad = _log_energy_grad(pts)
    return energy, grad.flatten()

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _halton_sphere(n):
    """Generate Halton sequence points projected onto the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, 3))
    for i in range(n):
        u1 = halton(i + 1, 2)
        u2 = halton(i + 1, 3)
        u3 = halton(i + 1, 5)
        pts[i] = [u1 - 0.5, u2 - 0.5, u3 - 0.5]
    
    return _normalize(pts)

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _riemannian_cg(pts, time_limit=15.0, max_iter=500):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _log_energy_fast(current), _log_energy_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy = _log_energy_fast(new_pts)
            new_grad = _log_energy_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if new_energy > energy:
                break
            
            alpha *= 0.8
        
        if best_alpha == 0.0:
            break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta for conjugate gradient
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy

def _lbfgs_optimize(pts, time_limit=30.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p_pts = x.reshape(N, 3)
        p_pts = _normalize(p_pts)
        energy = _log_energy_fast(p_pts)
        grad = _log_energy_grad(p_pts)
        return energy, grad.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 20000, 'maxls': 100})
    
    x_opt = _normalize(res.x.reshape(N, 3))
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy

def _polish(pts, time_limit=5.0):
    """High-precision polish with very small steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-4
    
    for it in range(20000):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad = _log_energy_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(40):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < energy - 1e-16:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.02, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-18:
                break
    
    return current, energy

def _stiefel_manifold_optimize(pts, time_limit=60.0):
    """
    Stiefel manifold optimization: jointly optimize points and a rotation matrix.
    The idea: represent points as P @ R where P is the initial configuration
    and R is a 3x3 rotation matrix. This breaks symmetry and allows escaping
    local minima inherent in fixed-frame optimizations.
    
    We optimize over the Stiefel manifold of 3x3 orthogonal matrices.
    """
    start = time.time()
    deadline = start + time_limit
    
    # Initialize with identity rotation
    R = np.eye(3)
    base_pts = pts.copy()
    
    # Current points = base_pts @ R.T
    current_pts = _normalize(base_pts @ R.T)
    energy = _log_energy_fast(current_pts)
    
    # Gradient of energy w.r.t. R
    def _energy_and_grad_R(R_mat):
        pts_curr = _normalize(base_pts @ R_mat.T)
        e = _log_energy_fast(pts_curr)
        
        # Gradient w.r.t. points
        grad_pts = _log_energy_grad(pts_curr)
        
        # Chain rule: dE/dR = (dE/dpts) @ base_pts.T (with appropriate transpose)
        # pts = base_pts @ R.T, so dpts/dR = base_pts @ d(R.T)/dR
        # grad_R = (grad_pts.T @ base_pts).T = base_pts.T @ grad_pts
        grad_R = base_pts.T @ grad_pts
        
        # Project to tangent space of Stiefel manifold
        # Tangent space at R: {R @ Omega : Omega is 3x3 skew-symmetric}
        # Projection: grad_R - R @ sym(R.T @ grad_R)
        A = R.T @ grad_R
        sym_A = 0.5 * (A + A.T)
        grad_R_tangent = grad_R - R @ sym_A
        
        return e, grad_R_tangent.flatten()
    
    # Riemannian gradient descent on Stiefel manifold
    lr = 0.01
    prev_energy = energy
    prev_R = R.copy()
    
    for it in range(1000):
        if time.time() > deadline - 1.0:
            break
        
        e, grad_flat = _energy_and_grad_R(R)
        grad = grad_flat.reshape(3, 3)
        
        # Gradient step with exponential map on Stiefel manifold
        # Use Cayley transform for rotation
        Omega = -lr * grad
        # Make Omega skew-symmetric
        Omega = 0.5 * (Omega - Omega.T)
        
        # Cayley transform: R_new = (I - Omega/2) @ (I + Omega/2)^{-1} @ R
        I3 = np.eye(3)
        try:
            R_new = np.linalg.solve(I3 + 0.5 * Omega, I3 - 0.5 * Omega) @ R
        except np.linalg.LinAlgError:
            break
        
        # Orthogonalize R_new
        U, _, Vt = np.linalg.svd(R_new)
        R_new = U @ Vt
        
        new_pts = _normalize(base_pts @ R_new.T)
        new_energy = _log_energy_fast(new_pts)
        
        if new_energy < energy:
            R = R_new
            energy = new_energy
            current_pts = new_pts
            # Increase learning rate slightly if improving
            lr = min(lr * 1.05, 0.1)
        else:
            # Decrease learning rate
            lr *= 0.5
            if lr < 1e-10:
                break
        
        # Check convergence
        if abs(energy - prev_energy) < 1e-14:
            break
        prev_energy = energy
    
    return current_pts, energy

def _basin_hopping(pts, time_limit=30.0):
    """Basin hopping: perturb points and re-optimize to escape local minima."""
    start = time.time()
    deadline = start + time_limit
    
    current_pts = pts.copy()
    current_energy = _log_energy_fast(current_pts)
    
    rng = np.random.default_rng(42)
    
    for it in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation
        noise_scale = 0.05 * (1.0 + 0.5 * np.cos(it * 0.1))
        perturbed = current_pts + rng.standard_normal((N, 3)) * noise_scale
        perturbed = _normalize(perturbed)
        
        # Optimize the perturbed configuration
        try:
            opt_pts, opt_energy = _lbfgs_optimize(perturbed, time_limit=min(10.0, deadline - time.time() - 2.0))
            
            if opt_energy < current_energy:
                current_pts = opt_pts
                current_energy = opt_energy
        except Exception:
            continue
    
    return current_pts, current_energy

def _spectral_gap_aware_init(n):
    """
    Generate initial points that are aware of the spectral gap structure.
    Uses a combination of Fibonacci and random points with careful spacing.
    """
    # Start with Fibonacci
    pts = _fibonacci_sphere(n).copy()
    
    # Add small random perturbations to break symmetry
    rng = np.random.default_rng(123)
    pts = pts + rng.standard_normal((n, 3)) * 0.01
    pts = _normalize(pts)
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_pts = None
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Halton Sphere
    hal_pts = _halton_sphere(N)
    initializations.append(hal_pts)
    
    # 3. Random uniform on sphere
    for seed in range(6):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 4. Perturbed Fibonacci
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.03
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 5. Spectral gap aware init
    initializations.append(_spectral_gap_aware_init(N))
    
    # 6. More varied random inits
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 789 + 23)
        # Clustered random
        centers = rng_local.standard_normal((10, 3))
        centers = _normalize(centers)
        pts_clustered = np.zeros((N, 3))
        for i in range(N):
            c = centers[i % 10]
            pts_clustered[i] = c + rng_local.standard_normal(3) * 0.3
        pts_clustered = _normalize(pts_clustered)
        initializations.append(pts_clustered)
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        current_pts = init_pts.copy()
        
        # Phase 1: L-BFGS-B for coarse optimization
        try:
            current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=15.0)
            
            # Phase 2: Riemannian CG for fine optimization
            if time.time() < deadline - 10:
                current_pts, current_energy = _riemannian_cg(current_pts, time_limit=8.0)
            
            # Phase 3: Stiefel manifold optimization to break symmetry
            if time.time() < deadline - 8:
                stiefel_pts, stiefel_energy = _stiefel_manifold_optimize(current_pts, time_limit=min(15.0, deadline - time.time() - 5.0))
                if stiefel_energy < current_energy:
                    current_pts = stiefel_pts
                    current_energy = stiefel_energy
            
            # Phase 4: Final L-BFGS-B polish
            if time.time() < deadline - 5:
                current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=5.0)
            
            # Phase 5: Gradient polish
            if time.time() < deadline - 2:
                current_pts, current_energy = _polish(current_pts, time_limit=2.0)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        except Exception:
            pass
    
    # Basin hopping on the best result found so far
    if best_pts is not None and time.time() < deadline - 10:
        try:
            bh_pts, bh_energy = _basin_hopping(best_pts, time_limit=min(20.0, deadline - time.time() - 5.0))
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pts = bh_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        current_pts = best_pts.copy()
        current_energy = best_energy
        
        # Use a very tight L-BFGS-B polish
        try:
            x0 = current_pts.flatten()
            res = minimize(
                _energy_and_grad_3d,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 10000,
                    'ftol': 1e-16,
                    'gtol': 1e-13,
                    'maxls': 200
                }
            )
            polished_pts = _normalize(res.x.reshape(N, 3))
            polished_energy = _log_energy_fast(polished_pts)
            
            if polished_energy < current_energy:
                best_energy = polished_energy
                best_pts = polished_pts
        except Exception:
            pass
        
        # Final gradient polish
        if time.time() < deadline - 1:
            try:
                polished_pts, polished_energy = _polish(best_pts, time_limit=1.5)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_pts = polished_pts
            except Exception:
                pass
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
