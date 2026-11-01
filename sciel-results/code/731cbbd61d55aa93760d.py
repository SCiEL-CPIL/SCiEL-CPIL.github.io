"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Compute energy and gradient (projected to tangent space) in one pass."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of log(1/d_ij)
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    d_upper = np.maximum(d_upper, 1e-15)
    energy = np.sum(np.log(1.0 / d_upper))
    
    # Gradient
    inv_d2 = 1.0 / (dists ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, delta):
    """Exponential map on sphere: move points by tangent vectors delta."""
    v_norms = np.linalg.norm(delta, axis=1, keepdims=True)
    v_norms_safe = np.maximum(v_norms, 1e-15)
    cos_t = np.cos(v_norms_safe)
    sin_t = np.sin(v_norms_safe)
    new_pts = cos_t * pts + sin_t * (delta / v_norms_safe)
    return _normalize(new_pts)


def _riemannian_lbfgs(pts, max_iter=1000, tol=1e-12, m=10):
    """
    Riemannian L-BFGS optimization.
    """
    n = len(pts)
    pts = pts.copy()
    
    # Store history of gradients and steps for L-BFGS
    grad_hist = []
    step_hist = []
    
    energy, grad = _energy_and_grad(pts)
    
    for k in range(max_iter):
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        if len(grad_hist) == 0:
            dir_vec = -grad
        else:
            # Two-loop recursion
            # We store (s_i, y_i) where s_i = x_{i+1} - x_i (in tangent space approx)
            # and y_i = grad_{i+1} - grad_i
            # Note: In Riemannian L-BFGS, we need to parallel transport gradients.
            # For simplicity and robustness, we can use a standard L-BFGS on the
            # flattened coordinates with projection, or approximate the transport.
            # Given the time limit, a simpler Riemannian Gradient Descent with
            # adaptive step size might be more stable if L-BFGS transport is tricky.
            # However, L-BFGS is much faster.
            # Let's use a simplified approach: Use scipy L-BFGS-B on flattened coords
            # but with a very tight tolerance and multiple restarts.
            # Actually, the current code already does this.
            # Let's try a Riemannian Conjugate Gradient instead, which is easier to implement correctly.
            pass
            
    # Fallback: Use the robust approach from the seed but with better initialization and polishing.
    # The seed's approach is actually quite good. The key is better initialization and polishing.
    return pts, energy


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _halton_sphere(n, base_start=2):
    """Generate n points on sphere using Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u = halton(i, base_start)
        v = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u
        phi = np.arccos(2.0 * v - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    return _normalize(np.array(pts))


def _optimize_single(pts, time_limit):
    """Optimize a single set of points using L-BFGS and polishing."""
    start = time.time()
    n = len(pts)
    
    def objective_and_grad(flat_pts):
        pts_3d = flat_pts.reshape(n, 3)
        pts_3d = _normalize(pts_3d)
        energy, grad = _energy_and_grad(pts_3d)
        flat_grad = grad.flatten()
        return energy, flat_grad
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'ftol': 1e-14,
                'gtol': 1e-14,
                'maxls': 50,
            }
        )
        opt_pts = result.x.reshape(n, 3)
        opt_pts = _normalize(opt_pts)
    except Exception:
        opt_pts = pts
        result = None
    
    # Polish with gradient descent
    current_pts = opt_pts
    current_energy, _ = _energy_and_grad(current_pts)
    
    step_size = 0.01
    for _ in range(1000):
        if time.time() > time_limit:
            break
            
        energy, grad = _energy_and_grad(current_pts)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(15):
            new_pts = _exponential_map(current_pts, -alpha * grad)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < current_energy - 1e-16:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current_pts = new_pts
            current_energy = new_energy
            step_size = min(alpha * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-10:
                break
        
    return current_pts, current_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate diverse initializations
    inits = []
    
    # Fibonacci spiral
    fib_pts = _fibonacci_sphere(N)
    inits.append(fib_pts)
    
    # Halton sequence
    halton_pts = _halton_sphere(N)
    inits.append(halton_pts)
    
    # Random uniform with different seeds
    for seed in range(6):
        rng = np.random.default_rng(seed * 7919 + 13)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        inits.append(pts_rand)
        
    # Perturbed Fibonacci
    for seed in range(3):
        rng = np.random.default_rng(seed * 331 + 17)
        pts_pert = fib_pts + rng.standard_normal((N, 3)) * 0.05
        pts_pert = _normalize(pts_pert)
        inits.append(pts_pert)

    best_pts = None
    best_energy = float('inf')
    
    # Optimize each initialization
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
            
        try:
            opt_pts, opt_energy = _optimize_single(init_pts, time.time() + remaining)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            continue

    if best_pts is None:
        best_pts = fib_pts
    else:
        best_pts = _normalize(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
