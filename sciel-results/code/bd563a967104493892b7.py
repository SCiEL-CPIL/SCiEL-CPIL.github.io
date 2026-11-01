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


def _riemannian_lbfgs(pts, max_iter=1000, tol=1e-12, m=10, time_limit=None):
    """
    Riemannian L-BFGS optimization with proper two-loop recursion.
    Uses approximate parallel transport via projection.
    """
    n = len(pts)
    pts = pts.copy()
    
    # Store history of (s, y) pairs for L-BFGS
    s_hist = []  # step vectors (flattened)
    y_hist = []  # gradient differences (flatted)
    
    energy, grad = _energy_and_grad(pts)
    grad_flat = grad.flatten()
    
    for k in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
            
        # Check convergence
        grad_norm = np.linalg.norm(grad_flat)
        if grad_norm < tol:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        if len(s_hist) == 0:
            dir_flat = -grad_flat
        else:
            # Two-loop recursion
            m_actual = min(m, len(s_hist))
            # Get last m pairs
            s_list = s_hist[-m_actual:]
            y_list = y_hist[-m_actual:]
            
            # Compute alpha coefficients
            rho = []
            for s, y in zip(s_list, y_list):
                sy = np.dot(s, y)
                if abs(sy) < 1e-15:
                    rho.append(0.0)
                else:
                    rho.append(1.0 / sy)
            
            # Initialize
            r = grad_flat.copy()
            alpha = [0.0] * m_actual
            
            # Forward loop
            for i in range(m_actual - 1, -1, -1):
                alpha[i] = rho[i] * np.dot(s_list[i], r)
                r -= alpha[i] * y_list[i]
            
            # Scale r by gamma (inverse of Hessian approximation)
            if m_actual > 0:
                sy_last = np.dot(s_list[-1], y_list[-1])
                yy_last = np.dot(y_list[-1], y_list[-1])
                gamma = sy_last / max(yy_last, 1e-15)
            else:
                gamma = 1.0
            
            r = gamma * r
            
            # Backward loop
            beta = [0.0] * m_actual
            for i in range(m_actual):
                beta[i] = rho[i] * np.dot(y_list[i], r)
                r += (alpha[i] - beta[i]) * s_list[i]
            
            dir_flat = -r
        
        # Project direction to tangent space
        dir_3d = dir_flat.reshape(n, 3)
        dot_prod = np.sum(dir_3d * pts, axis=1)
        dir_tangent = dir_3d - dot_prod[:, np.newaxis] * pts
        
        # Line search with backtracking
        alpha_ls = 0.1
        accepted = False
        for _ in range(30):
            new_pts = _exponential_map(pts, alpha_ls * dir_tangent)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Check sufficient decrease (Armijo)
            if new_energy < energy + 1e-4 * alpha_ls * np.sum(dir_tangent * grad):
                accepted = True
                break
            alpha_ls *= 0.5
        
        if not accepted:
            # Try smaller step
            alpha_ls = 0.01
            for _ in range(10):
                new_pts = _exponential_map(pts, alpha_ls * dir_tangent)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    accepted = True
                    break
                alpha_ls *= 0.5
            
            if not accepted:
                break
        
        # Update history
        s_new = (new_pts - pts).flatten()
        y_new = (new_grad - grad).flatten()
        s_hist.append(s_new)
        y_hist.append(y_new)
        if len(s_hist) > m:
            s_hist.pop(0)
            y_hist.pop(0)
        
        # Update
        pts = new_pts
        energy = new_energy
        grad = new_grad
        grad_flat = grad.flatten()
    
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
    """Optimize a single set of points using Riemannian L-BFGS and high-precision polishing."""
    n = len(pts)
    
    # Phase 1: Riemannian L-BFGS
    try:
        opt_pts, opt_energy = _riemannian_lbfgs(pts, max_iter=500, tol=1e-10, m=10, time_limit=time_limit - 10)
    except Exception:
        opt_pts = pts
        opt_energy, _ = _energy_and_grad(opt_pts)
    
    # Phase 2: High-precision polish with very tight tolerance
    current_pts = _normalize(opt_pts)
    current_energy, _ = _energy_and_grad(current_pts)
    
    # Use scipy L-BFGS-B for final polish with tight tolerances
    def objective_and_grad(flat_pts):
        pts_3d = flat_pts.reshape(n, 3)
        pts_3d = _normalize(pts_3d)
        energy, grad = _energy_and_grad(pts_3d)
        flat_grad = grad.flatten()
        return energy, flat_grad
    
    x0 = current_pts.flatten()
    
    try:
        if time.time() < time_limit - 5:
            result = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-15,
                    'gtol': 1e-15,
                    'maxls': 40,
                }
            )
            polish_pts = result.x.reshape(n, 3)
            polish_pts = _normalize(polish_pts)
            polish_energy, _ = _energy_and_grad(polish_pts)
            if polish_energy < current_energy:
                current_pts = polish_pts
                current_energy = polish_energy
    except Exception:
        pass
    
    # Phase 3: Final gradient descent polish with adaptive step
    step_size = 0.005
    for _ in range(2000):
        if time.time() > time_limit - 2:
            break
            
        energy, grad = _energy_and_grad(current_pts)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(20):
            new_pts = _exponential_map(current_pts, -alpha * grad)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < current_energy - 1e-17:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current_pts = new_pts
            current_energy = new_energy
            step_size = min(alpha * 1.05, 0.05)
        else:
            step_size *= 0.5
            if step_size < 1e-12:
                break
    
    return current_pts, current_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    # Generate diverse initializations
    inits = []
    
    # Fibonacci spiral
    fib_pts = _fibonacci_sphere(N)
    inits.append(fib_pts)
    
    # Halton sequence
    halton_pts = _halton_sphere(N)
    inits.append(halton_pts)
    
    # Random uniform with different seeds
    for seed in range(8):
        rng = np.random.default_rng(seed * 7919 + 13)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        inits.append(pts_rand)
        
    # Perturbed Fibonacci with varying perturbation sizes
    for seed, pert_size in [(0, 0.03), (1, 0.07), (2, 0.12), (3, 0.2)]:
        rng = np.random.default_rng(seed * 331 + 17)
        pts_pert = fib_pts + rng.standard_normal((N, 3)) * pert_size
        pts_pert = _normalize(pts_pert)
        inits.append(pts_pert)
    
    # Also try Halton with different bases
    for base in [3, 5, 7]:
        halton_pts2 = _halton_sphere(N, base_start=base)
        inits.append(halton_pts2)

    best_pts = None
    best_energy = float('inf')
    
    # Sort initializations by initial energy to prioritize better starts
    init_energies = []
    for init_pts in inits:
        e, _ = _energy_and_grad(init_pts)
        init_energies.append(e)
    
    # Sort indices by initial energy
    sorted_indices = np.argsort(init_energies)
    
    # Optimize each initialization in order of initial quality
    for idx in sorted_indices:
        if time.time() > deadline - 15:
            break
            
        init_pts = inits[idx]
        remaining = deadline - time.time() - 10
        if remaining < 8:
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
