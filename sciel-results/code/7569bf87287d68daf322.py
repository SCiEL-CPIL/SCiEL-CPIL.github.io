"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    d = np.sqrt(d_sq)
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _log_energy_grad(pts):
    """Compute logarithmic energy and its gradient w.r.t. free R^3 coordinates.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero
    inv_dist_sq = np.zeros_like(dist_sq)
    off_diag = dist_sq > 1e-30
    inv_dist_sq[off_diag] = 1.0 / dist_sq[off_diag]
    
    # Gradient of E = -sum log(d) is -sum (x_i - x_j)/d^2
    # diff[i,j] = x_i - x_j
    # grad_i = - sum_j diff[i,j] / dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _project_to_tangent(pts, grad):
    """Project gradient onto tangent space of unit sphere."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

def _make_fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _adam_optimize(pts0, time_limit=100.0, lr_init=0.01, no_improve_break=500):
    """Optimize points on the sphere using Adam gradient descent with adaptive learning rate."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Adam parameters
    lr = lr_init
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # Initialize moments
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    
    start_time = time.time()
    no_improve_steps = 0
    max_no_improve = no_improve_break
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _log_energy_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        if no_improve_steps > max_no_improve:
            break
            
        # Project gradient to tangent space
        grad_tangent = _project_to_tangent(pts, grad)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent * grad_tangent)
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update
        step = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        pts_new = pts - step
        
        # Retract to sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check for convergence or divergence
        if not np.all(np.isfinite(pts_new)):
            break
            
        pts = pts_new
        
        # Adaptive learning rate: reduce if energy is increasing
        if t % 50 == 0:
            current_energy = _log_energy(pts)
            if current_energy > best_energy + 1.0:
                lr *= 0.95
                if lr < 1e-7:
                    break

    return best_pts, best_energy

def _lbfgs_optimize(pts0, maxiter=2000, ftol=1e-15, gtol=1e-12):
    """Optimize using L-BFGS-B with tight tolerances, starting from pts0."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    def objective(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        e, _ = _log_energy_grad(pts_norm)
        return e
    
    def gradient(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        _, g = _log_energy_grad(pts_norm)
        g_tan = _project_to_tangent(pts_norm, g)
        return g_tan.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': ftol, 'gtol': gtol}
        )
        
        if result.fun < best_energy:
            pts_opt = result.x.reshape(-1, 3)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            pts_opt = pts_opt / norms
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    return best_pts, best_energy

def _lbfgs_polish(pts0, maxiter=500, ftol=1e-14, gtol=1e-10):
    """Polish solution using L-BFGS-B with tight tolerances."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Flatten to 1D for scipy
    def objective(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        e, _ = _log_energy_grad(pts_norm)
        return e
    
    def gradient(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        _, g = _log_energy_grad(pts_norm)
        # Project to tangent and flatten
        g_tan = _project_to_tangent(pts_norm, g)
        return g_tan.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 3, 'ftol': ftol, 'gtol': gtol}
        )
        
        if result.success or result.fun < best_energy:
            pts_opt = result.x.reshape(-1, 3)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            pts_opt = pts_opt / norms
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    return best_pts, best_energy

def _spherical_design(n, rng=None):
    """Generate points using a spherical design approach for better initial coverage."""
    if rng is None:
        rng = np.random
    # Use golden ratio spiral with different parameters
    i = np.arange(n) + 0.5
    # Try different spiral patterns
    pattern = rng.randint(3)
    if pattern == 0:
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    elif pattern == 1:
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(2.0)) * i
    else:
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(3.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _lbfgs_optimize(pts0, maxiter=2000, ftol=1e-15, gtol=1e-12):
    """Optimize using L-BFGS-B with tight tolerances, starting from pts0."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    def objective(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts_norm = pts_2d / norms
        e, _ = _log_energy_grad(pts_norm)
        return e
    
    def gradient(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts_norm = pts_2d / norms
        _, g = _log_energy_grad(pts_norm)
        g_tan = _project_to_tangent(pts_norm, g)
        return g_tan.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': ftol, 'gtol': gtol}
        )
        
        if result.fun < best_energy:
            pts_opt = result.x.reshape(-1, 3)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = pts_opt / norms
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    return best_pts, best_energy

def _conjugate_gradient_optimize(pts0, maxiter=5000, ftol=1e-14, gtol=1e-11):
    """Optimize using conjugate gradient method on the sphere."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Initial gradient
    _, grad0 = _log_energy_grad(pts)
    grad_tan = _project_to_tangent(pts, grad0)
    
    # Conjugate gradient parameters
    beta_method = 'PR'  # Polak-Ribiere
    d = -grad_tan
    alpha = 1.0
    
    for it in range(maxiter):
        # Line search to find optimal step size
        best_alpha = alpha
        best_e = _log_energy(pts)
        
        # Backtracking line search
        alpha_ls = alpha
        for _ in range(20):
            pts_trial = pts + alpha_ls * d
            pts_trial /= np.linalg.norm(pts_trial, axis=1, keepdims=True)
            e_trial = _log_energy(pts_trial)
            if e_trial < best_e:
                best_alpha = alpha_ls
                best_e = e_trial
            else:
                alpha_ls *= 0.5
                if alpha_ls < 1e-10:
                    break
        
        if best_e >= best_energy:
            alpha *= 0.5
            if alpha < 1e-10:
                break
            continue
        
        pts = pts + best_alpha * d
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        if best_e < best_energy:
            best_energy = best_e
            best_pts = pts.copy()
        
        # Compute new gradient
        _, grad_new = _log_energy_grad(pts)
        grad_new_tan = _project_to_tangent(pts, grad_new)
        
        # Check convergence
        if np.linalg.norm(grad_new_tan) < gtol:
            break
        
        # Compute beta
        if beta_method == 'PR':
            num = np.sum(grad_new_tan * (grad_new_tan - grad_tan))
            den = np.sum(grad_tan * grad_tan) + 1e-15
            beta = max(0, num / den)
        else:
            beta = np.sum((grad_new_tan - grad_tan) * (grad_new_tan - grad_tan)) / (np.sum(grad_tan * grad_tan) + 1e-15)
        
        d = -grad_new_tan + beta * d
        grad_tan = grad_new_tan
        alpha = best_alpha
    
    return best_pts, best_energy

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 148.0  # 2 second safety margin
    
    # Start with multiple initialization strategies
    inits = []
    inits.append(_make_fibonacci_sphere())
    rng_init = np.random.RandomState(123)
    inits.append(_spherical_design(N, rng_init))
    inits.append(_random_sphere(N, np.random.RandomState(456)))
    
    best_pts = inits[0].copy()
    best_energy = _log_energy(inits[0])
    
    # Phase 1: L-BFGS from each initialization (fast convergence)
    for init_pts in inits:
        if time.time() > deadline - 20.0:
            break
        pts_opt, energy_opt = _lbfgs_optimize(init_pts, maxiter=3000, ftol=1e-15, gtol=1e-12)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 2: Multi-restart with diverse strategies
    rng = np.random.RandomState(42)
    n_restarts = 40
    perturbation_scales = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    for r in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        # Generate diverse starting points using various strategies
        strategy = r % 7
        if strategy == 0:
            # Small perturbation of best
            pts_rand = best_pts + perturbation_scales[r % len(perturbation_scales)] * rng.randn(N, 3)
        elif strategy == 1:
            # Perturbation of Fibonacci
            pts_rand = inits[0] + perturbation_scales[(r + 1) % len(perturbation_scales)] * rng.randn(N, 3)
        elif strategy == 2:
            # Random sphere
            pts_rand = _random_sphere(N, rng)
        elif strategy == 3:
            # Spherical design
            pts_rand = _spherical_design(N, rng)
        elif strategy == 4:
            # Mix of best and random
            pts_rand = 0.5 * best_pts + 0.5 * _random_sphere(N, rng)
        elif strategy == 5:
            # Large perturbation
            pts_rand = best_pts + 0.5 * rng.randn(N, 3)
        else:
            # Perturbation of second init
            pts_rand = inits[1] + perturbation_scales[(r + 2) % len(perturbation_scales)] * rng.randn(N, 3)
        
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Use L-BFGS for each restart
        maxiter = min(4000, max(800, int(remaining * 25)))
        pts_opt, energy_opt = _lbfgs_optimize(pts_rand, maxiter=maxiter, ftol=1e-14, gtol=1e-11)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 3: Conjugate gradient for faster convergence in some cases
    if time.time() < deadline - 15.0:
        remaining = max(deadline - time.time() - 12.0, 3.0)
        maxiter_cg = int(remaining * 30)
        pts_opt, energy_opt = _conjugate_gradient_optimize(best_pts, maxiter=maxiter_cg, ftol=1e-14, gtol=1e-11)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 4: Adam fine-tuning on best solution
    if time.time() < deadline - 12.0:
        remaining = max(deadline - time.time() - 10.0, 2.0)
        pts_opt, energy_opt = _adam_optimize(best_pts, time_limit=remaining, lr_init=0.001, no_improve_break=200)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 5: Final L-BFGS polish with very tight tolerances
    if time.time() < deadline - 8.0:
        pts_opt, energy_opt = _lbfgs_optimize(best_pts, maxiter=8000, ftol=1e-16, gtol=1e-13)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 6: One more ultra-tight L-BFGS pass
    if time.time() < deadline - 4.0:
        pts_opt, energy_opt = _lbfgs_optimize(best_pts, maxiter=15000, ftol=1e-16, gtol=1e-14)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 7: Final conjugate gradient polish
    if time.time() < deadline - 2.0:
        remaining = max(deadline - time.time() - 1.5, 0.5)
        maxiter_cg = int(remaining * 40)
        pts_opt, energy_opt = _conjugate_gradient_optimize(best_pts, maxiter=maxiter_cg, ftol=1e-15, gtol=1e-12)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Ensure points are on the unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    # Check for very close points and perturb if necessary
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if np.min(dist[iu]) < 1e-9:
        for _ in range(100):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
            rand_dir = np.random.randn(3)
            rand_dir /= np.linalg.norm(rand_dir)
            best_pts[j] = best_pts[j] + 1e-6 * rand_dir
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            if np.min(dist[iu]) >= 1e-9:
                break
    
    return best_pts

# EVOLVE-BLOCK-END
