"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time
import math

N = 100
DEADLINE = 147.0  # Safety margin before 150s kill


def _energy_and_grad(points):
    """Compute logarithmic energy and Riemannian gradient for points on unit sphere."""
    n = points.shape[0]
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    log_d2 = np.log(d2)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    
    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    W = 1.0 / d2_safe
    np.fill_diagonal(W, 0.0)
    
    row_sums = W.sum(axis=1, keepdims=True)
    grad = W @ points - points * row_sums
    
    radial = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - radial * points
    
    return energy, grad


def _energy_only(points):
    """Just the energy value (no gradient)."""
    n = points.shape[0]
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    log_d2 = np.log(d2)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    return energy


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_gradient_flow(points, n_iter=5000, lr_init=0.01, lr_decay=0.999,
                               momentum=0.9, deadline=None):
    """Riemannian gradient flow with momentum and adaptive step size."""
    n = points.shape[0]
    pts = points.copy()
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    v = np.zeros_like(pts)
    lr = lr_init
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        v = momentum * v + (1 - momentum) * (-grad)
        
        step_norms = np.linalg.norm(v, axis=1, keepdims=True)
        step_norms = np.maximum(step_norms, 1e-15)
        
        cos_factor = np.cos(step_norms)
        sin_factor = np.sin(step_norms) / step_norms
        
        new_pts = cos_factor * pts + sin_factor * v
        new_pts = _normalize(new_pts)
        
        new_energy = _energy_only(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        if new_energy < energy + 1e-12:
            pts = new_pts
        else:
            lr *= 0.5
            v = v * 0.5
            continue
        
        lr = max(lr_init * (lr_decay ** it), 1e-6)
    
    return best_pts, best_energy


def _fibonacci_spiral(n, offset=0.5, scale=1.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * scale
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _shifted_spiral(n, shift):
    """Generate a shifted spiral pattern."""
    i = np.arange(n) + 0.5 + shift
    z = 1.0 - 2.0 * np.clip(i / n, 0, 1)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + shift * n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _sobol_points(n, seed=42):
    """Generate well-distributed points using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    samples = sampler.random(n)
    z = 2.0 * samples[:, 0] - 1.0
    theta = 2.0 * np.pi * samples[:, 1]
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _normalize(pts)


def _structured_lattice(n, rng):
    """Generate a structured lattice-like distribution with perturbation."""
    n_theta = int(np.sqrt(n))
    n_phi = n // n_theta
    if n_phi * n_theta < n:
        n_phi += 1
    
    thetas = np.linspace(0, np.pi, n_theta + 1)[1:-1]
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()
    
    if len(theta_flat) > n:
        theta_flat = theta_flat[:n]
        phi_flat = phi_flat[:n]
    elif len(theta_flat) < n:
        extra_n = n - len(theta_flat)
        theta_flat = np.concatenate([theta_flat, rng.uniform(0, np.pi, extra_n)])
        phi_flat = np.concatenate([phi_flat, rng.uniform(0, 2*np.pi, extra_n)])
    
    theta_flat += rng.normal(0, 0.05, n)
    phi_flat += rng.normal(0, 0.05, n)
    
    x = np.sin(theta_flat) * np.cos(phi_flat)
    y = np.sin(theta_flat) * np.sin(phi_flat)
    z = np.cos(theta_flat)
    pts = np.column_stack([x, y, z])
    return _normalize(pts)


def _lbfgs_optimize(points, max_iter=2000, ftol=1e-12, gtol=1e-8, deadline=None):
    """Run L-BFGS-B optimization with deadline guard."""
    start = time.time()
    
    def fun(x):
        if deadline is not None and time.time() > deadline:
            raise Exception("Deadline exceeded")
        return _energy_only(_normalize(x.reshape(N, 3)))
    
    def jac(x):
        if deadline is not None and time.time() > deadline:
            raise Exception("Deadline exceeded")
        pts = _normalize(x.reshape(N, 3))
        _, grad = _energy_and_grad(pts)
        return grad.flatten()
    
    try:
        result = minimize(
            fun=fun,
            x0=points.flatten(),
            jac=jac,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        if result.fun < np.inf:
            return _normalize(result.x.reshape(N, 3)), result.fun
    except Exception:
        pass
    return points, _energy_only(points)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci spiral
    base = _fibonacci_spiral(N)
    perturbed = base + 0.05 * rng.standard_normal((N, 3))
    initializations.append(_normalize(perturbed))
    
    # 3. Shifted spirals
    for shift in [0.1, 0.3, 0.7]:
        initializations.append(_shifted_spiral(N, shift))
    
    # 4. Sobol-based points
    for seed in [42, 123, 456, 789, 101, 202, 333, 555]:
        initializations.append(_sobol_points(N, seed=seed))
    
    # 5. Structured lattice with perturbation
    initializations.append(_structured_lattice(N, rng))
    
    # 6. Random points
    for _ in range(5):
        initializations.append(_random_points(N, rng))
    
    # 7. Anti-symmetric pairs
    half = N // 2
    pts_half = _random_points(half, rng)
    pts_full = np.vstack([pts_half, -pts_half])
    initializations.append(_normalize(pts_full))
    
    # 8. More perturbed spirals
    for scale in [0.8, 1.2, 1.5]:
        initializations.append(_fibonacci_spiral(N, scale=scale))
    
    # 9. Additional perturbed versions
    for _ in range(3):
        p = _fibonacci_spiral(N) + 0.1 * rng.standard_normal((N, 3))
        initializations.append(_normalize(p))
    
    # Pre-compute initial energies to sort by quality
    init_energies = []
    for init_pts in initializations:
        init_energies.append(_energy_only(init_pts))
    
    # Sort initializations by energy (best first)
    sorted_indices = np.argsort(init_energies)
    sorted_initializations = [initializations[i] for i in sorted_indices]
    
    # Track best
    best_points = sorted_initializations[0].copy()
    best_energy = init_energies[sorted_indices[0]]
    
    # Phase 1: For each top initialization, do gradient flow + L-BFGS-B
    n_inits = min(len(sorted_initializations), 12)
    time_per_init = (DEADLINE - 30.0) / n_inits  # Reserve 30s for final polish
    
    for init_idx in range(n_inits):
        if time.time() > deadline - 30.0:
            break
        
        init_pts = sorted_initializations[init_idx]
        current = init_pts.copy()
        
        # Phase 1a: Riemannian gradient flow
        time_left_for_flow = max(0, time_per_init * 0.35)
        flow_deadline = time.time() + time_left_for_flow
        flow_iters = int(time_left_for_flow * 800)
        flow_iters = max(200, min(flow_iters, 10000))
        
        try:
            flow_pts, flow_energy = _riemannian_gradient_flow(
                current, n_iter=flow_iters, lr_init=0.005, lr_decay=0.9995,
                momentum=0.9, deadline=flow_deadline
            )
            if flow_energy < best_energy:
                best_energy = flow_energy
                best_points = flow_pts.copy()
            current = flow_pts
        except Exception:
            pass
        
        if time.time() > deadline - 30.0:
            break
        
        # Phase 1b: L-BFGS-B from the flow result
        time_left_for_lbfgs = max(0, time_per_init * 0.65)
        lbfgs_deadline = time.time() + time_left_for_lbfgs
        max_iter = int(time_left_for_lbfgs * 300)
        max_iter = max(200, min(max_iter, 5000))
        
        try:
            lbfgs_pts, lbfgs_energy = _lbfgs_optimize(
                current, max_iter=max_iter, ftol=1e-12, gtol=1e-8, deadline=lbfgs_deadline
            )
            if lbfgs_energy < best_energy:
                best_energy = lbfgs_energy
                best_points = lbfgs_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Final high-precision polish
    if time.time() < deadline - 10.0:
        # Strategy A: L-BFGS-B with very tight tolerances
        try:
            polish_deadline = deadline - 8.0
            polish_pts, polish_energy = _lbfgs_optimize(
                best_points, max_iter=10000, ftol=1e-15, gtol=1e-13, deadline=polish_deadline
            )
            if polish_energy < best_energy:
                best_energy = polish_energy
                best_points = polish_pts.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 8.0:
        # Strategy B: Riemannian gradient flow with very small steps for final polish
        try:
            polish_deadline = deadline - 5.0
            polish_pts, polish_energy = _riemannian_gradient_flow(
                best_points, n_iter=20000, lr_init=0.0003, lr_decay=0.99995,
                momentum=0.95, deadline=polish_deadline
            )
            if polish_energy < best_energy:
                best_energy = polish_energy
                best_points = polish_pts.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 5.0:
        # Strategy C: One more L-BFGS-B pass with tightest tolerances
        try:
            polish_deadline = deadline - 2.0
            polish_pts, polish_energy = _lbfgs_optimize(
                best_points, max_iter=5000, ftol=1e-16, gtol=1e-14, deadline=polish_deadline
            )
            if polish_energy < best_energy:
                best_energy = polish_energy
                best_points = polish_pts.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_points = _normalize(best_points)
    
    # Check for any points too close together and nudge if needed
    for _nudge in range(20):
        min_d = np.inf
        min_pair = (0, 1)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        if min_d >= 1e-9:
            break
        i, j = min_pair
        direction = best_points[i] - best_points[j]
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-15:
            direction = direction / norm_dir
        else:
            direction = rng.standard_normal(3)
            direction = direction / np.linalg.norm(direction)
        best_points[i] = best_points[i] + 1e-8 * direction
        best_points[j] = best_points[j] - 1e-8 * direction
        best_points = _normalize(best_points)
    
    # Final verification
    final_energy = _energy_only(best_points)
    if not np.isfinite(final_energy):
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral(N)
    
    return best_points
