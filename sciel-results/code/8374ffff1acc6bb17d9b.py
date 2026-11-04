"""Optimize 150 points on the unit sphere to minimize logarithmic energy."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

N = 150


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (fast vectorized)."""
    n = len(pts)
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)

    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(dist_upper))

    dist_sq = dist ** 2
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum

    # Project gradient onto tangent space
    dot = np.sum(grad * pts, axis=1)
    grad_tangent = grad - dot[:, np.newaxis] * pts

    return energy, grad_tangent


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        return 0.0
    return -np.sum(np.log(dist_upper))


def _project_to_sphere(pts):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _lbfgs_optimize(pts, time_limit=5.0, ftol=1e-12, gtol=1e-10, maxiter=2000):
    """L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    try:
        result = minimize(
            objective_flat,
            x0,
            jac=grad_flat,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'maxfun': 5000,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        final_pts = _project_to_sphere(result.x.reshape(-1, 3))
        energy, _ = _energy_and_grad(final_pts)
        return final_pts, energy
    except Exception:
        return pts, _energy_only(pts)


def _riemannian_line_search(pts, time_limit=10.0, seed=42):
    """Riemannian gradient descent with Armijo line search on the sphere."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    lr_initial = 0.05
    lr_min = 0.001
    c1 = 1e-4  # Armijo condition
    rho = 0.5  # Backtracking factor
    max_ls_iter = 30

    momentum = 0.0
    velocity = np.zeros_like(pts)
    momentum_decay = 0.9
    momentum_increase = 0.05

    lr = lr_initial
    no_improve = 0
    patience = 500

    for iteration in range(100000):
        if time.time() - start_time > time_limit:
            break

        # Compute tangent gradient
        grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

        # Update momentum
        velocity = momentum_decay * velocity - lr * grad_tan
        momentum = min(0.99, momentum + momentum_increase)

        # Armijo line search
        step = velocity
        step_norm_sq = np.sum(step ** 2)
        if step_norm_sq < 1e-20:
            break

        # Ensure step is tangent
        step_tan = step - np.sum(step * pts, axis=1, keepdims=True) * pts
        step_norm_sq = np.sum(step_tan ** 2)
        if step_norm_sq < 1e-20:
            break

        alpha = 1.0
        grad_dot_step = np.sum(grad_tan * step_tan)
        
        found = False
        for ls_iter in range(max_ls_iter):
            # Exponential map: move along geodesic
            # For small steps, this is approximately pts + alpha * step_tan
            # But we should use the proper spherical interpolation
            new_pts = pts + alpha * step_tan
            new_pts = _project_to_sphere(new_pts)
            
            new_energy = _energy_only(new_pts)
            
            # Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * grad^T * d
            if new_energy <= current_energy + c1 * alpha * grad_dot_step:
                found = True
                break
            
            alpha *= rho
        
        if not found:
            # If line search fails, try a very small step
            alpha = 1e-6
            new_pts = pts + alpha * step_tan
            new_pts = _project_to_sphere(new_pts)
            new_energy = _energy_only(new_pts)
        
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            no_improve = 0
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            no_improve += 1
            # Reduce learning rate if no improvement
            lr = max(lr_min, lr * 0.9)
            
            if no_improve >= patience:
                break

        # Decay learning rate
        lr = max(lr_min, lr * 0.999)

        # Recompute gradient
        _, grad = _energy_and_grad(pts)

        # Check convergence
        if np.linalg.norm(grad_tan) < 1e-12:
            break

    return best_pts, best_energy


def _adam_optimize(pts, time_limit=10.0, seed=42):
    """Adam optimizer on the sphere with adaptive learning rate."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    lr_initial = 0.02
    lr_min = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_decay = 0.995

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    lr = lr_initial

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    no_improve_count = 0
    patience = 500

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        m_tangent = m_hat - np.sum(m_hat * pts, axis=1, keepdims=True) * pts

        pts = pts - lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)

        lr = max(lr_min, lr * lr_decay)

        if t % 100 == 0 and np.linalg.norm(grad_tangent) < 1e-10:
            break

    return best_pts, best_energy


def _polish(pts, time_limit=5.0):
    """Final high-precision polish with very tight tolerances."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    try:
        result = minimize(
            objective_flat,
            x0,
            jac=grad_flat,
            method='L-BFGS-B',
            options={
                'maxiter': 5000,
                'maxfun': 10000,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 100
            }
        )
        final_pts = _project_to_sphere(result.x.reshape(-1, 3))
        energy, _ = _energy_and_grad(final_pts)
        return final_pts, energy
    except Exception:
        return pts, _energy_only(pts)


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill

    # Start with Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_pts = _project_to_sphere(best_pts)
    best_energy, _ = _energy_and_grad(best_pts)

    rng = np.random.default_rng(42)

    # Phase 1: L-BFGS from Fibonacci start
    if time.time() - start_time < time_limit - 60.0:
        pts, energy = _lbfgs_optimize(best_pts, time_limit=10.0, ftol=1e-12, gtol=1e-10)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy

    # Phase 2: Riemannian line search for escape from local minima
    if time.time() - start_time < time_limit - 40.0:
        pts, energy = _riemannian_line_search(best_pts, time_limit=15.0)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy

    # Phase 3: Adam optimization for fine-tuning
    if time.time() - start_time < time_limit - 30.0:
        pts, energy = _adam_optimize(best_pts, time_limit=10.0)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy

    # Phase 4: Multiple random restarts with L-BFGS
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 20.0:
            break

        # Random perturbation of best solution
        noise_scale = 0.05 * (restart + 1)
        noise = rng.standard_normal((N, 3)) * noise_scale
        x0 = _project_to_sphere(best_pts + noise)

        pts, energy = _lbfgs_optimize(x0, time_limit=5.0, ftol=1e-12, gtol=1e-10)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy

    # Phase 5: Final polish with tight tolerances
    if time.time() - start_time < time_limit - 5.0:
        pts, energy = _polish(best_pts, time_limit=5.0)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy

    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)

    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] = _project_to_sphere(best_pts[j] + rng.standard_normal(3) * 1e-6)

    # Final normalization
    best_pts = _project_to_sphere(best_pts)

    return best_pts
