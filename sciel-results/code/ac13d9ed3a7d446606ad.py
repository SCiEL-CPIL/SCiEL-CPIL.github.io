"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 148.0
_START = time.time()

# Precompute upper triangle indices for N=100
_IU = np.triu_indices(N, k=1)


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Returns (energy, gradient) where gradient has shape (N, 3).
    The gradient is projected onto the tangent space of the sphere.
    """
    n = pts.shape[0]
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    W_row_sum = W.sum(axis=1, keepdims=True)
    grad = W @ pts - W_row_sum * pts
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= dot * pts
    
    return float(energy), grad


def _fast_energy_only(pts):
    """Compute just the logarithmic energy (faster, no gradient)."""
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    return float(-0.5 * np.sum(np.log(dist_sq[_IU])))


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate quasi-random points on the sphere using Sobol sequence."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    m = max(1, int(np.ceil(np.log2(n))))
    while (1 << m) < n:
        m += 1
    points = sampler.random(1 << m)[:n]
    theta = 2.0 * np.pi * points[:, 0]
    z = 2.0 * points[:, 1] - 1.0
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def _make_objective(n):
    """Create objective function that normalizes to sphere."""
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    return objective_and_grad


def _lbfgs_pass(pts, deadline, maxiter, maxfun, ftol, gtol, maxls):
    """Run a single L-BFGS-B pass with given parameters, return (pts, energy) if improved."""
    n = pts.shape[0]
    if time.time() > deadline - 2.0:
        return pts, _precise_energy(pts)
    
    obj = _make_objective(n)
    try:
        result = minimize(
            obj, pts.reshape(-1),
            method='L-BFGS-B', jac=True,
            options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': ftol, 'gtol': gtol, 'maxls': maxls}
        )
        new_pts = _project(result.x.reshape(n, 3))
        e_precise = _precise_energy(new_pts)
        if np.isfinite(e_precise) and e_precise < _precise_energy(pts):
            return new_pts, e_precise
    except Exception:
        pass
    return pts, _precise_energy(pts)


def _gradient_flow(pts, deadline, max_steps=300):
    """Riemannian gradient flow with backtracking line search."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _fast_energy_only(pts)
    
    for step_iter in range(max_steps):
        if time.time() > deadline - 3.0:
            break
        e, g = _fast_energy_and_grad(pts)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        direction = -g
        lr = 0.05 / (1.0 + 0.005 * step_iter)
        accepted = False
        for backtrack in range(20):
            pts_new = pts + lr * direction
            pts_new = _project(pts_new)
            e_new = _fast_energy_only(pts_new)
            armijo_bound = e + 1e-4 * lr * (-g_norm ** 2)
            if e_new < armijo_bound and e_new < e:
                pts = pts_new
                e = e_new
                accepted = True
                break
            lr *= 0.5
        if not accepted:
            lr = 0.005
            pts_new = pts + lr * direction
            pts_new = _project(pts_new)
            e_new = _fast_energy_only(pts_new)
            if e_new < e:
                pts = pts_new
                e = e_new
            else:
                break
        
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
    
    return best_pts, _precise_energy(best_pts)


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.5, 0.3, 0.7, 0.1, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(6):
        amp = 0.01 + 0.015 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(8):
        initializations.append(_random_sphere(100 + s * 17))
    
    # Golden ratio spiral
    i = np.arange(N)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / N
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    theta = golden_angle * i
    golden_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    initializations.append(_project(golden_pts))
    
    # Sobol sequence initializations
    for sob_seed in range(4):
        initializations.append(_sobol_sphere(N, seed=sob_seed))
    
    # Additional structured starts
    # Two-hemisphere pattern
    top = _fibonacci_spiral(50, offset=0.5)
    bottom = _fibonacci_spiral(50, offset=0.5) * np.array([1, 1, -1])
    initializations.append(_project(np.vstack([top, bottom])))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase A: Quick screen — short optimization from each init to find best basin
    init_energies = []
    for idx, init_pts in enumerate(initializations):
        e_init = _fast_energy_only(init_pts)
        init_energies.append((e_init, idx, init_pts))
    init_energies.sort(key=lambda x: x[0])
    
    # Allocate time: try top initializations with more budget
    for e_init, idx, init_pts in init_energies:
        if time.time() > deadline - 40.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Short optimization to identify basin
        alloc_check = time.time() + 5.0
        if alloc_check > deadline - 35.0:
            break
        
        try:
            obj = _make_objective(N)
            result = minimize(
                obj, init_pts.reshape(-1),
                method='L-BFGS-B', jac=True,
                options={'maxiter': 150, 'maxfun': 800, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            new_pts = _project(result.x.reshape(N, 3))
            e_precise = _precise_energy(new_pts)
            if np.isfinite(e_precise) and e_precise < best_energy:
                best_energy = e_precise
                best_pts = new_pts
        except Exception:
            pass
    
    # Phase B: Deep optimization from the best found so far
    if best_pts is not None and time.time() < deadline - 30.0:
        deep_deadline = time.time() + min(deadline - time.time() - 20.0, 50.0)
        current = best_pts.copy()
        
        # Progressive refinement with L-BFGS-B
        configs = [
            (300, 1500, 1e-12, 1e-8, 30),
            (500, 3000, 1e-14, 1e-10, 40),
            (700, 4500, 1e-15, 1e-12, 50),
            (900, 6000, 1e-15, 1e-13, 60),
            (1200, 8000, 1e-16, 1e-14, 70),
        ]
        
        for maxiter, maxfun, ftol, gtol, maxls in configs:
            if time.time() > deep_deadline - 3.0:
                break
            current, e_precise = _lbfgs_pass(current, deep_deadline, maxiter, maxfun, ftol, gtol, maxls)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = current
    
    # Phase C: Gradient flow for fine-tuning
    if best_pts is not None and time.time() < deadline - 15.0:
        flow_deadline = time.time() + min(deadline - time.time() - 10.0, 25.0)
        current, e_precise = _gradient_flow(best_pts, flow_deadline, max_steps=200)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = current
    
    # Phase D: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 8.0:
        polish_deadline = time.time() + min(deadline - time.time() - 5.0, 15.0)
        configs = [
            (800, 5000, 1e-15, 1e-13, 60),
            (1000, 7000, 1e-16, 1e-14, 70),
            (1500, 10000, 1e-16, 1e-15, 80),
        ]
        current = best_pts.copy()
        for maxiter, maxfun, ftol, gtol, maxls in configs:
            if time.time() > polish_deadline - 2.0:
                break
            current, e_precise = _lbfgs_pass(current, polish_deadline, maxiter, maxfun, ftol, gtol, maxls)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = current
    
    # Phase E: One more gradient flow pass on the polished result
    if best_pts is not None and time.time() < deadline - 4.0:
        flow_deadline = time.time() + min(deadline - time.time() - 2.0, 8.0)
        current, e_precise = _gradient_flow(best_pts, flow_deadline, max_steps=100)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = current
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
