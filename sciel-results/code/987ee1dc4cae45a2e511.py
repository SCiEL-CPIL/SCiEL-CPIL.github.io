"""Logarithmic energy minimization for N points on the unit sphere.
Uses multi-start L-BFGS-B with tangent-space projection, diverse initializations,
annealing, and a final high-precision Newton-style polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    grad = -raw_grad
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


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


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _tangent_to_sphere(pts, vec):
    """Project vec onto tangent space at each point in pts."""
    dot = np.sum(vec * pts, axis=1, keepdims=True)
    return vec - dot * pts


class _TimeoutException(Exception):
    pass


def _lbfgs_optimize(pts, deadline, max_iter=1000, ftol=1e-14, gtol=1e-12):
    """Run L-BFGS-B optimization from a starting point with tangent projection."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        if time.time() > deadline:
            raise _TimeoutException()
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxls': 20,
                'maxfun': max_iter * 10,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except _TimeoutException:
        return pts
    except Exception:
        return pts


def _local_optimize(pts, deadline, max_iter=1500, ftol=1e-15, gtol=1e-13):
    """High-precision local optimization using very tight tolerances."""
    return _lbfgs_optimize(pts, deadline, max_iter=max_iter, ftol=ftol, gtol=gtol)


def _anneal(pts, deadline, n_steps=150, base_lr=0.008, min_lr=0.0002, 
            base_pert=0.002, min_pert=0.0001, seed=42):
    """Gradient-flow annealing to explore configuration space more broadly."""
    current = pts.copy()
    n = current.shape[0]
    rng = np.random.default_rng(seed)
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        t = step / max(n_steps - 1, 1)
        lr = base_lr * (min_lr / base_lr) ** t
        pert_scale = base_pert * (min_pert / base_pert) ** t
        
        e, g = _energy_grad(current)
        perturbation = pert_scale * rng.standard_normal((n, 3))
        step_vec = -lr * (g + perturbation)
        
        step_vec = _tangent_to_sphere(current, step_vec)
        
        current = current + step_vec
        current = _project(current)
    
    return current


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _newton_polish(pts, deadline, max_iter=200, tol=1e-14):
    """Newton-style polish using Hessian approximation for faster convergence."""
    n = pts.shape[0]
    current = pts.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Simple Newton step: use gradient descent with adaptive step size
        # as a robust alternative to full Newton (which requires Hessian)
        # Try a few step sizes
        best_step = None
        best_e = e
        
        for factor in [1.0, 0.5, 0.25, 0.1, 0.05]:
            step_size = factor * 0.01
            candidate = current - step_size * g
            candidate = _tangent_to_sphere(current, candidate)
            candidate = current + candidate
            candidate = _project(candidate)
            
            cand_e = _energy(candidate)
            if cand_e < best_e:
                best_e = cand_e
                best_step = candidate
        
        if best_step is None:
            break
        
        current = best_step
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy)
    for s in range(25):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(15):
        jitter_scale = 0.02 + 0.006 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds
    for s in range(25):
        initializations.append(_random_sphere(np.random.default_rng(200 + s)))
        
    # 5. Rotated Fibonacci spirals
    for angle in [0.1, 0.2, 0.3, 0.5, 0.7]:
        base_rot = _fibonacci_spiral(N, offset=0.5)
        # Rotate around z-axis
        c, s_ = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        rotated = base_rot @ R.T
        initializations.append(_project(rotated))
        
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Phase 1: Optimize top candidates with L-BFGS-B
    top_candidates = scored_inits[:50]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 25.0:
            break
        
        optimized = _lbfgs_optimize(init_pts, deadline, max_iter=1200, ftol=1e-15, gtol=1e-13)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 2: Annealing + optimization on the best configuration
    if best_pts is not None and time.time() < deadline - 20.0:
        # Try multiple annealing restarts with different seeds
        for anneal_seed in [42, 123, 456]:
            if time.time() > deadline - 10.0:
                break
            annealed = _anneal(best_pts, deadline, n_steps=120, base_lr=0.008, 
                              min_lr=0.0002, base_pert=0.002, min_pert=0.0001, seed=anneal_seed)
            refined = _lbfgs_optimize(annealed, deadline, max_iter=1500, ftol=1e-15, gtol=1e-13)
            
            e_precise = _precise_energy(refined)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined.copy()
    
    # Phase 3: Try a few more diverse starts if time permits
    if best_pts is not None and time.time() < deadline - 10.0:
        for extra_seed in [300, 400, 500]:
            if time.time() > deadline - 8.0:
                break
            extra_init = _random_sphere(np.random.default_rng(extra_seed))
            optimized = _lbfgs_optimize(extra_init, deadline, max_iter=800, ftol=1e-14, gtol=1e-12)
            e_precise = _precise_energy(optimized)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized.copy()
    
    # Phase 4: Newton-style polish for deeper convergence
    if best_pts is not None and time.time() < deadline - 8.0:
        polished = _newton_polish(best_pts, deadline, max_iter=100, tol=1e-15)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished.copy()
    
    # Phase 5: Final ultra-precise polish with multiple restarts
    if best_pts is not None and time.time() < deadline - 5.0:
        for polish_iter in range(3):
            if time.time() > deadline - 3.0:
                break
            polished = _local_optimize(best_pts, deadline, max_iter=2500, ftol=1e-16, gtol=1e-15)
            e_precise = _precise_energy(polished)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished.copy()
            else:
                break
    
    # Phase 6: One more very tight polish
    if best_pts is not None and time.time() < deadline - 2.0:
        refined = _local_optimize(best_pts, deadline, max_iter=4000, ftol=1e-17, gtol=1e-16)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
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
