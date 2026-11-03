"""Logarithmic energy minimization for N points on the unit sphere.
Search-based approach: generate many varied candidates and keep the best."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space).
    
    Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient wrt x_i: sum_{j!=i} (x_j - x_i) / d_ij^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist_sq)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # W_ij = 1 / d_ij^2 for i != j, 0 on diagonal
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij * (x_j - x_i) = (W @ x)_i - x_i * sum_j W_ij
    grad = W @ pts - W.sum(axis=1, keepdims=True) * pts
    
    # Project onto tangent space
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


def _sobol_like(n, seed=0):
    """Generate well-distributed points using a simple quasi-random approach."""
    rng = np.random.default_rng(seed)
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        t = (i + 0.5) / n
        z = 1.0 - 2.0 * t
        angle = 2.0 * np.pi * i * (1.0 + 1.0 / golden_ratio) + rng.standard_normal() * 0.1
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        points.append([r * np.cos(angle), r * np.sin(angle), z])
    return _project(np.array(points))


def _optimize(pts, deadline, max_iter=1000, ftol=1e-15, gtol=1e-12):
    """Run L-BFGS-B optimization from a starting point with corrected gradient."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
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
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _steepest_descent_polish(pts, deadline, max_iter=5000):
    """Steepest descent with backtracking line search for final polish."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _energy(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, grad = _energy_grad(best_pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Normalize gradient direction
        direction = -grad / (grad_norm + 1e-30)
        
        # Backtracking line search
        step = 0.01
        improved = False
        for _ in range(20):
            # Move in direction and project back to sphere
            new_pts = best_pts + step * direction
            new_pts = _project(new_pts)
            
            new_e = _energy(new_pts)
            if new_e < best_e - 1e-15:
                best_pts = new_pts
                best_e = new_e
                improved = True
                break
            step *= 0.5
        
        if not improved:
            # Try smaller steps
            for s in [1e-4, 1e-5, 1e-6]:
                new_pts = _project(best_pts + s * direction)
                new_e = _energy(new_pts)
                if new_e < best_e - 1e-15:
                    best_pts = new_pts
                    best_e = new_e
                    break
            else:
                break
    
    return best_pts


def _pairwise_refinement(pts, deadline, max_rounds=50):
    """Iteratively optimize pairs of points for local refinement."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _energy(pts)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                # Try small perturbations of point i while keeping others fixed
                # Optimize the angle between x_i and x_j
                x_i = best_pts[i]
                x_j = best_pts[j]
                
                # Compute optimal position for x_i given all other points
                # Gradient of energy wrt x_i (excluding self)
                others = np.delete(best_pts, i, axis=0)
                dists = np.linalg.norm(others - x_i, axis=1)
                dists = np.maximum(dists, 1e-12)
                
                # Force on x_i from all other points
                force = np.sum((others - x_i) / dists[:, np.newaxis]**2, axis=0)
                
                # Project force to tangent space at x_i
                tangent_force = force - np.dot(force, x_i) * x_i
                
                if np.linalg.norm(tangent_force) > 1e-12:
                    # Small step in direction of force
                    step_size = 0.001
                    new_xi = x_i + step_size * tangent_force / (np.linalg.norm(tangent_force) + 1e-30)
                    new_xi = new_xi / np.linalg.norm(new_xi)
                    
                    new_pts = best_pts.copy()
                    new_pts[i] = new_xi
                    new_e = _energy(new_pts)
                    
                    if new_e < best_e - 1e-14:
                        best_pts = new_pts
                        best_e = new_e
                        improved = True
        
        if not improved:
            break
    
    return best_pts


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


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(10):
        jitter_scale = 0.02 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Sobol-like points
    for s in range(5):
        initializations.append(_sobol_like(N, seed=s))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: quick evaluation of all initializations
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        
        e_quick = _energy(init_pts)
        if e_quick < best_energy:
            best_energy = e_quick
            best_pts = init_pts.copy()
    
    # Second pass: optimize the top candidates with corrected gradient
    if best_pts is not None and time.time() < deadline - 20.0:
        # Sort initializations by quick energy and take top ones
        scored_inits = []
        for init_pts in initializations:
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 6 candidates (leave more time for polish)
        top_candidates = scored_inits[:6]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 15.0:
                break
            
            optimized = _optimize(init_pts, deadline, max_iter=800, ftol=1e-13, gtol=1e-9)
            e_precise = _precise_energy(optimized)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized
    
    # Final polish on the global best using steepest descent
    if best_pts is not None and time.time() < deadline - 5.0:
        refined = _steepest_descent_polish(best_pts, deadline, max_iter=3000)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    # Another L-BFGS refinement pass with tighter tolerances
    if best_pts is not None and time.time() < deadline - 3.0:
        refined = _optimize(best_pts, deadline, max_iter=1000, ftol=1e-16, gtol=1e-13)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    # Final steepest descent polish
    if best_pts is not None and time.time() < deadline - 1.0:
        refined = _steepest_descent_polish(best_pts, deadline, max_iter=2000)
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
