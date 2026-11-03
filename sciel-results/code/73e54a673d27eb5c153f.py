"""Logarithmic energy minimization for N points on the unit sphere.
Uses multi-start L-BFGS-B optimization with diverse initializations including Sobol sequences
and an iterative polishing phase to squeeze out additional energy reduction."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
# The evaluator kills the program after 150s. We use 140s as a hard deadline to be safe.
_DEADLINE = 140.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    """Project points onto the unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation for scoring initializations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space).
    
    Args:
        pts: (N, 3) array of points on the sphere.
        
    Returns:
        energy: float
        grad: (N, 3) array of gradients projected to tangent space.
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of E = -sum log(d_ij)
    # dE/dx_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # Let W_ij = 1 / |x_i - x_j|^2 for i != j, 0 otherwise.
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # The gradient for point k is: sum_j W_kj * (x_k - x_j) = x_k * sum_j W_kj - sum_j W_kj * x_j
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    
    # Project onto tangent space: g = raw_grad - (raw_grad . x_k) x_k
    dot = np.sum(raw_grad * pts, axis=1, keepdims=True)
    grad = raw_grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for high accuracy."""
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
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    """Generate random points on the sphere."""
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence (low discrepancy)."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    # Map [0,1]^2 to sphere uniformly
    # theta in [0, pi], phi in [0, 2*pi]
    # u = 1 - 2*theta/2 -> theta = arccos(1-2u)
    # v = phi/2pi -> phi = 2*pi*v
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _repulsion_step(pts, iterations=10, step_size=0.1):
    """Simple gradient ascent on the potential to spread points out.
    This helps escape clusters in initializations."""
    for _ in range(iterations):
        e, g = _energy_grad(pts)
        # We want to minimize energy, so we move in direction of -grad
        # But for repulsion to spread, we might just do a small step
        # Actually, the gradient points towards lower energy (closer points increase energy? 
        # No, log(1/d) increases as d decreases. So gradient points towards increasing d?
        # Let's check: E = sum log(1/d). dE/dd = -1/d. 
        # dE/dx_k = sum (x_k - x_j)/d^2. 
        # If x_k and x_j are close, (x_k-x_j) is small but d^2 is very small, so term is large.
        # Direction is x_k - x_j, which pushes x_k away from x_j.
        # So -grad pushes x_k towards x_j? 
        # Wait. If I move x_k in direction (x_k - x_j), I increase distance.
        # So gradient points in direction of increasing energy (closer points).
        # So -gradient points in direction of decreasing energy (further points).
        # So standard gradient descent minimizes energy.
        # "Repulsion" usually implies maximizing minimum distance, which is different.
        # But for log energy, minimizing energy *is* maximizing distances on average.
        # So a few steps of gradient descent should spread them out if they start clustered.
        pts = pts - step_size * g
        pts = _project(pts)
    return pts


def _optimize(pts, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B optimization from a starting point."""
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
                'gtol': gtol,
                'ftol': ftol,
                'maxls': 50,
                'maxcor': 20
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


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
    deadline = start_time + 135.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy) - Primary source of good starts
    for s in range(30):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(15):
        jitter_scale = 0.05 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds
    for s in range(20):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
        
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations and sort
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    # L-BFGS is faster, so we can try more candidates
    top_candidates = scored_inits[:60]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 15.0:
            break
        
        # Run optimization
        optimized = _optimize(init_pts, max_iter=1000, ftol=1e-15, gtol=1e-12)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Iterative Polish Phase
    # Take the best result, perturb it slightly, and re-optimize.
    # This can help escape shallow local minima or refine the solution further.
    if best_pts is not None:
        current_best = best_pts.copy()
        current_e = best_energy
        
        # Try a few rounds of perturbation and re-optimization
        for polish_round in range(5):
            if time.time() > deadline - 5.0:
                break
                
            # Generate a few perturbations
            found_improvement = False
            for p in range(3):
                if time.time() > deadline - 5.0:
                    break
                    
                # Small random perturbation
                noise = 0.01 * rng.standard_normal((N, 3))
                perturbed = current_best + noise
                perturbed = _project(perturbed)
                
                # Optimize the perturbed version
                optimized = _optimize(perturbed, max_iter=500, ftol=1e-15, gtol=1e-12)
                e_precise = _precise_energy(optimized)
                
                if e_precise < current_e:
                    current_e = e_precise
                    current_best = optimized.copy()
                    found_improvement = True
            
            if not found_improvement:
                break
                
        if current_e < best_energy:
            best_energy = current_e
            best_pts = current_best

    # Final high-precision polish on the absolute best
    if best_pts is not None and time.time() < deadline - 2.0:
        refined = _optimize(best_pts, max_iter=2000, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    if not _valid(best_pts):
        # Fallback to a known valid configuration
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
