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

def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space).
    
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = sum_{j != i} (x_j - x_i) / d_ij^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # Gradient: dE/dx_i = sum_j W_ij*(x_j - x_i)
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

def _sobol_sphere(n, seed=42):
    """Generate well-distributed points using a Sobol-like sequence."""
    rng = np.random.default_rng(seed)
    # Use a combination of low-discrepancy sequences
    points = []
    for i in range(n):
        # Use a Halton-like sequence with different bases
        x = 0.0
        y = 0.0
        z = 0.0
        # Simple deterministic spread
        x = (i * 0.618033988749895) % 1.0
        y = (i * 0.7548776662466928) % 1.0
        z = (i * 0.3819660112501051) % 1.0
        # Map to sphere using Fibonacci-like distribution
        theta = 2 * math.pi * x
        phi = math.acos(2 * y - 1)
        px = math.sin(phi) * math.cos(theta)
        py = math.sin(phi) * math.sin(theta)
        pz = math.cos(phi)
        points.append([px, py, pz])
    return np.array(points)

def _optimize(pts, deadline, max_iter=1000, ftol=1e-15, gtol=1e-12):
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

def _polish_gradient_descent(pts, max_iter=5000, tol=1e-14):
    """Custom gradient descent with adaptive step size for final polish."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _precise_energy(pts)
    step = 0.5
    
    for it in range(max_iter):
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        # Adaptive step size with line search
        candidate = best_pts - step * g
        candidate = _project(candidate)
        cand_e = _precise_energy(candidate)
        if cand_e < best_e:
            best_e = cand_e
            best_pts = candidate
            step = min(step * 1.2, 1.0)
        else:
            # Backtracking line search
            for _ in range(10):
                step *= 0.5
                candidate = best_pts - step * g
                candidate = _project(candidate)
                cand_e = _precise_energy(candidate)
                if cand_e < best_e:
                    best_e = cand_e
                    best_pts = candidate
                    break
            else:
                if step < 1e-10:
                    break
    return best_pts

def _riemannian_perturb(pts, rng, scale=0.3):
    """Perturb points using Riemannian geometry (tangent space perturbation)."""
    n = pts.shape[0]
    perturbed = pts.copy()
    for i in range(n):
        # Generate random tangent vector
        v = rng.standard_normal(3)
        # Project to tangent space
        v = v - np.dot(v, pts[i]) * pts[i]
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            v = v / v_norm * scale
            # Exponential map: move along geodesic
            new_pt = pts[i] * math.cos(scale) + v * math.sin(scale)
            perturbed[i] = new_pt
    return _project(perturbed)

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
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Random points
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Sobol-like points
    for s in range(5):
        initializations.append(_sobol_sphere(N, seed=200 + s))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: quick evaluation and optimization of top candidates
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        # Quick energy check
        e_quick = _precise_energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top 5 candidates
    for e_quick, init_pts in scored_inits[:5]:
        if time.time() > deadline - 15.0:
            break
        optimized = _optimize(init_pts, deadline, max_iter=1000, ftol=1e-13, gtol=1e-9)
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Basin Hopping on the best candidate
    if best_pts is not None:
        current_pts = best_pts.copy()
        current_e = best_energy
        kick_scale = 0.5
        
        # Simple Basin Hopping: Kick, Optimize, Accept if better
        max_hops = 20
        for hop in range(max_hops):
            if time.time() > deadline - 10.0:
                break
            
            # Kick using Riemannian perturbation
            kicked = _riemannian_perturb(current_pts, rng, scale=kick_scale)
            
            # Optimize
            optimized = _optimize(kicked, deadline, max_iter=500, ftol=1e-12, gtol=1e-8)
            e_new = _precise_energy(optimized)
            
            if e_new < current_e:
                current_e = e_new
                current_pts = optimized
                # Reduce kick scale slightly as we find better minima
                kick_scale *= 0.9
            else:
                # If no improvement, keep current
                pass
        
        if current_e < best_energy:
            best_energy = current_e
            best_pts = current_pts

    # Final polish
    if best_pts is not None and time.time() < deadline - 5.0:
        polished = _polish_gradient_descent(best_pts, max_iter=5000, tol=1e-15)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
            
    # One more L-BFGS-B pass
    if best_pts is not None and time.time() < deadline - 1.0:
        final_refined = _optimize(best_pts, deadline, max_iter=2000, ftol=1e-16, gtol=1e-15)
        e_precise = _precise_energy(final_refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = final_refined

    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
