# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/d) and its gradient.
    
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    mask = dist_sq > 1e-30
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        energy = 0.0
    else:
        energy = math.fsum(np.log(1.0 / d).tolist())

    return energy, grad


def _energy_only(pts):
    """Compute only the energy, faster when gradient not needed."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    return math.fsum(np.log(1.0 / d).tolist())


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
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


def _random_sphere(n, seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, time_limit=5.0):
    """Run L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e = _energy_only(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 15000,
            'maxfun': 40000,
            'ftol': 1e-16,
            'gtol': 1e-14,
            'maxls': 200
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy = _energy_only(final_pts)
    return final_pts, energy


def _riemannian_polish(pts, time_limit=10.0):
    """Riemannian Gradient Descent with Armijo Line Search."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    current_energy = _energy_only(pts)
    
    alpha = 0.01
    c = 1e-4
    rho = 0.5
    max_iter = 5000
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(pts)
        # Riemannian gradient
        rgrad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        grad_norm = np.linalg.norm(rgrad)
        if grad_norm < 1e-12:
            break
            
        # Armijo Line Search
        while True:
            # Retraction: R_X(V) = (X + V) / ||X + V||
            new_pts = pts - alpha * rgrad
            new_pts = _project_to_sphere(new_pts)
            new_energy = _energy_only(new_pts)
            
            # Check Armijo condition
            if new_energy <= current_energy - c * alpha * grad_norm**2:
                break
            alpha *= rho
            if alpha < 1e-16:
                break
        
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            # Adaptive step size increase
            alpha = min(alpha * 1.1, 0.1)
        else:
            alpha *= 0.5
            
    return pts, current_energy


def _local_search(pts, time_limit=10.0):
    """Coordinate Descent Local Search: Optimize each point sequentially."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    current_energy = _energy_only(pts)
    
    # Iterate over points, optimizing each one while keeping others fixed
    # We do this for a few passes
    passes = 0
    while time.time() - start_time < time_limit:
        passes += 1
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 0.5:
                break
            
            # Define objective for point i
            # x_i is the variable, others are fixed
            # We optimize on the tangent plane of x_i
            x_i = pts[i]
            others = np.delete(pts, i, axis=0)
            
            # Build a local coordinate system (tangent basis)
            # e1, e2 are orthonormal vectors in the tangent plane
            # e1 = normalize(any vector not parallel to x_i)
            if abs(x_i[0]) < 0.9:
                e1 = np.array([1.0, 0.0, 0.0])
            else:
                e1 = np.array([0.0, 1.0, 0.0])
            e1 = e1 - np.dot(e1, x_i) * x_i
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(x_i, e1)
            
            def obj_2d(coords):
                # coords = [u, v]
                # x_i_new = cos(theta) * x_i + sin(theta) * (u*e1 + v*e2) ? 
                # Actually, simpler: x_i_new = (x_i + u*e1 + v*e2) / ||x_i + u*e1 + v*e2||
                # This is a retraction.
                vec = x_i + coords[0]*e1 + coords[1]*e2
                norm = np.linalg.norm(vec)
                if norm < 1e-15:
                    return 1e10
                new_xi = vec / norm
                
                # Compute energy contribution of point i
                # d = ||new_xi - others_j||
                # energy = sum log(1/d)
                dists = np.linalg.norm(others - new_xi, axis=1)
                dists = dists[dists > 1e-15]
                if len(dists) == 0:
                    return 0.0
                return math.fsum(np.log(1.0 / dists).tolist())
            
            def grad_2d(coords):
                # Gradient of obj_2d w.r.t u, v
                vec = x_i + coords[0]*e1 + coords[1]*e2
                norm = np.linalg.norm(vec)
                if norm < 1e-15:
                    return np.array([0.0, 0.0])
                new_xi = vec / norm
                
                # dE/dx_i = -sum (x_i - x_j) / ||x_i - x_j||^2
                diff = new_xi - others
                dist_sq = np.sum(diff*diff, axis=1)
                mask = dist_sq > 1e-30
                inv_dist_sq = np.zeros_like(dist_sq)
                inv_dist_sq[mask] = 1.0 / dist_sq[mask]
                grad_xi = -np.sum(diff * inv_dist_sq[:, np.newaxis], axis=0)
                
                # Project grad_xi to tangent plane
                grad_tangent = grad_xi - np.dot(grad_xi, new_xi) * new_xi
                
                # Project to e1, e2
                g_u = np.dot(grad_tangent, e1)
                g_v = np.dot(grad_tangent, e2)
                return np.array([g_u, g_v])
                
            x0 = np.array([0.0, 0.0])
            try:
                res = minimize(obj_2d, x0, jac=grad_2d, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.success and res.fun < obj_2d(x0) - 1e-10:
                    # Update point i
                    vec = x_i + res.x[0]*e1 + res.x[1]*e2
                    pts[i] = vec / np.linalg.norm(vec)
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
            
    final_energy = _energy_only(pts)
    return pts, final_energy


def construct():
    start_time = time.time()
    deadline = 145.0

    best_pts = None
    best_energy = np.inf

    # Generate diverse starting points
    starts = []
    starts.append(_fibonacci_sphere(N))
    starts.append(_fibonacci_sphere(N, seed=1))
    starts.append(_fibonacci_sphere(N, seed=7))
    starts.append(_fibonacci_sphere(N, seed=13))
    starts.append(_fibonacci_sphere(N, seed=42))
    for s in range(2, 10):
        starts.append(_random_sphere(N, seed=s * 100))
    starts.append(_random_sphere(N, seed=999))
    starts.append(_random_sphere(N, seed=12345))

    # Phase 1: L-BFGS-B from each diverse start
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        time_limit = min(5.0, deadline - elapsed - 25)
        if time_limit < 1.0:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts.copy()

    # Phase 2: Local Search (Coordinate Descent)
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 25:
            time_limit = min(10.0, deadline - elapsed - 20)
            candidate_pts, candidate_energy = _local_search(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 3: Riemannian Polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            time_limit = min(10.0, deadline - elapsed - 10)
            candidate_pts, candidate_energy = _riemannian_polish(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 4: Another Local Search pass
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            time_limit = min(5.0, deadline - elapsed - 5)
            candidate_pts, candidate_energy = _local_search(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 5: Final Riemannian Polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 2:
            time_limit = min(5.0, deadline - elapsed - 1)
            candidate_pts, candidate_energy = _riemannian_polish(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
