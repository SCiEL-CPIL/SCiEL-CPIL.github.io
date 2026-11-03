<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Only take upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Filter out any zero distances (shouldn't happen, but safe)
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    # Use math.fsum for accuracy as specified
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())


def _log_energy_fast(pts):
    """Fast logarithmic energy computation for optimization"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Set diagonal to 1
    np.fill_diagonal(dist_sq, 1.0)
    # Only take upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    d = np.sqrt(d_sq)
    # Filter out any very small distances
    mask = d > 1e-15
    d = d[mask]
    if len(d) == 0:
        return 0.0
    return np.sum(np.log(1.0 / d))


def _project_to_sphere(pts):
    """Project points to unit sphere"""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere"""
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


def _spherical_gradient_descent(pts, lr=0.01, max_iter=2000, tol=1e-8, time_limit=10.0):
    """Vectorized Gradient descent on the sphere for logarithmic energy minimization"""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    
    prev_energy = _log_energy_fast(pts)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradient of energy
        # E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
        # dE/dxi = sum_{j!=i} (xi - xj) / |xi - xj|^2
        
        # Vectorized gradient computation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        # grad[i] = sum_j (pts[i] - pts[j]) / |pts[i] - pts[j]|^2
        # This is equivalent to sum_j diff[i,j] / dist_sq[i,j]
        # We need to mask the diagonal where dist_sq is 1.0 (which is not 0, so it's fine, but the term should be 0)
        # The term for j=i is 0 because diff[i,i] is 0.
        
        # inv_dist_sq = 1.0 / dist_sq
        # grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # To be safe and fast:
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update
        pts = pts - lr * grad_tangent
        pts = _project_to_sphere(pts)
        
        energy = _log_energy_fast(pts)
        
        if abs(prev_energy - energy) < tol:
            break
        prev_energy = energy
    
    return pts, _log_energy_fast(pts)


def _local_search(pts, n_restarts=10, time_limit=100):
    """Local search with multiple restarts"""
    start_time = time.time()
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    # Try different starting points
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Estimate remaining time for this restart
        # Leave some buffer for the next restart and final checks
        remaining = time_limit - elapsed
        if remaining < 2.0:
            break
            
        # Allocate time for this restart, leaving a small buffer
        t_alloc = max(1.0, remaining - 1.0)
        
        # Generate different starting configurations
        if restart == 0:
            start_pts = pts.copy()
        elif restart == 1:
            start_pts = _fibonacci_sphere(len(pts), seed=restart)
        elif restart == 2:
            # Random uniform on sphere
            rng = np.random.RandomState(restart * 100)
            start_pts = rng.randn(len(pts), 3)
            start_pts = start_pts / np.linalg.norm(start_pts, axis=1, keepdims=True)
        else:
            # Perturbed version of best
            rng = np.random.RandomState(restart * 200)
            perturbation = rng.randn(len(pts), 3) * 0.1
            start_pts = best_pts + perturbation
            start_pts = _project_to_sphere(start_pts)
        
        # Run gradient descent with time limit
        result_pts, result_energy = _spherical_gradient_descent(start_pts, lr=0.01, max_iter=1000, time_limit=t_alloc)
        
        if result_energy < best_energy:
            best_energy = result_energy
            best_pts = result_pts
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin (evaluator kills at 120s/150s)
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    # Run local search
    # Use a time limit that leaves 10s for final processing
    local_time_limit = deadline - 10.0
    best_pts, best_energy = _local_search(pts, n_restarts=10, time_limit=local_time_limit)
    
    # Check time
    if time.time() - start_time > deadline:
        return _project_to_sphere(best_pts)
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: check for very close points
    # Vectorized min distance check
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Points too close, use Fibonacci instead
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient on the sphere.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient: grad[i] = sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy: sum over upper triangle of log(1/d)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere"""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(pts, vec):
    """Project vector to tangent space at each point"""
    return vec - np.sum(vec * pts, axis=1, keepdims=True) * pts


def _exponential_map(pts, tangent_vec, step_size=1.0):
    """Exponential map on the sphere: move along geodesic from pts by tangent_vec * step_size"""
    # For unit sphere: exp_p(v) = cos(|v|) * p + sin(|v|)/|v| * v
    # where v is in tangent space
    v = tangent_vec * step_size
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    # Handle zero vectors
    mask = v_norm > 1e-15
    result = pts.copy()
    if np.any(mask):
        vn = v_norm[mask]
        cos_term = np.cos(vn)
        sin_term = np.sin(vn) / vn
        result[mask] = cos_term * pts[mask] + sin_term * v[mask]
    return _project_to_sphere(result)


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere"""
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


def _icosahedron_based(n):
    """Generate points based on icosahedron vertices, then subdivide/refine"""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    if n <= 12:
        # Just use first n icosahedron vertices
        return ico[:n]
    
    # Start with icosahedron, add points via Fibonacci for the rest
    # Better: use a denser pattern
    # For now, blend icosahedron symmetry with Fibonacci
    ico_pts = ico.copy()
    fib_pts = _fibonacci_sphere(n, seed=42)
    
    # Mix: use icosahedron as anchor, Fibonacci for density
    # Actually, let's just use Fibonacci with a good seed
    return fib_pts


def _random_sphere(n, seed):
    """Random uniform points on sphere"""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _riemannian_cg(pts, max_iter=2000, tol=1e-10, time_limit=30.0):
    """Riemannian conjugate gradient with exponential map updates.
    
    Uses Polak-Ribiere momentum in tangent space.
    """
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    start_time = time.time()
    
    energy, grad = _energy_and_grad(pts)
    grad_t = _tangent_project(pts, grad)
    
    # Initial direction = negative gradient
    d_t = -grad_t.copy()
    
    # Line search parameters
    c1 = 1e-4  # Armijo parameter
    max_ls_iter = 20
    
    prev_grad_t = grad_t.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Check convergence
        if np.linalg.norm(grad_t) < tol:
            break
        
        # Backtracking line search in tangent space
        step = 1.0
        new_energy = energy
        new_pts = pts
        new_grad_t = grad_t
        
        for ls_iter in range(max_ls_iter):
            new_pts = _exponential_map(pts, d_t, step)
            new_energy, new_grad = _energy_and_grad(new_pts)
            new_grad_t = _tangent_project(new_pts, new_grad)
            
            # Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * <grad, d>
            if new_energy <= energy + c1 * step * np.sum(grad_t * d_t):
                break
            step *= 0.5
            if step < 1e-12:
                break
        
        # Update
        pts = new_pts
        energy = new_energy
        grad_t = new_grad_t
        
        # Polak-Ribiere momentum
        if it > 0:
            # beta = max(0, <grad_t - prev_grad_t, -d_t> / ||prev_grad_t||^2)
            num = np.sum((grad_t - prev_grad_t) * (-d_t))
            den = np.sum(prev_grad_t * prev_grad_t)
            if den > 1e-15:
                beta = max(0.0, num / den)
            else:
                beta = 0.0
            d_t = -grad_t + beta * d_t
        else:
            d_t = -grad_t
        
        prev_grad_t = grad_t
    
    return pts, energy


def _polish(pts, max_iter=5000, tol=1e-12, time_limit=20.0):
    """High-precision polish with very tight tolerance"""
    pts = _project_to_sphere(pts.copy())
    start_time = time.time()
    
    energy, grad = _energy_and_grad(pts)
    grad_t = _tangent_project(pts, grad)
    
    # Small step gradient descent with adaptive step size
    step_size = 0.01
    prev_energy = energy
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        if np.linalg.norm(grad_t) < tol:
            break
        
        # Adaptive step: reduce if energy increases
        new_pts = _exponential_map(pts, -grad_t, step_size)
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            grad_t = _tangent_project(pts, new_grad)
            # Can increase step slightly
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-10:
                break
        
        if abs(prev_energy - energy) < tol * 0.1:
            break
        prev_energy = energy
    
    return pts, energy


def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin (evaluator kills at 150s)
    
    # Generate diverse starting configurations
    n = N
    starts = []
    
    # Fibonacci with different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        starts.append(_fibonacci_sphere(n, seed=seed))
    
    # Random starts
    for seed in [100, 200, 300]:
        starts.append(_random_sphere(n, seed=seed))
    
    # Icosahedron-based
    starts.append(_icosahedron_based(n))
    
    best_pts = None
    best_energy = np.inf
    
    # Phase 1: Riemannian CG on each start
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 30.0  # Leave 30s for polish + final
        if remaining < 5.0:
            break
        
        t_alloc = max(3.0, remaining / max(1, len(starts) - i))
        result_pts, result_energy = _riemannian_cg(start_pts, max_iter=3000, tol=1e-10, time_limit=t_alloc)
        
        if result_energy < best_energy:
            best_energy = result_energy
            best_pts = result_pts
    
    # Phase 2: Polish the best result
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 5.0
    if remaining > 2.0 and best_pts is not None:
        best_pts, best_energy = _polish(best_pts, max_iter=5000, tol=1e-12, time_limit=remaining)
    
    # Phase 3: Try a few more starts if time permits
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 10.0
    if remaining > 5.0 and best_pts is not None:
        # Perturb best and try again
        rng = np.random.RandomState(999)
        for attempt in range(3):
            elapsed = time.time() - start_time
            remaining = deadline - elapsed - 5.0
            if remaining < 3.0:
                break
            
            perturbation = rng.randn(n, 3) * 0.05
            start_pts = _project_to_sphere(best_pts + perturbation)
            t_alloc = max(2.0, remaining / 3.0)
            result_pts, result_energy = _riemannian_cg(start_pts, max_iter=2000, tol=1e-10, time_limit=t_alloc)
            
            if result_energy < best_energy:
                best_energy = result_energy
                best_pts = result_pts
    
    # Final projection and validation
    if best_pts is None:
        best_pts = _fibonacci_sphere(n)
    
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Points too close, use Fibonacci instead
        return _fibonacci_sphere(n)
    
    return best_pts
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
