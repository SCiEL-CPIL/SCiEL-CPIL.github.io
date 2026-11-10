import numpy as np
import time
import scipy.spatial
import scipy.optimize

N = 50
TIME_LIMIT = 120.0  # Safety margin from 150s

_global_start_time = time.time()

def _compute_circumradii_smooth(points, beta=50.0):
    """
    Compute soft-max of squared circumradii of the facets of the convex hull.
    Returns (soft_max_value, max_sq_rad).
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return 1e10, 1e10
    
    theta_sq_list = []
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-12:
            theta_sq_list.append(1e10)
            continue
            
        c = n / norm_n
        dot = np.dot(c, v1)
        dot = np.clip(dot, -1.0, 1.0)
        
        if dot < 0:
            c = -c
            dot = -dot
            
        theta = np.arccos(dot)
        theta_sq = theta * theta
        theta_sq_list.append(theta_sq)
    
    theta_sq_arr = np.array(theta_sq_list)
    max_sq_rad = np.max(theta_sq_arr)
    
    # Soft-max via log-sum-exp
    shifted = beta * theta_sq_arr
    shifted -= np.max(shifted)
    soft_max = np.max(theta_sq_arr) + np.log(np.sum(np.exp(shifted))) / beta
    
    return soft_max, max_sq_rad

def _objective(x, beta=50.0):
    """
    Objective: smooth max of squared circumradii + sphere constraint penalty.
    """
    points = x.reshape(N, 3)
    
    norms = np.linalg.norm(points, axis=1)
    sphere_penalty = np.sum((norms - 1.0)**2) * 50.0
    
    soft_max, _ = _compute_circumradii_smooth(points, beta)
    
    return soft_max + sphere_penalty

def _repulsive_dynamics(points, steps=50, step_size=0.01, k=3):
    """
    Repulsive force-based dynamics to spread points.
    Each point is pushed away from its k nearest neighbors.
    This helps escape local minima where points are clustered.
    """
    pts = points.copy()
    
    for step in range(steps):
        elapsed = time.time() - _global_start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        # Compute pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    dists[i, j] = np.linalg.norm(pts[i] - pts[j])
        
        # Find k nearest neighbors for each point
        forces = np.zeros_like(pts)
        for i in range(N):
            # Sort neighbors by distance
            neighbor_dists = dists[i].copy()
            neighbor_dists[i] = np.inf  # Exclude self
            sorted_indices = np.argsort(neighbor_dists)
            k_nearest = sorted_indices[:k]
            
            # Compute repulsive forces from k nearest neighbors
            for j in k_nearest:
                if j == i:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Force is proportional to 1/d^2 in the direction away from neighbor
                direction = (pts[i] - pts[j]) / d
                force_magnitude = 1.0 / (d * d + 0.01)
                forces[i] += force_magnitude * direction
        
        # Update positions with forces, then project back to sphere
        pts = pts + step_size * forces
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
    
    return pts

def _local_optimize(start_points, beta_start=30.0, beta_end=150.0, total_iters=300):
    """
    Run local optimization with adaptive beta annealing and repulsive dynamics.
    """
    x0 = start_points.flatten()
    current = x0.copy()
    
    # Anneal beta from low to high
    n_phases = 4
    betas = np.linspace(beta_start, beta_end, n_phases)
    iters_per_phase = total_iters // n_phases
    
    for i, beta in enumerate(betas):
        elapsed_remaining = TIME_LIMIT - (time.time() - _global_start_time)
        if elapsed_remaining < 5:
            break
        
        # Apply repulsive dynamics before optimization to spread points
        if i == 0:
            pts = current.reshape(N, 3)
            pts = _repulsive_dynamics(pts, steps=30, step_size=0.005, k=3)
            current = pts.flatten()
        
        try:
            res = scipy.optimize.minimize(
                _objective, 
                current, 
                args=(beta,),
                method='L-BFGS-B', 
                options={'maxiter': iters_per_phase, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            current = res.x
        except Exception:
            pass
    
    x_opt = current.reshape(N, 3)
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    _, max_sq_rad = _compute_circumradii_smooth(x_opt, beta=200.0)
    score = np.sqrt(max_sq_rad)
    
    return x_opt, score

def _polish(points, maxiter=300):
    """
    Final high-precision polish with multiple passes at increasing beta.
    """
    x0 = points.flatten()
    
    for beta in [80.0, 120.0, 200.0]:
        elapsed_remaining = TIME_LIMIT - (time.time() - _global_start_time)
        if elapsed_remaining < 5:
            break
        
        try:
            res = scipy.optimize.minimize(
                _objective, 
                x0, 
                args=(beta,),
                method='L-BFGS-B', 
                options={'maxiter': maxiter, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x0 = res.x
        except Exception:
            pass
    
    x_opt = x0.reshape(N, 3)
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    return x_opt

def _fibonacci_sphere(n):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    """Generate random points on sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def construct():
    global _global_start_time
    _global_start_time = time.time()
    start_time = time.time()
    best_points = None
    best_score = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    # More diverse random starts
    for _ in range(10):
        starts.append(_random_sphere(N, rng))
    
    # Perturbed Fibonacci with varying scales
    fib = _fibonacci_sphere(N)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        perturbed = fib + scale * rng.standard_normal((N, 3))
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        starts.append(perturbed)
    
    # Antipodal pairs perturbation
    antipodal = np.vstack([fib[:N//2], -fib[:N//2]])
    if antipodal.shape[0] < N:
        extra = _random_sphere(N - antipodal.shape[0], rng)
        antipodal = np.vstack([antipodal, extra])
    antipodal = antipodal[:N]
    starts.append(antipodal)
    
    for idx, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
        
        points, score = _local_optimize(start, beta_start=20.0, beta_end=120.0, total_iters=200)
        
        if score < best_score:
            best_score = score
            best_points = points
    
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 15 and best_points is not None:
        best_points = _polish(best_points, maxiter=250)
        _, max_sq_rad = _compute_circumradii_smooth(best_points, beta=200.0)
        best_score = np.sqrt(max_sq_rad)
    
    # Final check and normalize
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
