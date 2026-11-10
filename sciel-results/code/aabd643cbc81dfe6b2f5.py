import numpy as np
import time
import scipy.spatial
import scipy.optimize

N = 50
TIME_LIMIT = 145.0  # Safety margin from 150s

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

def _compute_circumradii_fast(points):
    """
    Fast computation of squared circumradii for all hull facets.
    Returns array of squared circumradii.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return np.array([1e10])
    
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
            dot = -dot
            
        theta = np.arccos(dot)
        theta_sq = theta * theta
        theta_sq_list.append(theta_sq)
    
    return np.array(theta_sq_list)

def _objective(x, beta=50.0):
    """
    Objective: smooth max of squared circumradii + sphere constraint penalty.
    """
    points = x.reshape(N, 3)
    
    norms = np.linalg.norm(points, axis=1)
    sphere_penalty = np.sum((norms - 1.0)**2) * 50.0
    
    soft_max, _ = _compute_circumradii_smooth(points, beta)
    
    return soft_max + sphere_penalty

def _repulsive_dynamics(points, steps=100, base_step_size=0.005, k=5):
    """
    Repulsive force-based dynamics to spread points on the sphere.
    Each point experiences repulsion from its k nearest neighbors,
    scaled inversely by local density (smaller angular distance = stronger repulsion).
    """
    pts = points.copy().astype(np.float64)
    
    for step in range(steps):
        elapsed = time.time() - _global_start_time
        if elapsed > TIME_LIMIT - 15:
            break
        
        # Compute pairwise angular distances using dot products (vectorized)
        dots = np.dot(pts, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # angular distance matrix
        
        # Set diagonal to inf so points don't repel themselves
        np.fill_diagonal(angles, np.inf)
        
        # Find k nearest neighbors for each point
        k_nearest_idx = np.argsort(angles, axis=1)[:, :k]
        
        # Compute repulsive forces
        forces = np.zeros_like(pts)
        for i in range(N):
            neighbors = k_nearest_idx[i]
            # Angular distances to k nearest neighbors
            ang_dists = angles[i, neighbors]
            
            # Repulsion strength: stronger for closer neighbors
            # Use inverse angular distance as weight, with a floor to avoid division by zero
            weights = 1.0 / np.maximum(ang_dists, 0.05)
            
            # Normalize weights
            weights /= np.sum(weights)
            
            # Force direction: away from each neighbor (tangent to sphere)
            for j_idx, neighbor_idx in enumerate(neighbors):
                neighbor = pts[neighbor_idx]
                # Vector from neighbor to point i
                diff = pts[i] - neighbor
                # Project onto tangent plane at pts[i]
                tangent = diff - np.dot(diff, pts[i]) * pts[i]
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    tangent /= norm_t
                    forces[i] += weights[j_idx] * tangent
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-10)
        forces /= force_norms
        
        # Adaptive step size: smaller when forces are large
        step_size = base_step_size * (1.0 + 0.5 * np.cos(step * 0.1))
        
        # Update positions
        pts += step_size * forces
        
        # Project back to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts /= norms
    
    return pts

def _local_optimize(start_points, beta_start=20.0, beta_end=150.0, total_iters=250, use_repulsion=True):
    """
    Run local optimization with adaptive beta annealing, optionally preceded by repulsive dynamics.
    """
    points = start_points.copy()
    
    # Optional repulsive dynamics pre-step
    if use_repulsion:
        elapsed = time.time() - _global_start_time
        if elapsed < TIME_LIMIT - 30:
            points = _repulsive_dynamics(points, steps=80, base_step_size=0.004, k=5)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    x0 = points.flatten()
    current = x0.copy()
    
    # Anneal beta from low to high
    n_phases = 4
    betas = np.linspace(beta_start, beta_end, n_phases)
    iters_per_phase = total_iters // n_phases
    
    for i, beta in enumerate(betas):
        elapsed_remaining = TIME_LIMIT - (time.time() - _global_start_time)
        if elapsed_remaining < 8:
            break
        
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
    
    for beta in [80.0, 120.0, 200.0, 300.0]:
        elapsed_remaining = TIME_LIMIT - (time.time() - _global_start_time)
        if elapsed_remaining < 8:
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
    for _ in range(8):
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
    
    # Run local optimization from each start
    for idx, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 25:
            break
        
        # Use repulsion for first few starts (more expensive), skip for later ones
        use_repulsion = idx < 5
        
        points, score = _local_optimize(start, beta_start=20.0, beta_end=150.0, 
                                         total_iters=200, use_repulsion=use_repulsion)
        
        if score < best_score:
            best_score = score
            best_points = points
    
    # Polish the best result
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 15 and best_points is not None:
        best_points = _polish(best_points, maxiter=250)
        _, max_sq_rad = _compute_circumradii_smooth(best_points, beta=300.0)
        best_score = np.sqrt(max_sq_rad)
    
    # Final check and normalize
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
