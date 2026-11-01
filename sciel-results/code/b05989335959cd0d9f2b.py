"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _compute_facet_data(points):
    """Compute convex hull and facet circumradii. Returns (max_radius, hull, points)."""
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    points = points / norms
    
    # Check for coincident points
    d = cdist(points, points)
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-10:
        return np.pi, None, points
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, None, points
    
    max_radius = 0.0
    simplices = hull.simplices
    
    for simplex in simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        
        u_dir = np.cross(a - b, b - c)
        u_norm = np.linalg.norm(u_dir)
        if u_norm < 1e-12:
            R = max(d_ab, d_bc, d_ca) / 2
        else:
            u = u_dir / u_norm
            cos_R1 = np.clip(np.dot(u, a), -1, 1)
            cos_R2 = np.clip(-np.dot(u, a), -1, 1)
            R1 = np.arccos(cos_R1)
            R2 = np.arccos(cos_R2)
            R = min(R1, R2)
            
            n_ab = np.cross(a, b)
            n_bc = np.cross(b, c)
            n_ca = np.cross(c, a)
            
            n_facet = np.cross(b - a, c - a)
            if np.dot(n_facet, a) < 0:
                n_facet = -n_facet
                n_ab = -n_ab
                n_bc = -n_bc
                n_ca = -n_ca
            
            inside = True
            if np.dot(n_ab, u) * np.dot(n_ab, c) < 0:
                inside = False
            if np.dot(n_bc, u) * np.dot(n_bc, a) < 0:
                inside = False
            if np.dot(n_ca, u) * np.dot(n_ca, b) < 0:
                inside = False
            
            if not inside:
                R = max(d_ab, d_bc, d_ca) / 2
        
        if R > max_radius:
            max_radius = R
    
    return max_radius, hull, points


def _soft_max_radius_and_grad(points, beta):
    """Compute soft-max covering radius and its gradient using a grid."""
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = points / norms
    
    # Grid
    n_theta = 120
    n_phi = 240
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi)
    
    grid_x = np.sin(THETA) * np.cos(PHI)
    grid_y = np.sin(THETA) * np.sin(PHI)
    grid_z = np.cos(THETA)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid_norms = np.linalg.norm(grid, axis=1, keepdims=True)
    grid_norms[grid_norms < 1e-12] = 1.0
    grid = grid / grid_norms
    
    # Angular distances: dots[i,j] = grid_i . pts_j
    dots = grid @ pts.T  # (n_grid, N)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)  # (n_grid, N)
    
    min_angles = np.min(angles, axis=1)  # (n_grid,)
    
    # Soft max
    exp_neg = np.exp(-beta * min_angles)
    log_sum = np.log(np.sum(exp_neg))
    soft_max = -log_sum / beta
    
    # Gradient: d(soft_max)/d(pts_k)
    # soft_max = -1/beta * log(sum_i exp(-beta * min_angle_i))
    # d(soft_max)/d(min_angle_i) = exp(-beta * min_angle_i) / sum_j exp(-beta * min_angle_j)
    weights = exp_neg / np.sum(exp_neg)  # (n_grid,)
    
    # d(min_angle_i)/d(pts_k): for each grid point i, the nearest point k_i
    nearest_idx = np.argmin(angles, axis=1)  # (n_grid,)
    
    grad = np.zeros_like(pts)
    for i in range(len(nearest_idx)):
        k = nearest_idx[i]
        w = weights[i]
        # d(angle)/d(pts_k) where angle = arccos(grid_i . pts_k)
        # d(arccos(x))/dx = -1/sqrt(1-x^2)
        # x = grid_i . pts_k
        # dx/d(pts_k) = grid_i
        dot_val = dots[i, k]
        if abs(1.0 - dot_val * dot_val) > 1e-12:
            d_angle_d_pts = -grid[i] / np.sqrt(1.0 - dot_val * dot_val)
        else:
            d_angle_d_pts = np.zeros(3)
        grad[k] += w * d_angle_d_pts
    
    # Project gradient to tangent space
    for k in range(N):
        grad[k] -= np.dot(grad[k], pts[k]) * pts[k]
    
    return soft_max, grad


def _project_to_sphere(x):
    """Project flat vector to sphere (N*3 -> Nx3 unit vectors)."""
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _lbfgs_stage(points, start_time, max_iter, beta, ftol, gtol):
    """Run L-BFGS-B optimization stage."""
    def objective_and_grad(x):
        if time.time() - start_time > DEADLINE - 5:
            return 1e10, np.zeros_like(x)
        pts = _project_to_sphere(x)
        val, grad = _soft_max_radius_and_grad(pts, beta)
        # Gradient in flat space: grad_flat[k] = grad_sphere[k] (since projection is identity on sphere)
        grad_flat = grad.ravel()
        return val, grad_flat
    
    x0 = points.ravel()
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                         jac=True,
                         options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                  'maxls': 50})
        if result.success or result.fun < 1e10:
            opt_pts = _project_to_sphere(result.x)
            return opt_pts
    except Exception:
        pass
    return points


def _local_polish(points, start_time):
    """Final polish with high beta and tight tolerances."""
    if time.time() - start_time > DEADLINE - 15:
        return points
    
    # Stage 1: beta=200
    pts = _lbfgs_stage(points, start_time, 100, 200, 1e-14, 1e-12)
    
    # Stage 2: beta=500
    if time.time() - start_time < DEADLINE - 10:
        pts = _lbfgs_stage(pts, start_time, 80, 500, 1e-15, 1e-13)
    
    # Stage 3: beta=800
    if time.time() - start_time < DEADLINE - 5:
        pts = _lbfgs_stage(pts, start_time, 50, 800, 1e-16, 1e-14)
    
    return pts


def _generate_seeds(rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # Random configurations with varied noise
    for r in range(60):
        noise_scale = 0.05 * (0.7 ** (r // 10))
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r_base = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r_base * np.cos(phi), r_base * np.sin(phi), z])
        noise = rng.standard_normal((N, 3)) * noise_scale
        pts += noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    
    # Pure random
    for _ in range(10):
        pts = rng.standard_normal((N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    
    return configs


def _lloyd_relaxation(points, max_iters=30, start_time=None):
    """Lloyd's algorithm for spherical k-means."""
    if start_time is None:
        start_time = time.time()
    
    points = np.array(points, dtype=np.float64)
    
    # Coarser grid for speed
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi)
    
    grid_x = np.sin(THETA) * np.cos(PHI)
    grid_y = np.sin(THETA) * np.sin(PHI)
    grid_z = np.cos(THETA)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid_norms = np.linalg.norm(grid, axis=1, keepdims=True)
    grid_norms[grid_norms < 1e-12] = 1.0
    grid = grid / grid_norms
    
    for iteration in range(max_iters):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        dists = cdist(grid, points)
        nearest = np.argmin(dists, axis=1)
        
        new_points = np.zeros_like(points)
        for i in range(N):
            mask = nearest == i
            if np.any(mask):
                mean_vec = np.sum(grid[mask], axis=0)
                mean_norm = np.linalg.norm(mean_vec)
                if mean_norm > 1e-12:
                    new_points[i] = mean_vec / mean_norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        diff = np.linalg.norm(new_points - points)
        points = new_points
        if diff < 1e-10:
            break
    
    return points


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Generate seeds
    configs = _generate_seeds(rng)
    
    best_points = None
    best_radius = np.pi
    
    # Phase 1: Quick Lloyd relaxation on all seeds, keep best
    for config in configs:
        if time.time() - start_time > DEADLINE - 60:
            break
        
        relaxed = _lloyd_relaxation(config, max_iters=20, start_time=start_time)
        radius, _, _ = _compute_facet_data(relaxed)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed.copy()
    
    # Phase 2: L-BFGS polish on the best
    if best_points is not None and time.time() - start_time < DEADLINE - 20:
        best_points = _local_polish(best_points, start_time)
        best_radius, _, _ = _compute_facet_data(best_points)
    
    # Phase 3: Try a few more seeds if time permits
    if best_points is not None and time.time() - start_time < DEADLINE - 40:
        # Take the best and perturb it
        for attempt in range(3):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            noise = rng.standard_normal((N, 3)) * 0.02
            perturbed = best_points + noise
            perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            relaxed = _lloyd_relaxation(perturbed, max_iters=15, start_time=start_time)
            relaxed = _local_polish(relaxed, start_time)
            
            radius, _, _ = _compute_facet_data(relaxed)
            if radius < best_radius:
                best_radius = radius
                best_points = relaxed.copy()
    
    if best_points is None:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final normalization
    best_points = np.asarray(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
