import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3
NP = N * (N - 1) // 2  # 91 pairwise distances

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score: max/min squared distance."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def objective_smooth(x, tau=10.0):
    """
    Smooth log-ratio objective using soft-max and soft-min.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and soft min via log-sum-exp
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    return np.log(soft_max) - np.log(soft_min)

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t subject to soft constraints.
    We use a smooth approximation where we minimize:
      log(soft_max(d_i)) - log(soft_min(d_i)) + penalty for t
    Actually, let's use a direct epigraph: minimize t + lambda * (soft_max - t)_+^2 + ...
    Simpler: just use the log-ratio with higher tau.
    """
    return objective_smooth(x, tau=tau)

def fibonacci_sphere(n_points, radius=2.0):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        r = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)

def generate_4d_polytope_vertices():
    """
    Generate vertices of the 4D 16-cell (regular polytope) projected to 3D.
    The 16-cell has 16 vertices at (±1,0,0,0), (0,±1,0,0), (0,0,±1,0), (0,0,0,±1).
    We'll optimize the 3x4 projection matrix.
    """
    # 16 vertices of the 16-cell
    vertices_4d = np.zeros((16, 4))
    for i in range(4):
        vertices_4d[i, i] = 1.0
        vertices_4d[i + 4, i] = -1.0
    return vertices_4d

def optimize_projection(vertices_4d, maxiter=200, timeout_remaining=30.0):
    """
    Optimize a 3x4 projection matrix P to project 16 4D points to 3D,
    selecting 14 of them (or all 16, but we need exactly 14).
    Actually, let's project all 16 and then pick the best 14.
    """
    start = time.time()
    
    def proj_objective(x):
        if time.time() - start > timeout_remaining:
            return 1e10
        P = x.reshape(3, 4)
        pts_3d = vertices_4d @ P.T  # (16, 3)
        # We need to pick 14 out of 16. For now, use all 16 and score them,
        # but we only need 14. Let's just score all 16 for the projection.
        # Actually, let's just use the first 14 for now as a heuristic.
        pts = pts_3d[:14]
        sq_dists = get_pairwise_sq_dists(pts)
        sq_dists = np.maximum(sq_dists, 1e-15)
        return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))
    
    # Initial projection: identity-like
    P0 = np.eye(3, 4)
    x0 = P0.flatten()
    
    try:
        res = minimize(
            proj_objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.x is not None and np.isfinite(res.fun):
            P = res.x.reshape(3, 4)
            pts_3d = vertices_4d @ P.T
            # Pick the 14 points with the best ratio
            # For simplicity, just take the first 14
            return pts_3d[:14]
    except Exception:
        pass
    
    return None

def normalize_points(points):
    """Center and scale points to unit norm for better conditioning."""
    center = np.mean(points, axis=0)
    pts = points - center
    norm = np.linalg.norm(pts)
    if norm > 1e-15:
        pts = pts / norm * 2.0
    return pts

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Grid-based initialization
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                grid_pts.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_grid = [p for p in grid_pts if p not in to_remove]
    grid_pts_arr = np.array(final_grid, dtype=float)
    grid_pts_arr = grid_pts_arr - np.mean(grid_pts_arr, axis=0)
    candidates.append(grid_pts_arr.flatten())
    
    # 2. Fibonacci sphere points
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 3. Random points on sphere with different seeds
    for seed in range(10):
        r = np.random.default_rng(seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout - 40:
            break
    
    # 4. Random points in ball
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        pts = r.normal(size=(N, D)) * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout - 40:
            break
    
    # 5. Perturbed Fibonacci
    for _ in range(3):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.2
        candidates.append(perturbed.flatten())
    
    # 6. Perturbed grid
    for _ in range(3):
        perturbed = grid_pts_arr + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed.flatten())
    
    # 7. 4D projection candidates
    if time.time() - start_time < timeout - 30:
        vertices_4d = generate_4d_polytope_vertices()
        proj_pts = optimize_projection(vertices_4d, maxiter=100, timeout_remaining=15.0)
        if proj_pts is not None:
            candidates.append(proj_pts.flatten())
    
    # Optimization with progressive tau schedule
    # Start with smooth objective to find good basin, then sharpen
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
    maxiter_schedule = [200, 300, 400, 500, 600, 800]
    
    # Optimization loop over candidates
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 15:
            break
        
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 10:
                    break
                
                obj = partial(objective_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                # Use L-BFGS-B for the main optimization
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Final high-precision polish with SLSQP at very high tau
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_final = partial(objective_smooth, tau=10000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='SLSQP',
                        options={'maxiter': 300, 'ftol': 1e-16, 'maxls': 50}
                    )
                    if res_polish.x is not None and np.isfinite(res_polish.fun):
                        x_curr = res_polish.x
                except Exception:
                    pass
            
            # Evaluate the true non-smooth score
            pts = x_curr.reshape(N, D)
            true_s = raw_score(pts)
            if np.isfinite(true_s) and true_s < best_score:
                best_score = true_s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # If optimization didn't find anything better, fall back to the best candidate
    if best_points is None:
        for init_x in candidates:
            pts = init_x.reshape(N, D)
            s = raw_score(pts)
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_points = pts.copy()
        
        if best_points is None:
            best_points = grid_pts_arr
            best_score = raw_score(best_points)

    # Final validation: ensure all points are distinct
    sq_dists = get_pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-8:
        best_points = grid_pts_arr
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
