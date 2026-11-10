import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import itertools

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 580  # Safety margin before 600s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if np.any(dists == 0):
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x, penalty):
        """
        Minimize negative log(d_min/d_max) with barrier for coincident points.
        Points are constrained to unit ball via bounds in SLSQP.
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-15:
            return 1e10
        if d_min < 1e-15:
            return 1e10 + penalty * (1.0 / d_min)
        
        # We want to maximize log(d_min) - log(d_max), so minimize its negative
        val = -(np.log(d_min) - np.log(d_max))
        # Add soft barrier if any distance is too small
        close_dists = dists[dists < 0.1 * d_max]
        if len(close_dists) > 0:
            barrier = penalty * np.sum(1.0 / close_dists)
        else:
            barrier = 0.0
        
        return val + barrier

    def unit_ball_constraint(x):
        """Constraint: each point must be in unit ball (norm <= 1)."""
        p = x.reshape((n, d))
        norms = np.linalg.norm(p, axis=1)
        return 1.0 - norms  # Must be >= 0

    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci sphere
    phi_fib = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / (phi_fib * phi_fib)
    i = np.arange(n)
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(1 - z**2)
    theta = golden_angle * i
    fib_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    initializations.append(fib_pts)
    
    # 2. Icosahedron vertices + 2 additional points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
    # Add 2 points: try opposite faces
    face_normal1 = ico_verts[0] + ico_verts[1] + ico_verts[5]
    face_normal1 = face_normal1 / np.linalg.norm(face_normal1)
    face_normal2 = ico_verts[2] + ico_verts[3] + ico_verts[7]
    face_normal2 = face_normal2 / np.linalg.norm(face_normal2)
    ico_plus = np.vstack([ico_verts, [face_normal1, face_normal2]])
    initializations.append(ico_plus)
    
    # 3. Octahedron-based: 6 axis points + 8 face centers
    axis_pts = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float)
    face_pts = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=float)
    face_pts = face_pts / np.linalg.norm(face_pts, axis=1)[:, np.newaxis]
    octa_plus = np.vstack([axis_pts, face_pts])
    initializations.append(octa_plus)
    
    # 4. Random initializations with different seeds
    for seed in range(20):
        rng = np.random.RandomState(seed)
        rand_pts = rng.randn(n, d)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        initializations.append(rand_pts)
    
    # 5. Perturbed Fibonacci
    for pert_scale in [0.1, 0.3, 0.5]:
        rng = np.random.RandomState(100 + int(pert_scale * 100))
        perturbed = fib_pts + pert_scale * rng.randn(n, d)
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1)[:, np.newaxis]
        initializations.append(perturbed)
    
    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    # Optimization with penalty annealing
    penalty_schedule = [1.0, 10.0, 100.0, 1000.0]
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - start_time > time_limit - 30:
            break
        
        current_pts = init_pts.copy()
        
        for penalty in penalty_schedule:
            if time.time() - start_time > time_limit - 15:
                break
            
            try:
                # Use SLSQP with unit ball constraint
                bounds = [(-1.0, 1.0)] * (n * d)
                
                res = minimize(
                    objective, 
                    current_pts.ravel(), 
                    method='SLSQP',
                    args=(penalty,),
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        # Scale to have max distance = 1 for consistency
                        dists_opt = pdist(opt_points)
                        max_d = np.max(dists_opt)
                        if max_d > 1e-15:
                            opt_points = opt_points / max_d
                        
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                        
                        current_pts = opt_points.copy()
                        
            except Exception:
                pass
    
    # Final local refinement with high penalty
    if time.time() - start_time < time_limit - 10:
        try:
            bounds = [(-1.0, 1.0)] * (n * d)
            res = minimize(
                objective,
                best_points.ravel(),
                method='SLSQP',
                args=(10000.0,),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if res.fun < 1e9:
                final_pts = res.x.reshape((n, d))
                if np.all(np.isfinite(final_pts)):
                    dists_final = pdist(final_pts)
                    max_d = np.max(dists_final)
                    if max_d > 1e-15:
                        final_pts = final_pts / max_d
                    final_score = get_score(final_pts)
                    if final_score > best_score:
                        best_points = final_pts
                        best_score = final_score
        except Exception:
            pass
    
    # Safety check
    if not np.all(np.isfinite(best_points)):
        best_points = initializations[0].copy()
    
    return best_points
