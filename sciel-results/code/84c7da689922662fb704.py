import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 130

def _fibonacci_init(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _soft_max_circumradii(radii, beta=50.0):
    """
    Smooth approximation of max(radii) using log-sum-exp.
    As beta -> inf, this approaches max(radii).
    """
    # Shift for numerical stability
    m = np.max(radii)
    if m < -1e15:
        return 0.0
    shifted = radii - m
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50.0, 50.0)
    log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum_exp

def _circumcenter_robust(p1, p2, p3):
    """Compute circumcenter of triangle on unit sphere, robust to degeneracy."""
    edges = [
        (p2 - p1, p3 - p1),
        (p1 - p2, p3 - p2),
        (p1 - p3, p2 - p3)
    ]
    
    for v1, v2 in edges:
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        n = n / n_norm
        center = n.copy()
        
        dot1 = np.dot(center, p1)
        dot2 = np.dot(center, p2)
        dot3 = np.dot(center, p3)
        
        if dot1 > 0 and dot2 > 0 and dot3 > 0:
            return center
        else:
            return -center
    
    # Fallback: average of the three points, normalized
    center = (p1 + p2 + p3) / 3.0
    norm = np.linalg.norm(center)
    if norm < 1e-10:
        return p1.copy()
    return center / norm

def _compute_circumradii(pts, hull):
    """
    Compute angular circumradii for all triangular facets of the convex hull.
    Returns array of circumradii (in radians) and the circumcenters.
    """
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for k, tri in enumerate(simplices):
        v0 = pts[tri[0]]
        v1 = pts[tri[1]]
        v2 = pts[tri[2]]
        
        c = _circumcenter_robust(v0, v1, v2)
        centers[k] = c
        
        dot = np.clip(np.dot(c, v0), -1.0, 1.0)
        radii[k] = np.arccos(dot)
        
    return radii, centers

def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, None, None
        
    radii, centers = _compute_circumradii(pts, hull)
    max_radius = np.max(radii)
    return np.degrees(max_radius), radii, centers

def _smooth_objective(pts, beta=50.0):
    """
    Smooth objective function for optimization.
    Returns the soft-max of circumradii (in radians).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0  # Large penalty if hull fails
        
    radii, centers = _compute_circumradii(pts, hull)
    return _soft_max_circumradii(radii, beta)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _repulsive_force(pts):
    """Compute repulsive force between all pairs of points to prevent clustering."""
    n = len(pts)
    rep_force = np.zeros_like(pts)
    
    # Compute pairwise distances
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    D = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                d_ij = D[i, j]
                if d_ij < 1e-6:
                    f = 100.0
                else:
                    f = 1.0 / (d_ij * d_ij)
                rep_force[i] += f * (pts[i] - pts[j])
                
    return rep_force

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Multi-restart strategy with different seeds
    best_cr_overall = 180.0
    best_pts_overall = None
    
    n_restarts = 5
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        restart_start = time.time()
        restart_deadline = restart_start + time_per_restart
        
        # Initialize with different seed for each restart
        seed = restart * 100
        pts = _fibonacci_init(seed)
        
        best_cr = 180.0
        best_pts = pts.copy()
        
        # Pre-compute initial hull
        try:
            hull = ConvexHull(pts)
            radii, centers = _compute_circumradii(pts, hull)
            current_cr = np.degrees(np.max(radii))
        except Exception:
            continue
        
        if current_cr < best_cr:
            best_cr = current_cr
            best_pts = pts.copy()
        
        step_size_greedy = 0.05
        step_size_lloyd = 0.02
        no_improve_count = 0
        max_no_improve = 30
        iteration = 0
        
        # Main loop: Lloyd/Greedy with periodic smooth optimization
        while time.time() < restart_deadline:
            iteration += 1
            
            # Alternate between Lloyd and Greedy
            use_lloyd = (iteration % 3 == 0)
            
            if use_lloyd:
                # Lloyd step: move each point towards centroid of adjacent facet circumcenters
                simplices = hull.simplices
                point_to_facets = [[] for _ in range(N)]
                for k, tri in enumerate(simplices):
                    for idx in tri:
                        point_to_facets[idx].append(k)
                        
                new_pts = pts.copy()
                for i in range(N):
                    facets = point_to_facets[i]
                    if not facets:
                        continue
                        
                    sum_c = np.sum(centers[facets], axis=0)
                    norm_c = np.linalg.norm(sum_c)
                    if norm_c < 1e-10:
                        continue
                        
                    target = sum_c / norm_c
                    new_pts[i] = pts[i] + step_size_lloyd * (target - pts[i])
                    
                new_pts = _normalize(new_pts)
                pts = new_pts
            else:
                # Greedy: Shrink the largest facet
                max_idx = np.argmax(radii)
                max_rad = radii[max_idx]
                
                if max_rad < 1e-6:
                    break
                    
                tri = hull.simplices[max_idx]
                c = centers[max_idx]
                
                moves = np.zeros(3)
                for idx in tri:
                    v = pts[idx]
                    # Project c onto tangent plane at v
                    proj = np.dot(v, c) * c
                    tangent = c - proj
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        moves += tangent / norm_t
                        
                move_norm = np.linalg.norm(moves)
                if move_norm < 1e-10:
                    # If stuck, try a small random perturbation
                    pts += np.random.normal(0, 0.001, (N, 3))
                    pts = _normalize(pts)
                else:
                    moves = moves / move_norm
                    for idx in tri:
                        pts[idx] += step_size_greedy * moves
                        pts[idx] /= np.linalg.norm(pts[idx])
            
            # Update hull and radii every few iterations to save time
            if iteration % 5 == 0 or iteration == 1:
                try:
                    hull = ConvexHull(pts)
                    radii, centers = _compute_circumradii(pts, hull)
                    new_cr = np.degrees(np.max(radii))
                    
                    if new_cr < best_cr:
                        best_cr = new_cr
                        best_pts = pts.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count > max_no_improve:
                            # Apply repulsive force to break symmetry
                            rep = _repulsive_force(pts)
                            pts = pts + 0.01 * rep
                            pts = _normalize(pts)
                            
                            # Reduce step sizes
                            step_size_greedy *= 0.9
                            step_size_lloyd *= 0.9
                            no_improve_count = 0
                            if step_size_greedy < 1e-5:
                                break
                except Exception:
                    # Hull failed, revert to best
                    pts = best_pts.copy()
                    break
            
            # Smooth optimization polish using log-sum-exp surrogate
            # Every 15 iterations, if we have time, try to optimize using smooth objective
            if iteration % 15 == 0 and time.time() < restart_deadline - 10.0:
                try:
                    # Identify indices of points in the top 5 largest facets
                    top_facet_indices = np.argsort(radii)[-5:]
                    point_indices_to_optimize = set()
                    for f_idx in top_facet_indices:
                        tri = hull.simplices[f_idx]
                        for idx in tri:
                            point_indices_to_optimize.add(idx)
                    
                    point_indices_to_optimize = list(point_indices_to_optimize)
                    
                    # Define smooth objective function
                    def smooth_obj(x_flat):
                        # Reconstruct pts
                        temp_pts = pts.copy()
                        for k, idx in enumerate(point_indices_to_optimize):
                            temp_pts[idx] = x_flat[3*k:3*k+3]
                        temp_pts = _normalize(temp_pts)
                        
                        # Calculate smooth covering radius
                        return _smooth_objective(temp_pts, beta=50.0)
                    
                    # Initial guess
                    x0 = np.array([pts[idx] for idx in point_indices_to_optimize]).flatten()
                    
                    # Run BFGS on smooth objective
                    result = minimize(smooth_obj, x0, method='BFGS', 
                                      options={'maxiter': 30, 'gtol': 1e-4})
                    
                    if result.success:
                        # Update pts with the optimized coordinates
                        for k, idx in enumerate(point_indices_to_optimize):
                            pts[idx] = result.x[3*k:3*k+3]
                        pts = _normalize(pts)
                        
                        # Update hull and best
                        hull = ConvexHull(pts)
                        radii, centers = _compute_circumradii(pts, hull)
                        new_cr = np.degrees(np.max(radii))
                        if new_cr < best_cr:
                            best_cr = new_cr
                            best_pts = pts.copy()
                            
                except Exception:
                    pass # Ignore polish errors
        
        # After each restart, update overall best
        if best_cr < best_cr_overall:
            best_cr_overall = best_cr
            best_pts_overall = best_pts.copy()
    
    # Final high-precision polish on the best configuration
    if best_pts_overall is not None and time.time() < deadline - 5.0:
        try:
            pts = best_pts_overall.copy()
            
            # Final L-BFGS-B polish with tighter tolerance
            def final_obj(x_flat):
                temp_pts = x_flat.reshape(-1, 3)
                temp_pts = _normalize(temp_pts)
                return _smooth_objective(temp_pts, beta=100.0)
            
            x0 = pts.flatten()
            
            result = minimize(final_obj, x0, method='L-BFGS-B', 
                              options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-10})
            
            if result.success:
                final_pts = _normalize(result.x.reshape(-1, 3))
                try:
                    hull = ConvexHull(final_pts)
                    radii, centers = _compute_circumradii(final_pts, hull)
                    final_cr = np.degrees(np.max(radii))
                    if final_cr < best_cr_overall:
                        best_cr_overall = final_cr
                        best_pts_overall = final_pts
                except Exception:
                    pass
        except Exception:
            pass
    
    if best_pts_overall is None:
        return _fibonacci_init(0)
    
    return best_pts_overall
