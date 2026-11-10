import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Start with random points on the unit sphere (or other symmetric seeds).
    2. Use Gradient Descent on a smooth surrogate objective (sum of exponentials of negative squared distances)
       to push points apart globally. This avoids the local minima traps of raw NM on the non-smooth min/max ratio.
    3. Polish the result using Nelder-Mead on the true objective function.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610  # 10s margin

    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Compute pairwise squared distances
        dists_sq = pdist(points, metric='sqeuclidean')
        
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        ratio_sq = d_min_sq / d_max_sq
        # The metric is (d_min/d_max)^2, which is ratio_sq
        # Wait, the task says: Metric: (d_min / d_max)^2
        # d_min is sqrt(min_sq), d_max is sqrt(max_sq)
        # (sqrt(min_sq)/sqrt(max_sq))^2 = min_sq / max_sq
        return ratio_sq

    def smooth_objective(points_flat):
        """
        Smooth surrogate objective for gradient descent.
        We want to maximize the minimum distance.
        A common smooth approximation is -sum(exp(-beta * d_ij^2)) or similar.
        Alternatively, we can use the sum of inverse squared distances, which is smooth and differentiable.
        Minimizing sum(1/d_ij^2) pushes points apart.
        """
        points = points_flat.reshape(n, d)
        dists_sq = pdist(points, metric='sqeuclidean')
        
        # Avoid division by zero
        dists_sq = np.maximum(dists_sq, 1e-6)
        
        # We want to MINIMIZE this function (so we maximize distance)
        # Using sum(1/d^2) is a standard "soft" repulsion potential
        return np.sum(1.0 / dists_sq)

    def negative_true_score(points_flat):
        points = points_flat.reshape(n, d)
        return -true_score(points)

    best_score = 0.0
    best_points = None

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Random points on sphere
    for _ in range(5):
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initial_configs.append(pts)

    # 2. Icosahedron + 2 random points (as a sanity check / fallback)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add 2 random points on sphere
    p1 = np.random.randn(1, 3)
    p1 /= np.linalg.norm(p1)
    p2 = np.random.randn(1, 3)
    p2 /= np.linalg.norm(p2)
    initial_configs.append(np.vstack([ico_verts, p1, p2]))
    
    # 3. Simplex-like distribution (if n was small, but 14 is too many for regular simplex in 3D, 
    #    so we just use random clusters)
    for _ in range(5):
        pts = np.random.randn(n, d) * 0.1
        initial_configs.append(pts)

    # Optimization Loop
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
            
        try:
            # Phase 1: Smooth Gradient Descent
            # We minimize the smooth repulsion potential.
            # Using scipy.optimize.minimize with BFGS or L-BFGS-B on the flattened array.
            # Note: Points are not constrained to a sphere in the final metric, 
            # but starting on a sphere and letting them move in 3D is fine.
            
            # To handle the scale invariance (metric is scale invariant), 
            # we can fix the scale during optimization by normalizing the points 
            # to have a fixed centroid and radius, or just let the optimizer handle it.
            # Since the metric is scale invariant, the optimizer might drift in scale.
            # We will normalize after each step or just rely on the relative distances.
            # Actually, sum(1/d^2) is scale dependent. If points expand, score goes to 0.
            # So we must constrain the scale. 
            # Let's fix the RMS radius to 1.0 during the smooth phase.
            
            def constrained_smooth_obj(x_flat):
                pts = x_flat.reshape(n, d)
                # Normalize to unit RMS radius to prevent scale drift
                rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
                if rms < 1e-6:
                    return 1e10
                pts = pts / rms
                return smooth_objective(pts.flatten())

            # Use BFGS for smooth optimization
            res_smooth = minimize(
                constrained_smooth_obj, 
                init_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            current_pts = res_smooth.x.reshape(n, d)
            
            # Phase 2: Nelder-Mead Polish on True Objective
            # The true objective is non-smooth (min/max). NM is good for this.
            # We start from the smoothed result.
            
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                # Optional: normalize again to help NM? 
                # NM works on the raw coords. Since metric is scale invariant,
                # NM might wander in scale. 
                # Let's fix one point to origin and one to (1,0,0) to break symmetry?
                # No, that complicates the parameter space.
                # Instead, we just let it run. The scale invariance creates a flat valley,
                # which NM can handle, though it might be slow.
                # To speed up, we can normalize inside the objective? 
                # If we normalize inside the objective, the gradient (quasi) will be affected.
                # Better: Fix the scale by normalizing the points to have unit RMS radius 
                # inside the objective function for NM as well? 
                # If we do that, we are effectively optimizing on the manifold.
                # Let's try normalizing inside the NM objective to prevent scale drift.
                
                rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
                if rms < 1e-6:
                    return 1e10
                pts_norm = pts / rms
                return -true_score(pts_norm)

            res_nm = minimize(
                nm_obj, 
                current_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            final_pts = res_nm.x.reshape(n, d)
            
            # Calculate score
            score = -res_nm.fun
            if score > best_score:
                best_score = score
                best_points = final_pts.copy()
                
        except Exception as e:
            continue

    # Fallback if all failed
    if best_points is None:
        # Return icosahedron + 2 points as fallback
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p1 = np.array([0, 0, 1])
        p2 = np.array([0, 0, -1])
        best_points = np.vstack([ico_verts, p1, p2])

    # Final validation
    if not np.all(np.isfinite(best_points)):
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        best_points = np.vstack([ico_verts, np.array([0,0,0]), np.array([0,0,1])])

    return best_points
