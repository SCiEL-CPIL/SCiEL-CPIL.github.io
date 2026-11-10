import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610  # 10s margin

    # 1. Define the Icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # The 12 points are fixed at the vertices of a regular icosahedron.
    # We need to place 2 additional points.
    # Due to symmetry, the optimal configuration for adding 2 points to an icosahedron
    # often involves placing them on an axis of symmetry (e.g., the z-axis) 
    # or in symmetric pairs.
    # Let's hypothesize the 2 points are at (0, 0, z1) and (0, 0, z2) or (0,0,0) and (0,0,z).
    # Actually, for 14 points, a common high-symmetry candidate is the icosahedron 
    # plus 2 points on the axis of a 5-fold symmetry, or perhaps the center and one axis point?
    # Let's parameterize the 2 extra points generally but start with symmetric guesses.
    
    # Strategy: 
    # 1. Fix 12 points to icosahedron vertices (scaled by R).
    # 2. Optimize the positions of the 2 remaining points (x1, y1, z1, x2, y2, z2) 
    #    and the scale R.
    # 3. Use Nelder-Mead on the 7 parameters (R, p1, p2).
    # 4. To handle rotation invariance, we can fix the orientation of the icosahedron 
    #    and allow the 2 points to move freely. Or better, fix 2 points to the z-axis 
    #    by symmetry? 
    #    If we assume the global optimum has the 2 points on the z-axis (a 5-fold axis),
    #    we reduce parameters to 3: R, z1, z2.
    #    Let's test this hypothesis first. If it yields a high score, we can try relaxing it.
    
    def compute_score_from_params(params):
        # params: [R, z1, z2]
        # z1, z2 are coordinates along the z-axis for the 2 extra points
        # The icosahedron is centered at origin, scaled by R.
        R = params[0]
        z1 = params[1]
        z2 = params[2]
        
        if R < 1e-6:
            return 0.0
            
        # Construct points
        ico_scaled = ico_verts * R
        p1 = np.array([0.0, 0.0, z1])
        p2 = np.array([0.0, 0.0, z2])
        
        pts = np.vstack([ico_scaled, p1, p2])
        
        # Calculate score
        dists = pdist(pts, metric='sqeuclidean')
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        score = d_min_sq / d_max_sq
        return score

    def negative_score(params):
        return -compute_score_from_params(params)

    best_score = 0.0
    best_points = None
    
    # Initial guesses for [R, z1, z2]
    # R is the radius of the icosahedron.
    # The distance between adjacent icosahedron vertices on unit sphere is sqrt(2 - 2/phi) ~ 1.051
    # If R=1, d_min_ico ~ 1.051.
    # The 2 points on the axis should be placed to maximize the minimum distance to all other points.
    # Symmetric placement: z1 = -z, z2 = z? Or z1=0, z2=z?
    # Let's try a few symmetric configurations.
    
    initial_guesses = [
        [1.0, 0.0, 0.0],       # Both at center? (Coincident -> score 0, bad start)
        [1.0, 0.5, -0.5],      # Symmetric pair
        [1.0, 1.0, -1.0],      # On the sphere
        [1.0, 0.0, 1.0],       # Center and top
        [1.2, 0.0, 0.0],       # Larger R
        [1.0, 0.3, -0.3],
        [1.0, 0.7, -0.7],
    ]
    
    # Run optimization for each guess
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            # Bounds for R: [0.1, 10]
            # Bounds for z: [-10, 10]
            res = minimize(
                negative_score, 
                guess, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    R, z1, z2 = res.x
                    ico_scaled = ico_verts * R
                    p1 = np.array([0.0, 0.0, z1])
                    p2 = np.array([0.0, 0.0, z2])
                    best_points = np.vstack([ico_scaled, p1, p2])
        except Exception:
            continue

    # If the symmetric assumption (z-axis) failed or yielded low score, 
    # try a more general search with 2 free points (6 params + R = 7 params).
    # This is more expensive but covers more space.
    if best_score < 0.3 and time.time() < deadline - 60:
        
        def compute_score_general(params):
            # params: [R, x1, y1, z1, x2, y2, z2]
            R = params[0]
            p1 = params[1:4]
            p2 = params[4:7]
            
            if R < 1e-6:
                return 0.0
                
            ico_scaled = ico_verts * R
            pts = np.vstack([ico_scaled, p1, p2])
            
            dists = pdist(pts, metric='sqeuclidean')
            d_min_sq = np.min(dists)
            d_max_sq = np.max(dists)
            
            if d_max_sq < 1e-12:
                return 0.0
            return d_min_sq / d_max_sq

        def negative_score_general(params):
            return -compute_score_general(params)

        # Start from the best symmetric solution if found, else random
        if best_points is not None:
            # Extract p1, p2 from best_points (last 2 rows)
            p1_best = best_points[12]
            p2_best = best_points[13]
            # Estimate R from ico verts
            R_best = np.linalg.norm(best_points[0])
            guess_gen = np.array([R_best, p1_best[0], p1_best[1], p1_best[2], p2_best[0], p2_best[1], p2_best[2]])
        else:
            guess_gen = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5])
            
        try:
            res = minimize(
                negative_score_general, 
                guess_gen, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    R = res.x[0]
                    p1 = res.x[1:4]
                    p2 = res.x[4:7]
                    ico_scaled = ico_verts * R
                    best_points = np.vstack([ico_scaled, p1, p2])
        except Exception:
            pass

    # Fallback if optimization failed completely
    if best_points is None:
        best_points = np.vstack([ico_verts, np.array([0,0,0]), np.array([0,0,1])])

    # Final validation and return
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = np.vstack([ico_verts, np.array([0,0,0]), np.array([0,0,1])])
        
    return best_points
