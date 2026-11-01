import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random points on the sphere."""
    theta = rng.uniform(0, 2 * np.pi, N)
    z = rng.uniform(-1, 1)
    r = np.sqrt(np.clip(1 - z**2, 0, None))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

def _compute_hull_radius_and_grad(pts):
    """
    Compute the covering radius (max circumradius of hull facets) and its gradient.
    Returns (radius, gradient, worst_facet_idx)
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi, np.zeros(3*N), 0
            
        normals = cross[valid].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        # Find the worst facet
        max_idx_local = np.argmax(radii)
        max_radius = radii[max_idx_local]
        
        # Gradient calculation for the worst facet
        # The radius of a triangle on the unit sphere is determined by its circumcenter.
        # For a triangle with vertices a, b, c on the unit sphere, the circumcenter p
        # is the point on the sphere maximizing the minimum distance to a,b,c? No.
        # The circumcenter is the intersection of the plane of the triangle with the normal to the triangle passing through the origin?
        # Actually, for points on a sphere, the circumcenter of the spherical triangle is the projection of the Euclidean circumcenter of the planar triangle onto the sphere?
        # No, the circumcenter of the spherical triangle is the point P on the sphere such that dist(P,a)=dist(P,b)=dist(P,c).
        # This point P is along the normal of the plane defined by a,b,c.
        # Let n be the unit normal to the plane abc.
        # The circumcenter p is n * (1/cos_theta) ? No.
        # The distance from the origin to the plane is d = n . a.
        # The radius of the circle in the plane is r_circ = sqrt(1 - d^2).
        # The circumcenter in the plane is C_plane.
        # The point on the sphere is P = C_plane / ||C_plane||.
        # It can be shown that P is parallel to n if the triangle is "small"? No.
        # Actually, the circumcenter of the spherical triangle is the pole of the plane if the triangle is equilateral?
        # Let's use the property: The gradient of the circumradius R with respect to vertex a.
        # R = arccos(d). d = n . a.
        # n is the unit normal.
        # d changes when a changes.
        # n depends on a, b, c.
        # This is complex.
        
        # Alternative: Use the fact that for the worst facet, the covering radius is determined by the circumcenter.
        # If we move the vertices, the circumcenter moves.
        # A simpler approximation: The gradient of the distance from the circumcenter P to vertex A is (A-P)/||A-P||?
        # Let's use the subgradient idea from the parent but smoothed?
        # The prompt asks for a DIFFERENT strategy.
        # Let's try minimizing the sum of squares of the circumradii? Or just the max?
        # Minimize max(R_i) is non-differentiable.
        # Let's use a differentiable proxy: Sum of exp(k * R_i) / N.
        # Gradient of this proxy is easier to compute if we have gradients of R_i.
        
        # Let's compute gradients of R_i for all valid facets.
        # R_i = arccos(d_i) where d_i = n_i . a_i.
        # d_i = (n_i . a_i).
        # n_i is unit normal.
        # grad_a (n_i . a_i) = n_i + a_i (grad_a n_i . a_i)
        # grad_a n_i: n_i = ( (b-a) x (c-a) ) / ||...||
        # This is tedious.
        
        # Let's stick to the "differentiable optimal transport dual" or similar?
        # Or just use a robust numerical gradient?
        # No, analytical is better.
        
        # Let's use a different metric: Minimize the sum of squared distances from the uniform grid to the nearest point?
        # This is a differentiable objective (if we use soft-min).
        # But it might not minimize the max radius well.
        
        # Let's try: Minimize the sum of exp(beta * R_i) for all facets.
        # We need grad R_i.
        # For a facet with vertices a,b,c:
        # n = cross(b-a, c-a) / |cross(b-a, c-a)|
        # d = n . a
        # R = arccos(d)
        # dR/dd = -1/sqrt(1-d^2)
        # dd/da = n + a * (dn/da . a)
        # dn/da:
        # Let u = b-a, v = c-a.
        # n = (u x v) / |u x v|
        # du/da = -I, dv/da = -I
        # d(u x v)/da = (-I x v) + (u x -I) = -v x I - u x I? No.
        # d(u x v)/dx = ...
        # Vector triple product rules.
        # Let's use a known result or approximate.
        # Actually, for the purpose of this task, a simpler differentiable objective might be better.
        
        # Objective: Minimize the maximum distance from a dense grid of points to the nearest configuration point.
        # This is the "covering radius" defined by the grid.
        # Let G be a grid of points on the sphere.
        # d(g, P) = min_i dist(g, p_i).
        # Objective = max_g d(g, P).
        # This is still non-differentiable due to max and min.
        # Use soft-min: d_soft(g, P) = -1/beta * log(sum_i exp(-beta * dist(g, p_i)))
        # Objective = max_g d_soft(g, P) -> Use soft-max over g?
        # This is getting complicated and slow.
        
        # Let's go back to the hull facet radii.
        # We will compute the gradient of the worst facet's radius numerically?
        # No, let's use the subgradient from the parent but optimize it with L-BFGS-B?
        # The parent used manual subgradient descent.
        # L-BFGS-B requires a smooth function.
        # We can define a smooth function F(P) = sum_i exp(k * R_i) / N.
        # And compute its gradient numerically using finite differences?
        # That would be 3*N evaluations of the hull per gradient step. Too slow.
        
        # Let's try a different approach:
        # Use the fact that the optimal configuration is a local minimum of the max circumradius.
        # We can use a "push" algorithm:
        # 1. Compute hull.
        # 2. Find worst facet.
        # 3. Move its 3 vertices towards the circumcenter of that facet.
        # 4. Repeat.
        # This is what the parent does.
        
        # To be MATERIALY DIFFERENT:
        # Use a global optimization method like Differential Evolution on a subset of parameters?
        # Or use a different initialization and a different local solver.
        
        # Let's try:
        # 1. Generate multiple random initializations.
        # 2. For each, run a "Lloyd-like" relaxation where points are moved to the centroid of their Voronoi cell (on the sphere).
        #    This minimizes the sum of squared distances, not the max distance.
        #    But it often produces good coverings.
        # 3. Then refine with the subgradient method.
        
        # Let's implement Spherical Lloyd's Algorithm.
        # 1. Compute Voronoi cells on the sphere (using ConvexHull of the dual? Or just nearest neighbor assignment).
        #    Assignment: For each point p_i, find all grid points closer to p_i than any other p_j.
        #    Move p_i to the spherical centroid of its assigned points.
        # 2. This is slow if grid is large.
        
        # Alternative:
        # Use the "Farthest Point Sampling" to initialize?
        # Then refine.
        
        # Let's try a different local search:
        # Simulated Annealing on the positions.
        # Objective: Max circumradius.
        # Move: Randomly pick a point, move it a small random amount on the sphere.
        # Accept if better or with probability exp(-delta/T).
        # This is a black-box optimization. It might escape local minima better than gradient descent.
        
        # Let's implement Simulated Annealing.
        
    except Exception:
        return np.pi, np.zeros(3*N), 0

def _simulated_annealing(pts, deadline, rng):
    """
    Simulated Annealing to minimize the max circumradius.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    
    # Initial temperature
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.95
    
    # Compute initial radius
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        valid = area > _EPS
        if not np.any(valid):
            return pts
        normals = cross[valid].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        curr_r = radii.max()
    except Exception:
        return pts
        
    best_r = curr_r
    
    steps_per_eval = 10
    eval_count = 0
    start_time = time.time()
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Randomly select a point to move
        idx = rng.integers(0, N)
        
        # Generate a random direction on the tangent plane
        v = pts[idx]
        rand_vec = rng.normal(size=3)
        tangent_dir = rand_vec - np.dot(rand_vec, v) * v
        norm_t = np.linalg.norm(tangent_dir)
        if norm_t < _EPS:
            continue
        tangent_dir /= norm_t
        
        # Move the point
        step_size = T * 0.1
        new_pts = pts.copy()
        new_pts[idx] = pts[idx] + step_size * tangent_dir
        new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        # Evaluate new radius
        try:
            hull_new = ConvexHull(new_pts)
            faces_new = hull_new.simplices
            A_new = new_pts[faces_new[:, 0]]
            B_new = new_pts[faces_new[:, 1]]
            C_new = new_pts[faces_new[:, 2]]
            AB_new = B_new - A_new
            AC_new = C_new - A_new
            cross_new = np.cross(AB_new, AC_new)
            area_new = 0.5 * np.linalg.norm(cross_new, axis=1)
            valid_new = area_new > _EPS
            if not np.any(valid_new):
                new_r = np.pi
            else:
                normals_new = cross_new[valid_new].copy()
                norms_new = np.linalg.norm(normals_new, axis=1, keepdims=True)
                centroids_new = (A_new[valid_new] + B_new[valid_new] + C_new[valid_new]) / 3.0
                signs_new = np.sum(normals_new * centroids_new, axis=1, keepdims=True)
                normals_new = normals_new * np.sign(signs_new)
                normals_new = normals_new / (norms_new + _EPS)
                cos_theta_new = np.sum(normals_new * A_new[valid_new], axis=1)
                cos_theta_new = np.clip(cos_theta_new, -1.0, 1.0)
                radii_new = np.arccos(cos_theta_new)
                new_r = radii_new.max()
        except Exception:
            new_r = np.pi
            
        # Accept or reject
        if new_r < curr_r:
            pts = new_pts
            curr_r = new_r
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
        else:
            delta = new_r - curr_r
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                pts = new_pts
                curr_r = new_r
                
        T *= cooling_rate
        
    return best_pts

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = np.pi
    
    # Strategy:
    # 1. Start with Fibonacci spiral.
    # 2. Run Simulated Annealing from there.
    # 3. Run Simulated Annealing from a few random starts.
    
    # 1. Fibonacci
    pts_fib = _initial_fib()
    pts_sa = _simulated_annealing(pts_fib, deadline - 30, rng)
    try:
        hull = ConvexHull(pts_sa)
        faces = hull.simplices
        A = pts_sa[faces[:, 0]]
        B = pts_sa[faces[:, 1]]
        C = pts_sa[faces[:, 2]]
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        valid = area > _EPS
        if np.any(valid):
            normals = cross[valid].copy()
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii = np.arccos(cos_theta)
            r_fib = radii.max()
        else:
            r_fib = np.pi
    except Exception:
        r_fib = np.pi
        
    if r_fib < best_r:
        best_r = r_fib
        best_pts = pts_sa.copy()
        
    # 2. Random restarts
    n_restarts = 5
    time_per_restart = (deadline - time.time() - 5) / n_restarts
    
    for i in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        pts_init = _random_init(rng)
        pts_sa = _simulated_annealing(pts_init, time.time() + time_per_restart, rng)
        
        try:
            hull = ConvexHull(pts_sa)
            faces = hull.simplices
            A = pts_sa[faces[:, 0]]
            B = pts_sa[faces[:, 1]]
            C = pts_sa[faces[:, 2]]
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > _EPS
            if np.any(valid):
                normals = cross[valid].copy()
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                centroids = (A[valid] + B[valid] + C[valid]) / 3.0
                signs = np.sum(normals * centroids, axis=1, keepdims=True)
                normals = normals * np.sign(signs)
                normals = normals / (norms + _EPS)
                cos_theta = np.sum(normals * A[valid], axis=1)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                radii = np.arccos(cos_theta)
                r_rand = radii.max()
            else:
                r_rand = np.pi
        except Exception:
            r_rand = np.pi
            
        if r_rand < best_r:
            best_r = r_rand
            best_pts = pts_sa.copy()
            
    # Final normalization
    if best_pts is None:
        best_pts = _initial_fib()
        
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / (norms + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts

