"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def get_hull_and_radius(pts):
        """
        Computes the covering radius exactly via convex hull facets.
        Returns the radius in radians and the index of the worst facet.
        """
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            worst_facet_idx = -1
            
            # Precompute Gram matrix for speed? N is small (50), so direct dot is fine.
            # But we need to solve linear systems for each facet.
            # Facets are triangles.
            
            for idx, simplex in enumerate(hull.simplices):
                p1, p2, p3 = pts[simplex]
                
                # Matrix G for the linear system G @ w = [1, 1, 1]
                # G_ii = 1, G_ij = p_i . p_j
                g12 = np.dot(p1, p2)
                g13 = np.dot(p1, p3)
                g23 = np.dot(p2, p3)
                
                # Avoid numerical issues with determinant
                det = 1.0 + 2.0 * g12 * g13 * g23 - g12**2 - g13**2 - g23**2
                if abs(det) < 1e-10:
                    # Degenerate, skip or approximate
                    continue
                
                # Solve for v = G^{-1} [1, 1, 1]
                # Using Cramer's rule or explicit inverse for 3x3 to avoid overhead
                # G = [[1, g12, g13], [g12, 1, g23], [g13, g23, 1]]
                
                # Inverse of G is (1/det) * adj(G)
                # adj(G) elements:
                # C11 = 1 - g23^2
                # C22 = 1 - g13^2
                # C33 = 1 - g12^2
                # C12 = g23 - g12*g13
                # C13 = g12 - g13*g23
                # C23 = g13 - g12*g23
                
                c11 = 1.0 - g23**2
                c22 = 1.0 - g13**2
                c33 = 1.0 - g12**2
                c12 = g23 - g12 * g13
                c13 = g12 - g13 * g23
                c23 = g13 - g12 * g23
                
                # v = G^{-1} * [1, 1, 1]
                v1 = (c11 * 1.0 + c12 * 1.0 + c13 * 1.0) / det
                v2 = (c12 * 1.0 + c22 * 1.0 + c23 * 1.0) / det
                v3 = (c13 * 1.0 + c23 * 1.0 + c33 * 1.0) / det
                
                # c_vec = v1*p1 + v2*p2 + v3*p3
                c_vec = v1 * p1 + v2 * p2 + v3 * p3
                norm_c = np.linalg.norm(c_vec)
                
                if norm_c < 1e-10:
                    continue
                    
                c_norm = c_vec / norm_c
                cos_R = np.dot(c_norm, p1)
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                
                if R > max_rad:
                    max_rad = R
                    worst_facet_idx = idx
            
            return max_rad, worst_facet_idx, hull
        except Exception:
            # Fallback to grid if hull fails (should be rare)
            tree = cKDTree(pts)
            # Sparse grid
            lat = np.linspace(-np.pi/2, np.pi/2, 40)
            lon = np.linspace(0, 2*np.pi, 80)
            lat, lon = np.meshgrid(lat, lon)
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dists, _ = tree.query(grid_points)
            angles = 2.0 * np.arcsin(np.clip(dists / 2.0, 0.0, 1.0))
            return np.max(angles), -1, None

    def objective(pts):
        rad, _, _ = get_hull_and_radius(pts)
        return rad

    # Initializations
    # 1. Fibonacci Spiral
    def init_fibonacci():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # 2. Random Uniform
    def init_random():
        pts = np.random.randn(N, 3)
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    # 3. Tammes-like heuristic (start with low density and add?)
    # Let's stick to Fib and Random for multi-start.

    initializations = [
        init_fibonacci(),
        init_random(),
        init_random()
    ]
    
    best_points = None
    best_score = np.inf
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
            
        current_points = init_pts.copy()
        current_score, worst_idx, hull = get_hull_and_radius(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Facet-Driven Optimization
        # We want to minimize the max circumradius.
        # Strategy: Identify the worst facet. Push its vertices apart.
        # To do this, we move vertices in the direction that increases the distance
        # from the circumcenter.
        
        step_size = 0.1
        patience = 0
        max_patience = 200
        
        while time.time() < deadline - 5.0:
            score, worst_idx, hull = get_hull_and_radius(current_points)
            
            if score < best_score:
                best_score = score
                best_points = current_points.copy()
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
                    
            if worst_idx == -1 or hull is None:
                # Fallback to random perturbation if hull fails
                idx = np.random.randint(N)
                rand_dir = np.random.randn(3)
                rand_dir /= np.linalg.norm(rand_dir)
                new_points = current_points.copy()
                new_points[idx] = new_points[idx] + step_size * rand_dir
                new_points[idx] /= np.linalg.norm(new_points[idx])
                current_points = new_points
                continue
            
            simplex = hull.simplices[worst_idx]
            p1_idx, p2_idx, p3_idx = simplex
            p1, p2, p3 = current_points[p1_idx], current_points[p2_idx], current_points[p3_idx]
            
            # Calculate circumcenter
            g12 = np.dot(p1, p2)
            g13 = np.dot(p1, p3)
            g23 = np.dot(p2, p3)
            det = 1.0 + 2.0 * g12 * g13 * g23 - g12**2 - g13**2 - g23**2
            if abs(det) < 1e-10:
                # Degenerate, random step
                idx = np.random.randint(N)
                rand_dir = np.random.randn(3)
                rand_dir /= np.linalg.norm(rand_dir)
                new_points = current_points.copy()
                new_points[idx] = new_points[idx] + step_size * rand_dir
                new_points[idx] /= np.linalg.norm(new_points[idx])
                current_points = new_points
                continue

            c11 = 1.0 - g23**2
            c22 = 1.0 - g13**2
            c33 = 1.0 - g12**2
            c12 = g23 - g12 * g13
            c13 = g12 - g13 * g23
            c23 = g13 - g12 * g23
            
            v1 = (c11 + c12 + c13) / det
            v2 = (c12 + c22 + c23) / det
            v3 = (c13 + c23 + c33) / det
            
            c_vec = v1 * p1 + v2 * p2 + v3 * p3
            norm_c = np.linalg.norm(c_vec)
            if norm_c < 1e-10:
                continue
            c_norm = c_vec / norm_c
            
            # We want to move p1, p2, p3 away from c_norm.
            # The gradient of distance to c_norm on the sphere for a point p is:
            # grad = (c_norm - (p . c_norm) * p)  ... wait.
            # f(p) = arccos(p . c). df/dp = - (c - (p.c)p) / sqrt(1-(p.c)^2)
            # To increase distance (decrease p.c), we should move in direction:
            # d = c - (p.c)p  (This is the projection of c onto tangent space at p)
            # Moving p in direction d increases p.c?
            # Let p(t) = normalize(p + t*d).
            # d(p.c)/dt at t=0 is d . c - (p.c)(d.p) = (c - (p.c)p).c = 1 - (p.c)^2 > 0.
            # So moving in direction d INCREASES the dot product, which DECREASES the angle.
            # We want to INCREASE the angle (move away from center).
            # So we should move in direction -d = (p.c)p - c.
            
            new_points = current_points.copy()
            
            for idx, p in zip([p1_idx, p2_idx, p3_idx], [p1, p2, p3]):
                dot_pc = np.dot(p, c_norm)
                # Direction to move away from c_norm
                dir_away = dot_pc * p - c_norm
                norm_dir = np.linalg.norm(dir_away)
                if norm_dir > 1e-10:
                    dir_away /= norm_dir
                else:
                    dir_away = np.random.randn(3)
                    dir_away /= np.linalg.norm(dir_away)
                
                new_p = p + step_size * dir_away
                new_p /= np.linalg.norm(new_p)
                new_points[idx] = new_p
                
            # Check if this move improved the global score
            new_score, _, _ = get_hull_and_radius(new_points)
            
            if new_score < current_score:
                current_points = new_points
                current_score = new_score
                # Slightly increase step size if improving? Or keep constant.
                # Let's keep constant for stability, maybe decay slightly.
                step_size = max(0.01, step_size * 0.999)
            else:
                # If it didn't improve, try a smaller step or random perturbation
                # to escape local minima of the "worst facet" logic.
                # Sometimes moving the worst facet's vertices makes another facet worse.
                # We accept small increases? No, strict improvement for now to ensure convergence.
                # But if stuck, we need to break out.
                if patience > 10:
                    # Random kick
                    idx = np.random.randint(N)
                    rand_dir = np.random.randn(3)
                    rand_dir /= np.linalg.norm(rand_dir)
                    new_points = current_points.copy()
                    new_points[idx] = new_points[idx] + step_size * rand_dir
                    new_points[idx] /= np.linalg.norm(new_points[idx])
                    current_points = new_points
                    step_size = max(0.01, step_size * 0.9)
                else:
                    # Try smaller step
                    step_size *= 0.95

    # Final polish with standard local search if time remains
    if best_points is not None and time.time() < deadline - 2.0:
        polish_points = best_points.copy()
        polish_score = best_score
        step = 0.02
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                for _ in range(3):
                    rand_dir = np.random.randn(3)
                    rand_dir /= np.linalg.norm(rand_dir)
                    old_p = polish_points[i]
                    new_p = polish_points[i] + step * rand_dir
                    new_p /= np.linalg.norm(new_p)
                    polish_points[i] = new_p
                    new_score, _, _ = get_hull_and_radius(polish_points)
                    if new_score < polish_score:
                        polish_score = new_score
                        improved = True
                    else:
                        polish_points[i] = old_p
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
        if polish_score < best_score:
            best_points = polish_points
            best_score = polish_score

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
