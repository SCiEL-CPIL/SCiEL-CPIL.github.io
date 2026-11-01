"""Program for covering the sphere with 130 equal caps using repulsion and facet-local search."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(2000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            return np.pi

    def repulsion_energy_grad(pts, exponent=2.0):
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        
        w_grad = exponent * (dist_sq ** (-exponent/2 - 1))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ pts
        
        return float(np.sum(w)), grad

    def gradient_descent_repulsion(pts, exponent, alpha_init, decay, max_iter):
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = repulsion_energy_grad(pts, exponent)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    def facet_local_search(pts, max_iter=100):
        """
        Structural change: Directly optimize the covering radius by targeting the worst facets.
        For the facet with the largest circumradius, move its vertices to reduce the radius.
        """
        best_rad = compute_covering_radius_exact(pts)
        current_pts = pts.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            try:
                hull = ConvexHull(current_pts)
                faces = hull.simplices
                a = current_pts[faces[:, 0]]
                b = current_pts[faces[:, 1]]
                c = current_pts[faces[:, 2]]
                
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norms = np.linalg.norm(c_dir, axis=1)
                
                radii = np.zeros(len(faces))
                valid = norms > 1e-10
                
                if np.any(valid):
                    c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                    cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                    r = np.arccos(cos_r)
                    r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                    radii[valid] = r
                
                if np.any(~valid):
                    a_deg = a[~valid]
                    b_deg = b[~valid]
                    c_deg = c[~valid]
                    d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                    d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                    d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                    radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                
                max_rad = np.max(radii)
                if max_rad <= best_rad:
                    break
                    
                best_rad = max_rad
                worst_face_idx = np.argmax(radii)
                face = faces[worst_face_idx]
                i, j, k = face
                
                # Move vertices i, j, k to reduce the circumradius of this facet.
                # Heuristic: Move them towards the centroid of the facet's "cap" or adjust to be more equilateral.
                # A simple gradient step: The gradient of the circumradius w.r.t vertex position on the sphere.
                # For a spherical triangle, the circumradius R is minimized when the triangle is equilateral.
                # We can try small perturbations of i, j, k in the direction that reduces the max side length or increases the minimum side length.
                
                # Let's try moving each vertex slightly in the direction of the other two vertices (to make it more compact/equilateral)
                # Or move away from the center of the sphere? No, they are on the sphere.
                
                # Better: Use the gradient of the repulsive energy restricted to these points?
                # Or just random local search on these 3 points.
                
                improved = False
                rng = np.random.RandomState(int(time.time() * 1000) + worst_face_idx)
                
                for _ in range(10):
                    # Perturb i, j, k
                    perturbations = rng.randn(3, 3) * 0.01
                    perturbations[i] = perturbations[i] - np.dot(perturbations[i], current_pts[i]) * current_pts[i]
                    perturbations[j] = perturbations[j] - np.dot(perturbations[j], current_pts[j]) * current_pts[j]
                    perturbations[k] = perturbations[k] - np.dot(perturbations[k], current_pts[k]) * current_pts[k]
                    
                    new_pts = current_pts.copy()
                    new_pts[i] = current_pts[i] + perturbations[i]
                    new_pts[j] = current_pts[j] + perturbations[j]
                    new_pts[k] = current_pts[k] + perturbations[k]
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    new_pts[j] /= np.linalg.norm(new_pts[j])
                    new_pts[k] /= np.linalg.norm(new_pts[k])
                    
                    new_rad = compute_covering_radius_exact(new_pts)
                    if new_rad < best_rad - 1e-10:
                        best_rad = new_rad
                        current_pts = new_pts
                        improved = True
                        break
                
                if not improved:
                    # If no improvement found after several tries, try a different strategy:
                    # Move the vertices in the direction of the global repulsion gradient for these points
                    _, grad = repulsion_energy_grad(current_pts, 2.0)
                    for idx in [i, j, k]:
                        g = grad[idx]
                        g = g - np.dot(g, current_pts[idx]) * current_pts[idx]
                        g_norm = np.linalg.norm(g)
                        if g_norm > 1e-10:
                            g /= g_norm
                        new_p = current_pts[idx] + 0.01 * g
                        new_p /= np.linalg.norm(new_p)
                        current_pts[idx] = new_p
                    
                    new_rad = compute_covering_radius_exact(current_pts)
                    if new_rad < best_rad - 1e-10:
                        best_rad = new_rad
                        improved = True
                    else:
                        # Revert if it got worse
                        # We need to restore current_pts to the state before this iteration
                        # This is tricky without a stack. Let's just break if we can't improve.
                        break

            except Exception:
                break
                
        return current_pts, best_rad

    best_points = None
    best_rad = np.inf

    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 50.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Phase 1: Repulsion
        points = gradient_descent_repulsion(points, 8.0, 0.1, 0.99, 100)
        points = gradient_descent_repulsion(points, 2.0, 0.05, 0.99, 100)
        
        # Phase 2: Facet Local Search
        points, rad = facet_local_search(points, max_iter=50)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 3: Further refinement
    if best_points is not None and time.time() < deadline - 10.0:
        # Run facet local search again on the best points
        best_points, best_rad = facet_local_search(best_points, max_iter=100)

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
