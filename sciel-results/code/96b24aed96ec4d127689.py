"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_and_worst_face(pts):
        """Compute covering radius and return (radius, worst_face_indices, hull, radii)."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices  # shape (F, 3)
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
            
            max_idx = np.argmax(radii)
            return float(radii[max_idx]), faces[max_idx], hull, radii
        except Exception:
            return np.inf, None, None, None

    def soft_max_covering_radius(pts, beta=20.0):
        """Compute a smooth approximation of the covering radius using soft-max over face circumradii."""
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
            
            # Soft-max for smooth maximum
            r_max = np.max(radii)
            shifted = radii - r_max
            weights = np.exp(beta * shifted)
            weights /= np.sum(weights)
            soft_max_val = r_max + np.log(np.sum(weights * np.exp(beta * shifted))) / beta
            return float(soft_max_val), radii, faces
        except Exception:
            return np.inf, None, None

    def repulsion_force(points, exponent=1.5):
        """Compute repulsion forces between all point pairs."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        return forces

    def gradient_descent(points, alpha_init, decay, max_iter, exponent=1.5):
        """Optimize points via gradient descent on the repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            forces = repulsion_force(points, exponent)
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    def smooth_surrogate_gradient(points, beta=20.0, alpha=0.01, max_iter=100):
        """Optimize using gradient of the soft-max covering radius surrogate."""
        pts = points.copy()
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            soft_val, radii, faces = soft_max_covering_radius(pts, beta)
            if radii is None:
                break
            
            # Compute gradient of soft-max w.r.t. each point
            # d(soft_max)/d(point_k) = sum_f w_f * d(r_f)/d(point_k)
            # Approximate by moving points in direction that reduces large radii
            
            r_max = np.max(radii)
            shifted = radii - r_max
            weights = np.exp(beta * shifted)
            weights /= np.sum(weights)
            
            # For each face, compute which vertices contribute to its radius
            # and move them toward the circumcenter
            forces = np.zeros_like(pts)
            
            for f_idx in range(len(faces)):
                if weights[f_idx] < 1e-10:
                    continue
                
                i, j, k = faces[f_idx]
                a = pts[i]
                b = pts[j]
                c = pts[k]
                
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                c_norm = np.linalg.norm(c_dir)
                
                if c_norm < 1e-10:
                    continue
                
                c_dir /= c_norm
                
                # Move each vertex toward circumcenter (reduces circumradius)
                w_f = weights[f_idx]
                for idx, p in [(i, a), (j, b), (k, c)]:
                    tangent = c_dir - np.dot(c_dir, p) * p
                    t_norm = np.linalg.norm(tangent)
                    if t_norm > 1e-10:
                        forces[idx] += w_f * tangent / t_norm
            
            # Normalize forces
            f_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            f_norms[f_norms < 1e-10] = 1.0
            forces = forces / f_norms
            
            # Update
            pts = pts + alpha * forces
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            # Adaptive step size
            new_soft, _, _ = soft_max_covering_radius(pts, beta)
            if new_soft > soft_val + 1e-8:
                alpha *= 0.9
                # Revert
                pts = pts - alpha * forces / f_norms
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                pts = pts / norms
            else:
                alpha *= 1.01
                alpha = min(alpha, 0.05)
        
        return pts

    def targeted_refinement(points, max_iters=300):
        """Refine by targeting the worst facet: move its 3 vertices to reduce circumradius."""
        pts = points.copy()
        step = 0.005
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            rad, face, hull, all_radii = compute_covering_radius_and_worst_face(pts)
            if face is None:
                break
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm < 1e-10:
                mid = (a + b + c) / 3.0
                mid /= np.linalg.norm(mid)
                for idx, p in [(i, a), (j, b), (k, c)]:
                    direction = p - 0.1 * mid
                    direction /= np.linalg.norm(direction)
                    pts[idx] = direction
                continue
            
            c_dir /= c_norm
            
            for idx, p in [(i, a), (j, b), (k, c)]:
                tangent = c_dir - np.dot(c_dir, p) * p
                t_norm = np.linalg.norm(tangent)
                if t_norm > 1e-10:
                    tangent /= t_norm
                    new_p = p + step * tangent
                    new_p /= np.linalg.norm(new_p)
                    pts[idx] = new_p
            
            new_rad, _, _, _ = compute_covering_radius_and_worst_face(pts)
            if new_rad >= rad:
                pts[i] = a
                pts[j] = b
                pts[k] = c
                step *= 0.5
                if step < 1e-7:
                    break
                for idx, p in [(i, a), (j, b), (k, c)]:
                    tangent = c_dir - np.dot(c_dir, p) * p
                    t_norm = np.linalg.norm(tangent)
                    if t_norm > 1e-10:
                        tangent /= t_norm
                        new_p = p + step * tangent
                        new_p /= np.linalg.norm(new_p)
                        pts[idx] = new_p
                new_rad, _, _, _ = compute_covering_radius_and_worst_face(pts)
                if new_rad >= rad:
                    pts[i] = a
                    pts[j] = b
                    pts[k] = c
                    break
            else:
                step = min(step * 1.1, 0.02)
        
        return pts

    def swap_local_search(points, max_swaps=200):
        """Swap-based local search: perturb pairs of points and keep improvements."""
        pts = points.copy()
        best_rad, _, _, _ = compute_covering_radius_and_worst_face(pts)
        improved = True
        swap_count = 0
        
        while improved and swap_count < max_swaps and time.time() < deadline - 5.0:
            improved = False
            rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
            
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                i = rng.randint(N)
                j = rng.randint(N)
                if i == j:
                    continue
                
                old_i = pts[i].copy()
                old_j = pts[j].copy()
                
                pert = 0.03 * rng.randn(2, 3)
                pts[i] = old_i + pert[0]
                pts[j] = old_j + pert[1]
                pts[i] /= np.linalg.norm(pts[i])
                pts[j] /= np.linalg.norm(pts[j])
                
                rad, _, _, _ = compute_covering_radius_and_worst_face(pts)
                if rad < best_rad - 1e-12:
                    best_rad = rad
                    improved = True
                    swap_count += 1
                else:
                    pts[i] = old_i
                    pts[j] = old_j
        
        return pts, best_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with gradient descent
    n_restarts = 4
    exponents = [1.0, 1.5, 2.0, 3.0]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 50.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300)
                points = best_points + 0.15 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent(points, 0.05, 0.995, 250, exp)
        rad, _, _, _ = compute_covering_radius_and_worst_face(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Targeted refinement
    if best_points is not None and time.time() < deadline - 30.0:
        refined = targeted_refinement(best_points.copy(), max_iters=400)
        rad, _, _, _ = compute_covering_radius_and_worst_face(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 3: Smooth surrogate gradient optimization
    if best_points is not None and time.time() < deadline - 20.0:
        surrogate_pts = smooth_surrogate_gradient(best_points.copy(), beta=20.0, alpha=0.01, max_iter=150)
        rad, _, _, _ = compute_covering_radius_and_worst_face(surrogate_pts)
        if rad < best_rad:
            best_rad = rad
            best_points = surrogate_pts

    # Phase 4: Swap-based local search
    if best_points is not None and time.time() < deadline - 15.0:
        swapped, rad = swap_local_search(best_points.copy(), max_swaps=300)
        if rad < best_rad:
            best_rad = rad
            best_points = swapped

    # Phase 5: Final targeted refinement
    if best_points is not None and time.time() < deadline - 10.0:
        refined = targeted_refinement(best_points.copy(), max_iters=200)
        rad, _, _, _ = compute_covering_radius_and_worst_face(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 6: Final smooth surrogate polish
    if best_points is not None and time.time() < deadline - 5.0:
        polished = smooth_surrogate_gradient(best_points.copy(), beta=30.0, alpha=0.005, max_iter=100)
        rad, _, _, _ = compute_covering_radius_and_worst_face(polished)
        if rad < best_rad:
            best_rad = rad
            best_points = polished

    # Final check
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
