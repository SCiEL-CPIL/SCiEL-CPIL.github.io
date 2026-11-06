"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """
        Compute the covering radius of the configuration on the unit sphere.
        The covering radius is the maximum circumradius of the spherical triangles
        (facets of the convex hull) formed by the points.
        """
        try:
            pts = normalize(pts)
            
            if np.any(np.linalg.norm(pts, axis=1) < 1e-10):
                return 180.0
                
            # Check for coincident points
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 1e-8:
                        return 180.0

            # Check if all points are in a hemisphere (degenerate for covering)
            # A simple check: if there exists a vector v such that v.p_i <= 0 for all i
            # This is equivalent to the origin being in the convex hull.
            # If origin is NOT in convex hull, it's in a hemisphere.
            # We can check this by seeing if ConvexHull fails or by checking the distance from origin to hull.
            # However, ConvexHull in 3D requires points not to be coplanar.
            # A robust check: try to find a point on the sphere that is farthest from all pts.
            # If the max distance is > 90 degrees, it's likely in a hemisphere.
            # But the problem states "configurations inside a closed hemisphere" are invalid.
            # We should avoid generating such configs.
            
            hull = ConvexHull(pts)
            
            max_r = 0.0
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute the circumcenter of the spherical triangle (p0, p1, p2).
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                
                try:
                    w = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    continue
                    
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    continue
                    
                c /= c_norm
                
                # Compute angular distance from c to p0
                cos_d = np.clip(c @ p0, -1.0, 1.0)
                d = np.arccos(cos_d)
                
                if d > max_r:
                    max_r = d
                    
            return np.degrees(max_r)
            
        except Exception:
            return 180.0

    def lloyd_polish(pts, max_iters=30, step_size=0.1):
        """
        Lloyd's algorithm for sphere covering.
        Moves each point towards the centroid of the circumcenters of its adjacent facets.
        """
        pts = normalize(pts)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            try:
                hull = ConvexHull(pts)
            except:
                break
                
            sum_centers = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                try:
                    w = np.linalg.solve(G, rhs)
                    c = np.sum(w[:, None] * p_face, axis=0)
                    c_norm = np.linalg.norm(c)
                    if c_norm > 1e-10:
                        c /= c_norm
                        for v_idx in face:
                            sum_centers[v_idx] += c
                            counts[v_idx] += 1
                except:
                    pass
            
            # Update points
            moved = False
            for i in range(N):
                if counts[i] > 0:
                    target = sum_centers[i] / counts[i]
                    target_norm = np.linalg.norm(target)
                    if target_norm > 1e-10:
                        target /= target_norm
                        new_pt = pts[i] + step_size * (target - pts[i])
                        new_pt_norm = np.linalg.norm(new_pt)
                        if new_pt_norm > 1e-10:
                            pts[i] = new_pt / new_pt_norm
                            moved = True
            
            if not moved:
                break
                
        return pts

    def micro_polish(pts, max_iters=50, base_step=0.05):
        """
        Iterative micro-polish with per-facet adaptive step size.
        Focuses on reducing the largest circumradius by moving vertices
        of the worst facets towards their circumcenters.
        """
        pts = normalize(pts)
        current_cr = compute_covering_radius(pts)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            try:
                hull = ConvexHull(pts)
            except:
                break
                
            # Compute circumcenters and radii for all facets
            facets = []
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                try:
                    w = np.linalg.solve(G, rhs)
                    c = np.sum(w[:, None] * p_face, axis=0)
                    c_norm = np.linalg.norm(c)
                    if c_norm > 1e-10:
                        c /= c_norm
                        cos_d = np.clip(c @ p0, -1.0, 1.0)
                        d = np.arccos(cos_d)
                        facets.append((d, face, c))
                except:
                    pass
            
            if not facets:
                break
                
            # Sort by radius descending
            facets.sort(key=lambda x: -x[0])
            
            # Only process the top few worst facets
            n_worst = min(5, len(facets))
            improved = False
            
            for d, face, c in facets[:n_worst]:
                if time.time() > deadline - 2:
                    break
                    
                # Adaptive step size: larger for larger radii
                step = base_step * (1.0 + d / (np.pi/2))
                
                # Move each vertex of this facet towards the circumcenter
                old_face_pts = [pts[i].copy() for i in face]
                
                for i in face:
                    target = c
                    new_pt = pts[i] + step * (target - pts[i])
                    new_pt_norm = np.linalg.norm(new_pt)
                    if new_pt_norm > 1e-10:
                        pts[i] = new_pt / new_pt_norm
                        improved = True
            
            if not improved:
                break
                
            # Check if improvement
            new_cr = compute_covering_radius(pts)
            if new_cr < current_cr:
                current_cr = new_cr
            else:
                # Revert if no improvement
                for i, old_pt in zip(face, old_face_pts):
                    pts[i] = old_pt
                break
                
        return pts

    def covering_energy(x, beta=50.0):
        """
        Soft-max approximation of the covering radius.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        if not hasattr(covering_energy, 'test_pts'):
            # Generate Fibonacci sphere points
            n_test = 5000
            i = np.arange(n_test) + 0.5
            z = 1.0 - 2.0 * i / n_test
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            covering_energy.test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
        test_pts = covering_energy.test_pts
        
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        min_dists = np.min(dists, axis=1)
        
        max_d = np.max(min_dists)
        if max_d == 0:
            return 0.0
        exp_dists = np.exp(beta * (min_dists - max_d))
        soft_max = max_d + (1.0 / beta) * np.log(np.sum(exp_dists))
        
        return soft_max

    def covering_energy_grad(x, beta=50.0):
        """
        Gradient of the soft-max covering energy.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        test_pts = covering_energy.test_pts
        n_test = len(test_pts)
        
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        min_idx = np.argmin(dists, axis=1)
        min_dists = np.take_along_axis(dists, min_idx[:, None], axis=1).flatten()
        
        max_d = np.max(min_dists)
        if max_d == 0:
            return np.zeros_like(x)
            
        exp_dists = np.exp(beta * (min_dists - max_d))
        weights = exp_dists / np.sum(exp_dists)
        
        grad_pts = np.zeros_like(pts)
        
        for i in range(n_test):
            j = min_idx[i]
            w = weights[i]
            if w == 0:
                continue
            
            u = dots[i, j]
            denom = np.sqrt(1.0 - u * u + 1e-12)
            grad_u = -1.0 / denom
            grad_pts[j] += w * beta * grad_u * test_pts[i]
            
        v = x.reshape(-1, 3)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        
        grad_v = np.zeros_like(v)
        for k in range(N):
            n_k = norms[k, 0]
            v_k = v[k]
            dot_vg = np.dot(v_k, grad_pts[k])
            grad_v[k] = grad_pts[k] / n_k - v_k * (dot_vg / (n_k ** 3))
            
        return grad_v.flatten()

    # Strategy: Multi-start with different initializations
    # 1. Fibonacci Sphere
    # 2. Random Points (multiple tries)
    
    best_cr = 180.0
    best_pts = None
    
    # Initial Configuration 1: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization 1
    if time.time() < deadline - 15:
        try:
            res = minimize(
                covering_energy, 
                init_pts.flatten(), 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
            )
            
            if res.fun < np.inf:
                pts_opt = res.x.reshape(-1, 3)
                pts_opt = normalize(pts_opt)
                cr_opt = compute_covering_radius(pts_opt)
                
                if cr_opt < best_cr:
                    best_cr = cr_opt
                    best_pts = pts_opt
                    
                # Lloyd Polish
                if time.time() < deadline - 10:
                    pts_lloyd = lloyd_polish(pts_opt, max_iters=30, step_size=0.1)
                    cr_lloyd = compute_covering_radius(pts_lloyd)
                    if cr_lloyd < best_cr:
                        best_cr = cr_lloyd
                        best_pts = pts_lloyd
                        
                # Micro Polish
                if time.time() < deadline - 5:
                    pts_micro = micro_polish(best_pts, max_iters=50, base_step=0.05)
                    cr_micro = compute_covering_radius(pts_micro)
                    if cr_micro < best_cr:
                        best_cr = cr_micro
                        best_pts = pts_micro
        except Exception:
            pass

    # Initial Configuration 2: Random Points (Seed 1)
    if time.time() < deadline - 15:
        np.random.seed(42)
        init_pts2 = np.random.randn(N, 3)
        init_pts2 = normalize(init_pts2)
        
        try:
            res = minimize(
                covering_energy, 
                init_pts2.flatten(), 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
            )
            
            if res.fun < np.inf:
                pts_opt = res.x.reshape(-1, 3)
                pts_opt = normalize(pts_opt)
                cr_opt = compute_covering_radius(pts_opt)
                
                if cr_opt < best_cr:
                    best_cr = cr_opt
                    best_pts = pts_opt
                    
                if time.time() < deadline - 10:
                    pts_lloyd = lloyd_polish(pts_opt, max_iters=30, step_size=0.1)
                    cr_lloyd = compute_covering_radius(pts_lloyd)
                    if cr_lloyd < best_cr:
                        best_cr = cr_lloyd
                        best_pts = pts_lloyd
                        
                if time.time() < deadline - 5:
                    pts_micro = micro_polish(best_pts, max_iters=50, base_step=0.05)
                    cr_micro = compute_covering_radius(pts_micro)
                    if cr_micro < best_cr:
                        best_cr = cr_micro
                        best_pts = pts_micro
        except Exception:
            pass

    # Initial Configuration 3: Random Points (Seed 2)
    if time.time() < deadline - 15:
        np.random.seed(123)
        init_pts3 = np.random.randn(N, 3)
        init_pts3 = normalize(init_pts3)
        
        try:
            res = minimize(
                covering_energy, 
                init_pts3.flatten(), 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
            )
            
            if res.fun < np.inf:
                pts_opt = res.x.reshape(-1, 3)
                pts_opt = normalize(pts_opt)
                cr_opt = compute_covering_radius(pts_opt)
                
                if cr_opt < best_cr:
                    best_cr = cr_opt
                    best_pts = pts_opt
                    
                if time.time() < deadline - 10:
                    pts_lloyd = lloyd_polish(pts_opt, max_iters=30, step_size=0.1)
                    cr_lloyd = compute_covering_radius(pts_lloyd)
                    if cr_lloyd < best_cr:
                        best_cr = cr_lloyd
                        best_pts = pts_lloyd
                        
                if time.time() < deadline - 5:
                    pts_micro = micro_polish(best_pts, max_iters=50, base_step=0.05)
                    cr_micro = compute_covering_radius(pts_micro)
                    if cr_micro < best_cr:
                        best_cr = cr_micro
                        best_pts = pts_micro
        except Exception:
            pass

    # Final Polish on the best found
    if best_pts is not None and time.time() < deadline - 5:
        # Try a few different step sizes for Lloyd
        for step in [0.2, 0.05, 0.01]:
            if time.time() > deadline - 2:
                break
            pts_final = lloyd_polish(best_pts, max_iters=20, step_size=step)
            cr_final = compute_covering_radius(pts_final)
            if cr_final < best_cr:
                best_cr = cr_final
                best_pts = pts_final
                
        # Final micro polish
        if time.time() < deadline - 2:
            pts_final = micro_polish(best_pts, max_iters=30, base_step=0.02)
            cr_final = compute_covering_radius(pts_final)
            if cr_final < best_cr:
                best_cr = cr_final
                best_pts = pts_final

    # If no valid points found (shouldn't happen), return Fibonacci
    if best_pts is None:
        return init_pts

    return normalize(best_pts)
# EVOLVE-BLOCK-END
