"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius_and_worst_indices(pts):
        """
        Computes the exact covering radius and identifies the facet responsible for it.
        Returns (radius_degrees, worst_face_indices, circumcenter).
        """
        try:
            hull = ConvexHull(pts)
            max_r_rad = 0.0
            worst_face = None
            worst_center = None
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of the spherical triangle
                v1 = p0 - p1
                v2 = p0 - p2
                c_vec = np.cross(v1, v2)
                
                norm_c = np.linalg.norm(c_vec)
                if norm_c < 1e-10:
                    continue
                    
                c_vec /= norm_c
                
                # Determine correct hemisphere for circumcenter
                centroid = (p0 + p1 + p2) / 3.0
                centroid /= np.linalg.norm(centroid)
                
                if np.dot(c_vec, centroid) < 0:
                    c_vec = -c_vec
                    
                # Check if circumcenter is inside the spherical triangle
                n1 = np.cross(p1, p2)
                n2 = np.cross(p2, p0)
                n3 = np.cross(p0, p1)
                
                # Orient normals towards centroid
                if np.dot(centroid, n1) < 0: n1 = -n1
                if np.dot(centroid, n2) < 0: n2 = -n2
                if np.dot(centroid, n3) < 0: n3 = -n3
                
                if (np.dot(c_vec, n1) >= -1e-8 and 
                    np.dot(c_vec, n2) >= -1e-8 and 
                    np.dot(c_vec, n3) >= -1e-8):
                    
                    dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                    if dist > max_r_rad:
                        max_r_rad = dist
                        worst_face = face
                        worst_center = c_vec
                        
            return np.degrees(max_r_rad), worst_face, worst_center
        except Exception:
            return 180.0, None, None

    def lloyd_relax(pts, steps=20):
        n_grid = 10000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        grid_pts = normalize(grid_pts)
        
        current_pts = pts.copy()
        for _ in range(steps):
            dots = grid_pts @ current_pts.T
            nearest_idx = np.argmax(dots, axis=1)
            new_pts = np.zeros_like(current_pts)
            for k in range(N):
                mask = (nearest_idx == k)
                if np.any(mask):
                    new_pts[k] = np.sum(grid_pts[mask], axis=0)
                else:
                    new_pts[k] = current_pts[k]
            new_pts = normalize(new_pts)
            current_pts = new_pts
        return current_pts

    def refine_with_hull(pts, iterations=10):
        current_pts = pts.copy()
        best_pts_so_far = current_pts.copy()
        best_cr, _, _ = compute_covering_radius_and_worst_indices(current_pts)
        
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
            cr, worst_face, worst_center = compute_covering_radius_and_worst_indices(current_pts)
            
            if worst_face is None: break
            
            if cr < best_cr:
                best_cr = cr
                best_pts_so_far = current_pts.copy()
            
            # Move vertices of worst face towards center
            step_size = 0.1
            for idx in worst_face:
                new_pos = current_pts[idx] + step_size * (worst_center - current_pts[idx])
                current_pts[idx] = new_pos / np.linalg.norm(new_pos)
                
            # Small Lloyd relaxation
            current_pts = lloyd_relax(current_pts, steps=2)
            
        return best_pts_so_far

    # Main Strategy
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = normalize(fib_pts)
    
    best_pts = fib_pts.copy()
    best_cr, _, _ = compute_covering_radius_and_worst_indices(best_pts)
    
    # 2. Optimize using L-BFGS-B on a smooth "soft-max" covering energy
    def soft_covering_energy_and_grad(pts):
        beta = 20.0
        n_grid = 5000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        dots = grid_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        max_dists = np.max(dists, axis=1, keepdims=True)
        exp_terms = np.exp(-beta * (dists - max_dists))
        log_S = np.log(np.sum(exp_terms, axis=1)) + (-beta * max_dists).squeeze(axis=1)
        m = -1.0 / beta * log_S
        
        max_m = np.max(m)
        exp_m = np.exp(beta * (m - max_m))
        T = np.sum(exp_m)
        P = exp_m / T
        
        energy = np.log(T) + beta * max_m
        
        sum_exp_terms = np.sum(exp_terms, axis=1, keepdims=True)
        W = exp_terms / sum_exp_terms
        
        denom = np.sqrt(1.0 - dots**2)
        denom[denom < 1e-8] = 1e-8
        
        coeff = P[:, np.newaxis] * W / denom
        grad = -beta * (coeff.T @ grid_pts)
        
        dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - dot_prod * pts
        
        return energy, grad

    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        pts = normalize(pts)
        
        if time.time() > deadline - 5:
            raise ValueError("Timeout")
            
        e, g = soft_covering_energy_and_grad(pts)
        return e, g
        
    x0 = fib_pts.flatten()
    
    try:
        result = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 200, 'maxfun': 1000, 'disp': False}
        )
        
        if result.success or result.status in [1, 2]:
            opt_pts = normalize(result.x.reshape(N, 3))
            opt_cr, _, _ = compute_covering_radius_and_worst_indices(opt_pts)
            if opt_cr < best_cr:
                best_cr = opt_cr
                best_pts = opt_pts
    except Exception as e:
        pass
        
    # 3. Local Refinement with Hull-based search
    lloyd_pts = lloyd_relax(best_pts, steps=10)
    lloyd_cr, _, _ = compute_covering_radius_and_worst_indices(lloyd_pts)
    if lloyd_cr < best_cr:
        best_cr = lloyd_cr
        best_pts = lloyd_pts

    final_pts = refine_with_hull(best_pts, iterations=15)
    final_cr, _, _ = compute_covering_radius_and_worst_indices(final_pts)
    if final_cr < best_cr:
        best_cr = final_cr
        best_pts = final_pts

    return best_pts
# EVOLVE-BLOCK-END
