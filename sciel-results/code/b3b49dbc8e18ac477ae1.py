import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Utility Functions
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def compute_covering_radius(pts):
        """
        Computes the covering radius using ConvexHull.
        Returns radius in radians.
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.pi
            
        max_radius = 0.0
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c
                
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            if radius > max_radius:
                max_radius = radius
                
        return max_radius

    # 2. Objective Function for Optimizer
    # We optimize a composite objective:
    # 1. Riesz S-Energy (s=2) to spread points uniformly (minimizes max distance indirectly)
    # 2. A penalty term if covering radius exceeds a target (soft constraint)
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        
        # Ensure points are on sphere (projection inside optimizer is not ideal, 
        # but we can use a penalty or just trust the optimizer stays close if started well.
        # Better: use a penalty for norm deviation to keep them on sphere during iterations)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norm_penalty = np.sum((norms - 1.0)**2)
        
        # Riesz Energy S=2: sum over pairs of 1/||pi - pj||^2
        # This is equivalent to maximizing the minimum distance roughly, but globally.
        # It is smooth and differentiable.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Energy = sum 1/d^2
        energy = np.sum(1.0 / dist_sq)
        
        # Gradient of energy w.r.t pts
        # d/dpi (1/||pi-pj||^2) = -2 * (pi - pj) / ||pi-pj||^4
        # Sum over j
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # grad_i = sum_j -2 * (pi - pj) * inv_dist_sq_ij
        # diff is pi - pj
        grad_energy = -2.0 * np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Gradient of norm penalty: 2 * (pi - pi/||pi||) = 2*pi*(1 - 1/||pi||)
        # Actually simpler: penalty = sum (||pi||^2 - 1)^2
        # grad = 4 * (||pi||^2 - 1) * pi
        norm_pen_grad = 4.0 * (np.sum(pts**2, axis=1, keepdims=True) - 1.0) * pts
        
        # Combine
        # We want to minimize energy.
        # We also want to minimize covering radius, but it's non-smooth.
        # Strategy: Minimize energy primarily. The "best" known configurations for small N
        # are often local minima of Riesz energy or related potentials.
        # For N=50, the Hardin-Sloane-Smith configuration is likely near a local minimum of 
        # a repulsive potential.
        
        # To specifically target the covering radius, we can add a soft penalty 
        # if we detect the radius is large. However, computing the hull inside the 
        # objective is expensive and non-smooth.
        
        # Alternative: Just minimize Riesz Energy. It is a proven heuristic for sphere packing/covering.
        # Let's try a weighted sum if we can estimate radius cheaply? No, hull is expensive.
        
        # Let's stick to Riesz Energy minimization. It forces uniform distribution.
        # If we start from a good seed (Fibonacci), L-BFGS should find a deep local minimum.
        
        val = energy + 100.0 * norm_penalty
        grad = grad_energy + norm_pen_grad
        
        return val, grad.flatten()

    # 3. Optimization Loop with Multi-Start
    
    best_points = None
    best_radius = np.pi
    
    # Generate diverse initial seeds
    seeds = []
    seeds.append(fibonacci_sphere(N))
    
    # Add random perturbations to Fibonacci
    for _ in range(3):
        seed = fibonacci_sphere(N).copy()
        noise = np.random.randn(N, 3) * 0.1
        seed += noise
        seed = normalize(seed)
        seeds.append(seed)
        
    # Add a few random points (might be bad, but helps escape if combined with restarts)
    # Actually, random is usually bad for high N. Stick to structured + perturbed.
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 20:
            break
            
        x0 = seed.flatten()
        
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
            )
        except Exception:
            continue
            
        opt_points = normalize(res.x.reshape(N, 3))
        
        # Evaluate the true metric (Covering Radius)
        radius = compute_covering_radius(opt_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = opt_points.copy()
            
            # Check if we are close to the theoretical limit or known best
            # Known best is ~18.3 deg = 0.319 rad
            if best_radius < 0.32:
                # If we are very good, we can stop early or try to polish
                pass

    # 4. Local Polish using Convex Hull Gradient
    # L-BFGS on Riesz energy might get stuck in a local minimum that is not the 
    # optimal covering configuration. The covering radius is determined by the 
    # largest facet. We can do a targeted local search on the worst facet.
    
    if best_points is not None and time.time() < deadline - 5:
        current_points = best_points.copy()
        current_radius = best_radius
        
        # Simple local search: move vertices of the worst facet towards the circumcenter
        step_size = 0.005
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
                
            radius, worst_simplex, worst_center = None, None, None
            # We need a function that returns the worst simplex and center
            # Let's re-implement the radius calc to return details
            
            try:
                hull = ConvexHull(current_points)
            except Exception:
                break
                
            max_rad = 0.0
            w_s = None
            w_c = None
            
            for simplex in hull.simplices:
                v1 = current_points[simplex[0]]
                v2 = current_points[simplex[1]]
                v3 = current_points[simplex[2]]
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                    
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                    
                d = np.linalg.norm(c - v1)
                rad = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
                
                if rad > max_rad:
                    max_rad = rad
                    w_s = simplex
                    w_c = c
                    
            if w_s is None:
                break
                
            if max_rad < current_radius - 1e-9:
                current_radius = max_rad
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = current_points.copy()
                    
            # Move vertices of worst simplex towards center
            for idx in w_s:
                v = current_points[idx]
                dir_vec = w_c - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir < 1e-10:
                    continue
                dir_vec /= norm_dir
                new_v = v + step_size * dir_vec
                new_v /= np.linalg.norm(new_v)
                current_points[idx] = new_v
                
            step_size *= 0.995
            if step_size < 1e-6:
                break

    # Final normalization
    if best_points is None:
        return fibonacci_sphere(N)
        
    best_points = normalize(best_points)
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
