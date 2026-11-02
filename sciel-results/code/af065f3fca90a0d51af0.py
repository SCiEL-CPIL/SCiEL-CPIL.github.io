"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

# Pre-compute a dense grid for the smooth objective
# 8000 points is a good balance between accuracy and computation time
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_FINE = _make_fib_grid(8000)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    def _fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def _compute_exact_covering_radius(points):
        """
        Compute the exact covering radius by checking all facets of the convex hull.
        Returns the max circumradius in radians.
        """
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            max_radius = 0.0
            for idx in facets:
                p1 = points[idx[0]]
                p2 = points[idx[1]]
                p3 = points[idx[2]]
                
                # Compute circumcenter
                # v is orthogonal to (p1-p2) and (p1-p3)
                v1 = p1 - p2
                v2 = p1 - p3
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                
                if norm_n < 1e-10:
                    # Degenerate, use max edge
                    d1 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
                    d3 = np.arccos(np.clip(np.dot(p3, p1), -1, 1))
                    r = max(d1, d2, d3)
                else:
                    n = n / norm_n
                    # Ensure n points towards the "interior" of the spherical triangle
                    # Check dot product with centroid
                    centroid = (p1 + p2 + p3) / 3.0
                    if np.dot(n, centroid) < 0:
                        n = -n
                    
                    # Circumradius is angular distance from n to p1
                    dot = np.dot(n, p1)
                    dot = np.clip(dot, -1.0, 1.0)
                    r = np.arccos(dot)
                
                if r > max_radius:
                    max_radius = r
            
            return max_radius
        except Exception:
            return np.inf

    def _soft_max_objective(flat_pts, beta=50.0):
        """
        Smooth surrogate for covering radius using soft-max.
        Returns objective value and gradient.
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # Compute distances from all grid points to all configuration points
        # dots[i, j] = dot(GRID[i], pts[j])
        dots = GRID_FINE @ pts.T  # shape (G, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # shape (G, N)
        
        # Soft-min over j (config points) for each grid point i
        # We want min_j dists[i, j]
        # soft_min = -1/beta * log(sum(exp(-beta * dists)))
        neg_beta_dists = -beta * dists
        max_neg = np.max(neg_beta_dists, axis=1, keepdims=True)
        # log_sum_exp = log(sum(exp(x))) = max_x + log(sum(exp(x - max_x)))
        log_sum_exp_inner = np.log(np.sum(np.exp(neg_beta_dists - max_neg), axis=1)) + max_neg.ravel()
        soft_min = -log_sum_exp_inner / beta  # shape (G,)
        
        # Soft-max over grid points i
        # We want max_i soft_min[i]
        # soft_max = 1/beta * log(sum(exp(beta * soft_min)))
        beta_soft_min = beta * soft_min
        max_outer = np.max(beta_soft_min)
        log_sum_exp_outer = np.log(np.sum(np.exp(beta_soft_min - max_outer))) + max_outer
        objective = log_sum_exp_outer / beta
        
        # Gradient computation
        # d(objective)/d(pts)
        
        # Outer weights: softmax(beta * soft_min)
        outer_weights = np.exp(beta_soft_min - max_outer)
        outer_weights /= np.sum(outer_weights)  # shape (G,)
        
        # Inner weights: softmax(-beta * dists)
        inner_weights = np.exp(neg_beta_dists - max_neg)  # shape (G, N)
        inner_weights /= np.sum(inner_weights, axis=1, keepdims=True)  # shape (G, N)
        
        # Gradient of arccos(x) is -1/sqrt(1-x^2)
        # d/dp_j arccos(g_i . p_j) = - (g_i - p_j (g_i . p_j)) / sqrt(1 - (g_i . p_j)^2)
        
        one_minus_dot_sq = 1.0 - dots * dots
        one_minus_dot_sq = np.maximum(one_minus_dot_sq, 1e-12)
        sqrt_oms = np.sqrt(one_minus_dot_sq)  # shape (G, N)
        
        # weighted_coeff[i, j] = outer_weights[i] * inner_weights[i, j]
        weighted_coeff = outer_weights[:, np.newaxis] * inner_weights  # shape (G, N)
        
        # coeff_div_sqrt[i, j] = weighted_coeff[i, j] / sqrt_oms[i, j]
        coeff_div_sqrt = weighted_coeff / sqrt_oms  # shape (G, N)
        
        # grad_pts[j] = - sum_i ( coeff_div_sqrt[i, j] * (GRID[i] - pts[j] * dots[i, j]) )
        # = - ( GRID.T @ coeff_div_sqrt )_j + pts[j] * sum_i ( coeff_div_sqrt[i, j] * dots[i, j] )
        
        term1 = GRID_FINE.T @ coeff_div_sqrt  # shape (3, N)
        term2_scalar = np.sum(coeff_div_sqrt * dots, axis=0)  # shape (N,)
        
        grad_pts = -(term1.T - term2_scalar[:, np.newaxis] * pts)
        
        # Project to tangent space to ensure we stay on the sphere
        # grad_tangent = grad_pts - (grad_pts . pts) * pts
        grad_tangent = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
        
        grad_flat = grad_tangent.ravel()
        
        return objective, grad_flat

    def _targeted_refinement(points, max_iters=2000):
        """
        Refine by targeting the worst facet: move vertices towards circumcenter.
        """
        pts = points.copy()
        prev_radius = _compute_exact_covering_radius(pts)
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
                
            facets = hull.simplices
            M = len(facets)
            radii = np.zeros(M)
            centers = np.zeros((M, 3))
            
            for i in range(M):
                idx = facets[i]
                p1 = pts[idx[0]]
                p2 = pts[idx[1]]
                p3 = pts[idx[2]]
                
                v1 = p1 - p2
                v2 = p1 - p3
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                
                if norm_n < 1e-10:
                    # Degenerate
                    d1 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
                    d3 = np.arccos(np.clip(np.dot(p3, p1), -1, 1))
                    r = max(d1, d2, d3)
                    c = (p1 + p2 + p3) / 3.0
                    c = c / np.linalg.norm(c)
                else:
                    n = n / norm_n
                    centroid = (p1 + p2 + p3) / 3.0
                    if np.dot(n, centroid) < 0:
                        n = -n
                    c = n
                    dot = np.dot(n, p1)
                    dot = np.clip(dot, -1.0, 1.0)
                    r = np.arccos(dot)
                
                radii[i] = r
                centers[i] = c
            
            max_radius = np.max(radii)
            
            # Check for improvement
            if max_radius < prev_radius - 1e-12:
                prev_radius = max_radius
            else:
                # If no improvement, break or reduce step size
                # For simplicity, we just break if we stagnate too much
                # But let's try a few more iterations with smaller steps
                if it > 50:
                    break

            # Identify facets with large radii
            # Threshold: 0.95 * max_radius to focus on the worst ones
            threshold = 0.95 * max_radius
            mask = radii > threshold
            
            forces = np.zeros((N, 3))
            
            for i in range(M):
                if not mask[i]:
                    continue
                    
                c = centers[i]
                idx = facets[i]
                p1 = pts[idx[0]]
                p2 = pts[idx[1]]
                p3 = pts[idx[2]]
                
                # Move vertices towards circumcenter
                # The vector from p to c is in the tangent space? No, it's a chord.
                # We project the movement onto the tangent space.
                # Or simply add the vector and renormalize.
                # Let's use a small step size.
                step = 0.05
                
                v1 = c - p1
                v2 = c - p2
                v3 = c - p3
                
                forces[idx[0]] += v1
                forces[idx[1]] += v2
                forces[idx[2]] += v3
            
            # Apply forces
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            normalized_forces = forces / force_norms
            
            pts = pts + step * normalized_forces
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
            # Check time
            if time.time() > deadline - 2.0:
                break

        return pts

    # 1. Initialize
    points = _fibonacci_sphere(N)
    
    # 2. Phase 1: Smooth optimization with L-BFGS-B
    # Use a subset of the time budget for this phase
    # L-BFGS-B can be slow if the grid is too large, but 8000 is manageable.
    try:
        # Flatten points
        x0 = points.ravel()
        
        # Define objective and gradient for scipy
        def obj_grad(x):
            val, grad = _soft_max_objective(x, beta=50.0)
            return val, grad
            
        # Run minimization
        # maxiter limited to ensure we don't blow the time budget
        # gtol small to ensure convergence
        res = minimize(
            obj_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'gtol': 1e-6, 'maxfun': 200}
        )
        
        if res.success or res.fun < np.inf:
            points = res.x.reshape(N, 3)
            # Renormalize
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
    except Exception:
        pass # If optimization fails, continue with initial points

    # 3. Phase 2: Targeted Geometric Refinement
    points = _targeted_refinement(points)
    
    return points

# EVOLVE-BLOCK-END
