import numpy as np
import time
import itertools

N = 11

# EVOLVE-BLOCK-START
import random

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # Vertices of the equilateral triangle
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, np.sqrt(3) / 2])
    
    # Area of the big triangle
    big_area = 0.5 * np.linalg.norm(np.cross(V1 - V0, V2 - V0))
    
    def point_in_triangle(p, eps=1e-9):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(p, V0, V1)
        d2 = sign(p, V1, V2)
        d3 = sign(p, V2, V0)
        has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
        has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
        return not (has_neg and has_pos)

    def compute_areas(points):
        """
        Compute areas for all triplets.
        Returns:
            areas: array of shape (N, N, N) where areas[i,j,k] is area of triangle (i,j,k) for i<j<k
            idx: list of indices (i,j,k) corresponding to the areas
        """
        n = len(points)
        areas = np.zeros((n, n, n))
        idx = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a = 0.5 * np.abs(np.cross(points[j] - points[i], points[k] - points[i]))
                    areas[i, j, k] = a
                    idx.append((i, j, k))
        return areas, idx

    def get_min_area_and_active(points):
        """
        Finds the minimum area triangle and returns its area, indices, and gradient.
        The gradient of the area of triangle (i,j,k) with respect to points is:
        grad_i = 0.5 * perp(p_k - p_j)
        grad_j = 0.5 * perp(p_i - p_k)
        grad_k = 0.5 * perp(p_j - p_i)
        where perp(v) = [-v_y, v_x]
        """
        areas, idx = compute_areas(points)
        flat_areas = areas.flatten()
        min_val = np.min(flat_areas)
        min_flat_idx = np.argmin(flat_areas)
        
        # Convert flat index back to (i, j, k)
        # This is slightly inefficient but N=11 is small. 
        # Better: keep idx list and find min in idx list?
        # Let's just iterate to be safe and simple, or use the idx list from compute_areas
        # Actually compute_areas returns idx list.
        
        # Let's refine: compute_areas is a bit heavy. Let's do a direct search for min and grad.
        min_area = float('inf')
        min_idx = (0, 1, 2)
        
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    a = 0.5 * np.abs(np.cross(points[j] - points[i], points[k] - points[i]))
                    if a < min_area:
                        min_area = a
                        min_idx = (i, j, k)
        
        i, j, k = min_idx
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        # Gradient of signed area? No, area is absolute.
        # If the triangle is oriented positively (cross > 0), grad is as above.
        # If negatively, grad is negated.
        cross_val = np.cross(pj - pi, pk - pi)
        sign_val = 1.0 if cross_val > 0 else -1.0
        
        # Gradients
        # d/dpi (0.5 * sign * cross(pj-pi, pk-pi))
        # cross(u, v) = u_x v_y - u_y v_x
        # Let u = pj - pi, v = pk - pi
        # cross = (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x)
        # d/dpi_x: -(pk_y - pi_y) - (-(pj_y - pi_y)) = -pk_y + pi_y + pj_y - pi_y = pj_y - pk_y
        # d/dpi_y: (pj_x - pi_x) - (pk_x - pi_x) = pj_x - pk_x
        # So d/dpi = 0.5 * sign * (pj_y - pk_y, pj_x - pk_x) = 0.5 * sign * perp(pk - pj)
        
        # perp(v) = [-v_y, v_x]
        # perp(pk - pj) = [-(pk_y - pj_y), pk_x - pj_x] = [pj_y - pk_y, pk_x - pj_x]
        # Wait.
        # Let's check:
        # A = 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
        # Let D = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
        # dD/dxi = -(yk-yi) - (-(yj-yi)) = -yk + yi + yj - yi = yj - yk
        # dD/dyi = (xj-xi) - (xk-xi) = xj - xk
        # So grad_i (signed) = 0.5 * (yj - yk, xj - xk)
        
        # dD/dxj = (yk-yi)
        # dD/dyj = -(xk-xi)
        # So grad_j (signed) = 0.5 * (yk-yi, -(xk-xi)) = 0.5 * (-(yi-yk), -(xk-xi)) = -0.5 * (yi-yk, xk-xi)
        # Note: perp(pi - pk) = [-(pi_y - pk_y), pi_x - pk_x] = [pk_y - pi_y, pi_x - pk_x]
        # My formula: 0.5 * (yk-yi, xi-xk). 
        # perp(pi - pk) = [-(yi-yk), xi-xk] = [yk-yi, xi-xk]. Matches.
        
        # dD/dxk = -(yj-yi)
        # dD/dyk = (xj-xi)
        # So grad_k (signed) = 0.5 * (-(yj-yi), xj-xi) = 0.5 * (yi-yj, xj-xi)
        # perp(pj - pi) = [-(yj-yi), xj-xi] = [yi-yj, xj-xi]. Matches.
        
        # So:
        # grad_i = 0.5 * sign * perp(pk - pj)
        # grad_j = 0.5 * sign * perp(pi - pk)
        # grad_k = 0.5 * sign * perp(pj - pi)
        
        def perp(v):
            return np.array([-v[1], v[0]])
            
        grad = np.zeros_like(points)
        grad[i] = 0.5 * sign_val * perp(pk - pj)
        grad[j] = 0.5 * sign_val * perp(pi - pk)
        grad[k] = 0.5 * sign_val * perp(pj - pi)
        
        return min_area, grad

    def project_to_triangle(points):
        """
        Projects points back into the triangle if they went out.
        Simple method: if a point is outside, move it towards the centroid.
        """
        centroid = (V0 + V1 + V2) / 3.0
        for i in range(len(points)):
            if not point_in_triangle(points[i]):
                # Move towards centroid until inside
                # Binary search on t in [0, 1] such that p + t*(c-p) is inside
                p = points[i]
                c = centroid
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    np_ = p + mid * (c - p)
                    if point_in_triangle(np_):
                        hi = mid
                    else:
                        lo = mid
                points[i] = p + hi * (c - p)
        return points

    def objective(points):
        """
        Returns the minimum triangle area.
        """
        min_area, _ = get_min_area_and_active(points)
        return min_area

    # Multi-start optimization
    best_score = 0.0
    best_points = None
    
    # Generate initial configurations
    # 1. Random
    # 2. Grid-like (hexagonal packing)
    # 3. Perturbed random
    
    configs = []
    
    # Random starts
    rng = random.Random(42)
    for _ in range(5):
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    p = u * V1 + v * V2
                    pts.append(p)
                    break
        configs.append(np.array(pts))
        
    # Hexagonal grid start (approximate)
    # 11 points in triangle.
    # Layer 1 (bottom): 4 points? No, 11 is small.
    # Let's just use a coarse grid and pick 11 points.
    grid_pts = []
    for i in range(5):
        for j in range(5):
            if i + j < 5:
                p = i * 0.2 * V1 + j * 0.2 * V2
                grid_pts.append(p)
    # Shuffle and take 11
    rng.shuffle(grid_pts)
    if len(grid_pts) >= N:
        configs.append(np.array(grid_pts[:N]))
        
    # Add a specific "known good" structure if possible? 
    # For N=11, optimal is likely a distorted grid.
    
    for init_pts in configs:
        if time.time() - start_time > timeout:
            break
            
        current_pts = init_pts.copy()
        current_score = objective(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
            
        # Gradient Ascent
        # We want to maximize min_area.
        # Step: p_new = p + alpha * grad
        # But grad points in direction of increasing area of the *current* min triangle.
        # This might break other triangles.
        # We need a small step size and re-evaluation.
        
        alpha = 0.05
        for step in range(200):
            if time.time() - start_time > timeout:
                break
                
            min_area, grad = get_min_area_and_active(current_pts)
            
            # Normalize gradient to control step size relative to triangle size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-9:
                break
                
            # Move in direction of gradient
            # We want to increase area, so add grad
            new_pts = current_pts + alpha * grad
            
            # Project back to triangle
            new_pts = project_to_triangle(new_pts)
            
            new_score = objective(new_pts)
            
            if new_score > min_area:
                current_pts = new_pts
                # Maybe increase alpha slightly? Or keep constant.
                # Adaptive alpha: if improvement is small, decrease alpha.
                if new_score - min_area < 1e-6:
                    alpha *= 0.9
            else:
                # If no improvement, decrease alpha and try again
                alpha *= 0.8
                if alpha < 1e-5:
                    break
                    
            if current_score < objective(current_pts):
                pass # Just update local best tracking if needed, but we track global best below
                
        # Update global best
        final_score = objective(current_pts)
        if final_score > best_score:
            best_score = final_score
            best_points = current_pts.copy()
            
    # Final check
    if best_points is None:
        # Fallback to random
        pts = []
        rng = random.Random(0)
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    p = u * V1 + v * V2
                    pts.append(p)
                    break
        best_points = np.array(pts)
        best_score = objective(best_points)
        
    return best_points
# EVOLVE-BLOCK-END
