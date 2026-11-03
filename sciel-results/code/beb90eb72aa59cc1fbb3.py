"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _min_triangle_area_fast(points):
    """
    Fast vectorized computation of the minimum triangle area.
    points: (N, 2) array.
    Returns: min area (float).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            d_ij = pj - pi
            diffs = points - pi
            crosses = d_ij[0] * diffs[:, 1] - d_ij[1] * diffs[:, 0]
            areas = 0.5 * np.abs(crosses)
            areas[i] = float('inf')
            areas[j] = float('inf')
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
    return min_area


def _is_inside_triangle(points):
    """
    Check if all points are inside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Returns: True if all points are inside (with small tolerance).
    """
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    
    # y >= 0
    if np.any(y < -tol):
        return False
    
    # y <= sqrt(3)*x (left edge)
    if np.any(y > np.sqrt(3) * x + tol):
        return False
    
    # y <= sqrt(3)*(1-x) (right edge)
    if np.any(y > np.sqrt(3) * (1 - x) + tol):
        return False
    
    # x >= 0, x <= 1
    if np.any(x < -tol) or np.any(x > 1 + tol):
        return False
    
    return True


def _clip_to_triangle(points):
    """
    Clip points to be inside the triangle.
    """
    result = points.copy()
    for i in range(len(result)):
        x, y = result[i]
        
        # Clip to bounding box first
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(np.sqrt(3)/2, y))
        
        # Left edge: y <= sqrt(3)*x
        if y > np.sqrt(3) * x:
            # Project to line y = sqrt(3)*x
            # Closest point: minimize (x'-x)^2 + (y'-y)^2 subject to y' = sqrt(3)*x'
            # Lagrangian: 2(x'-x) + lambda*sqrt(3) = 0, 2(y'-y) + lambda = 0
            # From second: lambda = -2(y'-y)
            # From first: x' - x = -lambda*sqrt(3)/2 = (y'-y)*sqrt(3)
            # So x' = x + sqrt(3)*(y'-y), and y' = sqrt(3)*x'
            # y' = sqrt(3)*(x + sqrt(3)*(y'-y)) = sqrt(3)*x + 3*(y'-y)
            # y' = sqrt(3)*x + 3*y' - 3*y
            # 3*y - y' = sqrt(3)*x + 3*y' - y'  -- wait, let me redo
            # y' = sqrt(3)*x + 3(y' - y)
            # y' = sqrt(3)*x + 3y' - 3y
            # 3y - sqrt(3)*x = 2y'
            # y' = (3y - sqrt(3)*x) / 2
            # x' = y' / sqrt(3)
            y_new = (3 * y - np.sqrt(3) * x) / 2.0
            x_new = y_new / np.sqrt(3)
            # But this might not be the closest point if the projection is outside the segment
            # For now, use this projection
            x, y = x_new, y_new
        
        # Right edge: y <= sqrt(3)*(1-x)
        if y > np.sqrt(3) * (1 - x):
            # Line: y = sqrt(3) - sqrt(3)*x, or sqrt(3)*x + y = sqrt(3)
            # Minimize (x'-x)^2 + (y'-y)^2 subject to sqrt(3)*x' + y' = sqrt(3)
            # Lagrangian: 2(x'-x) + lambda*sqrt(3) = 0, 2(y'-y) + lambda = 0
            # lambda = -2(y'-y)
            # x' - x = (y'-y)*sqrt(3)
            # x' = x + sqrt(3)*(y'-y)
            # sqrt(3)*(x + sqrt(3)*(y'-y)) + y' = sqrt(3)
            # sqrt(3)*x + 3(y'-y) + y' = sqrt(3)
            # sqrt(3)*x + 4y' - 3y = sqrt(3)
            # 4y' = sqrt(3) - sqrt(3)*x + 3y
            # y' = (sqrt(3)*(1-x) + 3y) / 4
            y_new = (np.sqrt(3) * (1 - x) + 3 * y) / 4.0
            x_new = (np.sqrt(3) - y_new) / np.sqrt(3)
            x, y = x_new, y_new
        
        # Final clip to bounding box
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(np.sqrt(3)/2, y))
        
        result[i] = [x, y]
    
    return result


def _objective_flat(flat_points):
    """
    Objective function for optimization.
    Returns: negative min area (to maximize) + penalty for being outside.
    """
    points = flat_points.reshape(N, 2)
    
    # Check if inside
    if not _is_inside_triangle(points):
        # Compute penalty
        x = points[:, 0]
        y = points[:, 1]
        pen = 0.0
        # y < 0
        pen += np.sum(np.clip(-y, 0, None))
        # y > sqrt(3)*x
        pen += np.sum(np.clip(y - np.sqrt(3) * x, 0, None))
        # y > sqrt(3)*(1-x)
        pen += np.sum(np.clip(y - np.sqrt(3) * (1 - x), 0, None))
        # x < 0 or x > 1
        pen += np.sum(np.clip(-x, 0, None)) + np.sum(np.clip(x - 1, 0, None))
        return pen * 1000.0
    
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _generate_structured_inits():
    """
    Generate diverse structured initial points.
    """
    inits = []
    
    # 1. Regular grid in barycentric coords
    # Place points on a 4x4 grid in barycentric space, then pick 11
    for gx in range(4):
        for gy in range(4):
            for gz in range(4):
                if gx + gy + gz == 3:
                    bary = np.array([gx, gy, gz], dtype=float) / 3.0
                    x = bary[1] + 0.5 * bary[2]
                    y = np.sqrt(3)/2 * bary[2]
                    inits.append([x, y])
    
    # Pick 11 from these
    if len(inits) >= 11:
        # Take first 11
        inits = inits[:11]
    
    # 2. Symmetric pattern: center + 10 around
    bary_center = [1/3, 1/3, 1/3]
    x_c = 1/3 + 0.5 * 1/3
    y_c = np.sqrt(3)/2 * 1/3
    inits2 = [[x_c, y_c]]
    for k in range(10):
        a = 2 * np.pi * k / 10
        r = 0.15
        # In barycentric, move in direction
        bary_k = [1/3 + r * np.cos(a), 1/3 + r * np.cos(a - np.pi/3), 1/3 + r * np.sin(a)]
        bary_k = np.clip(bary_k, 0, None)
        s = bary_k.sum()
        if s > 0:
            bary_k /= s
        x_k = bary_k[1] + 0.5 * bary_k[2]
        y_k = np.sqrt(3)/2 * bary_k[2]
        inits2.append([x_k, y_k])
    inits.append(inits2)
    
    # 3. Random barycentric
    rng = np.random.default_rng(42)
    for _ in range(5):
        bary = rng.random((N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = np.sqrt(3)/2 * bary[:, 2]
        inits.append(np.column_stack((x, y)))
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Generate structured inits
    inits = _generate_structured_inits()
    
    # Evaluate each init
    for init in inits:
        pts = np.array(init, dtype=float).reshape(N, 2)
        if _is_inside_triangle(pts):
            score = _min_triangle_area_fast(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    # If no valid init, use random
    if best_points is None:
        rng = np.random.default_rng(42)
        bary = rng.random((N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = np.sqrt(3)/2 * bary[:, 2]
        best_points = np.column_stack((x, y))
        best_score = _min_triangle_area_fast(best_points)
    
    # Run differential evolution
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    try:
        result = differential_evolution(
            _objective_flat,
            bounds,
            maxiter=500,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol',
            x0=best_points.flatten(),
        )
        
        if result.fun < 0:
            pts = result.x.reshape(N, 2)
            if _is_inside_triangle(pts):
                score = _min_triangle_area_fast(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
    except Exception:
        pass
    
    # Polish with Nelder-Mead from best point
    if time.time() < deadline - 10:
        try:
            nm_result = minimize(
                _objective_flat,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if nm_result.fun < 0:
                pts = nm_result.x.reshape(N, 2)
                if _is_inside_triangle(pts):
                    score = _min_triangle_area_fast(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
        except Exception:
            pass
    
    # Multi-start NM from diverse points
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation of best point
            perturb = rng.normal(0, 0.05, size=(N, 2))
            pts = best_points + perturb
            pts = _clip_to_triangle(pts)
            
            if not _is_inside_triangle(pts):
                continue
            
            try:
                nm_res = minimize(
                    _objective_flat,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if nm_res.fun < 0:
                    pts2 = nm_res.x.reshape(N, 2)
                    if _is_inside_triangle(pts2):
                        score = _min_triangle_area_fast(pts2)
                        if score > best_score:
                            best_score = score
                            best_points = pts2.copy()
            except Exception:
                pass
    
    # Final safety check
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
