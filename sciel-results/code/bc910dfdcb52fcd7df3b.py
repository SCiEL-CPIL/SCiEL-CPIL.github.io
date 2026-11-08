"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1
H = np.sqrt(3) / 2.0     # height of the triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def _barycentric_to_cartesian(b):
    """Convert barycentric coordinates (u,v,w) to Cartesian."""
    u, v, w = b
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.array([x, y])


def _cartesian_to_barycentric(p):
    """Convert Cartesian point to barycentric coordinates."""
    x, y = p
    # Solve for u, v, w such that p = u*V0 + v*V1 + w*V2, u+v+w=1
    # Using the standard formula
    w = y / H
    # x = v*1 + w*0.5 => v = x - 0.5*w
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.array([u, v, w])


def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the triangle."""
    b = _cartesian_to_barycentric(p)
    return all(bi >= -eps for bi in b)


def _clamp_to_triangle(p, eps=1e-6):
    """Clamp point to be inside the triangle."""
    b = _cartesian_to_barycentric(p)
    b = np.maximum(b, 0.0)
    s = b.sum()
    if s < 1e-12:
        return (V0 + V1 + V2) / 3.0
    b = b / s
    return _barycentric_to_cartesian(b)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                 - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_triangle_area_normalized(points):
    """Return min triangle area divided by triangle area."""
    return _min_triangle_area(points) / AREA


def _soft_min_area(points, T):
    """
    Compute the soft-min of triangle areas.
    Lower is better for minimization (we maximize area, so we minimize -soft_min).
    soft_min = -1/T * log(sum(exp(-T * area)))
    As T -> inf, this approaches -min(area).
    """
    n = len(points)
    areas = np.array([
        0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                  - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
    ])
    
    # Numerical stability: subtract max before exp
    max_area = np.max(areas)
    if max_area == 0:
        return 0.0
    
    # We want to maximize the min area.
    # Let's define the objective to minimize as: -soft_min(area)
    # soft_min(area) approximates min(area).
    # To handle the "min" in a differentiable way, we use the log-sum-exp trick on the areas.
    # min(a) = -1/T * log(sum(exp(-T*a)))
    
    # We want to MINIMIZE the objective, which is -min_area.
    # So objective = 1/T * log(sum(exp(-T*area)))
    
    # To avoid overflow, we compute:
    # log(sum(exp(-T*area))) = -T*max_area + log(sum(exp(-T*(area - max_area))))
    # Wait, if we subtract max, the terms are exp(-T*(area - max)). Since area <= max, (area-max) <= 0.
    # So -T*(area-max) >= 0. This can overflow if T is large and area is much smaller than max.
    
    # Better:
    # We want to maximize min(area).
    # Equivalent to minimizing -min(area).
    # Approximation: f(x) = -1/T * log( sum_{i} exp(-T * A_i) )
    # This function is smooth and approximates -min(A_i).
    
    # Let's compute S = sum(exp(-T * A_i))
    # log S = log sum exp(-T A_i)
    # To be stable, let A_min = min(A_i).
    # exp(-T A_i) = exp(-T A_min) * exp(-T (A_i - A_min))
    # S = exp(-T A_min) * sum(exp(-T (A_i - A_min)))
    # log S = -T A_min + log(sum(exp(-T (A_i - A_min))))
    
    # The term (A_i - A_min) is >= 0. So -T (A_i - A_min) is <= 0.
    # exp(negative) is safe.
    
    min_area_val = np.min(areas)
    shifted = areas - min_area_val
    exp_terms = np.exp(-T * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return 0.0
        
    log_sum = np.log(sum_exp)
    
    # soft_min = -1/T * ( -T * min_area_val + log_sum ) = min_area_val - log_sum/T
    soft_min_val = min_area_val - log_sum / T
    
    # We want to MAXIMIZE soft_min_val.
    # So we MINIMIZE -soft_min_val.
    return -soft_min_val


def _grad_soft_min_area(points, T):
    """
    Gradient of -soft_min_area with respect to point coordinates.
    """
    n = len(points)
    # Precompute areas and indices
    # This is inefficient for large N, but N=11 is small (165 triangles).
    
    areas = []
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p_i = points[i]
                p_j = points[j]
                p_k = points[k]
                
                # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
                # Let v1 = p_j - p_i, v2 = p_k - p_i
                # Cross product z-component: v1_x * v2_y - v1_y * v2_x
                cross = (p_j[0] - p_i[0]) * (p_k[1] - p_i[1]) - (p_j[1] - p_i[1]) * (p_k[0] - p_i[0])
                area = 0.5 * abs(cross)
                
                areas.append(area)
                indices.append((i, j, k, cross))
                
    areas = np.array(areas)
    min_area_val = np.min(areas)
    shifted = areas - min_area_val
    exp_terms = np.exp(-T * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return np.zeros_like(points)
        
    # We are minimizing f = -soft_min = 1/T * log(sum(exp(-T * A_i)))
    # df/dx = 1/T * (1/sum_exp) * sum( d/dx (exp(-T * A_i)) )
    # d/dx exp(-T * A_i) = exp(-T * A_i) * (-T) * dA_i/dx
    # So df/dx = 1/T * (1/sum_exp) * sum( exp(-T * A_i) * (-T) * dA_i/dx )
    # df/dx = - (1/sum_exp) * sum( exp(-T * A_i) * dA_i/dx )
    
    grad = np.zeros_like(points)
    
    for idx, (i, j, k, cross) in enumerate(indices):
        weight = exp_terms[idx] / sum_exp
        if weight < 1e-10:
            continue
            
        # dA_i/dx for point i, j, k
        # A = 0.5 * | cross |
        # If cross > 0, dA/dx = 0.5 * d(cross)/dx
        # If cross < 0, dA/dx = -0.5 * d(cross)/dx
        # sign = 1 if cross > 0 else -1
        sign = 1.0 if cross > 0 else -1.0
        factor = 0.5 * sign * weight
        
        # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # d(cross)/dx_i = -(y_k - y_i) - (-(y_j - y_i)) = -y_k + y_i + y_j - y_i = y_j - y_k
        # d(cross)/dy_i = -(x_k - x_i) - (-(x_j - x_i)) = -x_k + x_i + x_j - x_i = x_j - x_k
        
        # d(cross)/dx_j = (y_k - y_i)
        # d(cross)/dy_j = -(x_k - x_i)
        
        # d(cross)/dx_k = -(y_j - y_i)
        # d(cross)/dy_k = (x_j - x_i)
        
        # Note: These derivatives are for the signed cross product.
        # A = 0.5 * sign * cross.
        # dA/dx_i = 0.5 * sign * (y_j - y_k)
        # dA/dy_i = 0.5 * sign * (x_j - x_k)
        
        # dA/dx_j = 0.5 * sign * (y_k - y_i)
        # dA/dy_j = 0.5 * sign * (-(x_k - x_i)) = 0.5 * sign * (x_i - x_k)
        
        # dA/dx_k = 0.5 * sign * (-(y_j - y_i)) = 0.5 * sign * (y_i - y_j)
        # dA/dy_k = 0.5 * sign * (x_j - x_i)
        
        # The gradient of the objective f is -sum(weight * dA/dx)
        # So we subtract factor * dA/dx from grad?
        # df/dx = - sum( weight * dA/dx )
        
        grad[i][0] -= factor * (points[j][0] - points[k][0]) # Wait, y_j - y_k
        grad[i][1] -= factor * (points[j][1] - points[k][1])
        
        grad[j][0] -= factor * (points[k][0] - points[i][0]) # y_k - y_i
        grad[j][1] -= factor * (points[i][0] - points[k][0]) # x_i - x_k
        
        grad[k][0] -= factor * (points[i][0] - points[j][0]) # y_i - y_j
        grad[k][1] -= factor * (points[j][0] - points[i][0]) # x_j - x_i
        
    return grad

def _local_optimize_smooth(points, T_start=10.0, T_end=1000.0, steps=50):
    """
    Use L-BFGS-B on the smooth approximation, annealing T.
    """
    from scipy.optimize import minimize
    
    x = points.flatten()
    
    def obj(x):
        p = x.reshape(-1, 2)
        # Clamp to triangle during optimization? 
        # It's better to use bounds or penalty. 
        # For simplicity, we'll just clamp and hope L-BFGS-B stays close.
        # Or we can use a penalty.
        # Let's just clamp the input to the objective to ensure validity.
        p_clamped = np.array([_clamp_to_triangle(p) for p in p])
        return _soft_min_area(p_clamped, T)
        
    def grad(x):
        p = x.reshape(-1, 2)
        p_clamped = np.array([_clamp_to_triangle(p) for p in p])
        # Gradient of clamped points is tricky. 
        # If a point is on the boundary, the gradient of the clamp is 0 or undefined.
        # This might cause issues.
        # Alternative: Use bounds in L-BFGS-B? 
        # The triangle is not a box.
        # Let's assume the points stay inside or just barely cross and get clamped.
        # The gradient calculation on the clamped points might be wrong if the point is on the boundary.
        # For now, we return the gradient of the unclamped points, hoping the optimizer stays inside.
        # If it goes outside, the objective jumps, but L-BFGS-B might recover.
        g = _grad_soft_min_area(p_clamped, T)
        return g.flatten()
        
    best_x = x
    best_f = obj(x)
    
    for step in range(steps):
        # Anneal T
        T = T_start * (T_end / T_start) ** (step / max(1, steps - 1))
        
        res = minimize(obj, x, method='L-BFGS-B', jac=grad, options={'maxiter': 100, 'ftol': 1e-8})
        x = res.x
        f = obj(x)
        
        if f < best_f:
            best_f = f
            best_x = x
            
    return best_x.reshape(-1, 2)


def _generate_initial_config(rng, strategy='structured'):
    """Generate an initial configuration of 11 points."""
    if strategy == 'structured':
        # Use a structured grid-like approach in barycentric coordinates
        # Place points in a pattern that spreads them out
        points = []
        
        # Corners
        points.append(V0 + 0.02 * (V1 - V0))
        points.append(V1 - 0.02 * (V1 - V0))
        points.append(V2 + 0.02 * (V0 - V2))
        
        # Midpoints of edges
        points.append((V0 + V1) / 2.0)
        points.append((V1 + V2) / 2.0)
        points.append((V2 + V0) / 2.0)
        
        # Center
        points.append((V0 + V1 + V2) / 3.0)
        
        # Additional points
        points.append(V0 + 0.3 * (V1 - V0) + 0.1 * (V2 - V0))
        points.append(V0 + 0.1 * (V1 - V0) + 0.3 * (V2 - V0))
        points.append(V1 + 0.1 * (V2 - V1) + 0.1 * (V0 - V1))
        
        return np.array(points)
    elif strategy == 'random':
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                p = [u + 0.5 * v, H * v]
                points.append(p)
        return np.array(points)
    elif strategy == 'grid':
        # Create a grid in barycentric space
        points = []
        # Use a 3x3 grid in barycentric coordinates, picking 11 points
        grid_u = [0.05, 0.35, 0.65, 0.95]
        grid_v = [0.05, 0.35, 0.65, 0.95]
        candidates = []
        for u in grid_u:
            for v in grid_v:
                w = 1.0 - u - v
                if w >= -0.01:
                    w = max(w, 0.0)
                    candidates.append(_barycentric_to_cartesian(np.array([u, v, w])))
        
        # Select 11 points from candidates, trying to maximize spread
        rng.shuffle(candidates)
        for p in candidates:
            if len(points) >= N:
                break
            points.append(p)
        
        # If we don't have enough, add random points
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        
        return np.array(points)
    else:
        return _generate_initial_config(rng, 'random')


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Try multiple strategies and restarts
    strategies = ['structured', 'grid', 'random']
    
    # First, try the structured approach with local optimization
    for strategy in strategies:
        if time.time() > deadline - 30:
            break
        
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            
            points = _generate_initial_config(rng, strategy)
            
            # Local optimization using smooth approximation
            points_opt = _local_optimize_smooth(points, T_start=5.0, T_end=500.0, steps=20)
            score = _min_triangle_area_normalized(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
    
    # If we have time, do more fine-tuning on the best solution
    if best_points is not None and time.time() < deadline - 20:
        # Try to further optimize the best solution with more iterations
        points_fine = _local_optimize_smooth(best_points, T_start=50.0, T_end=1000.0, steps=30)
        score_fine = _min_triangle_area_normalized(points_fine)
        if score_fine > best_score:
            best_score = score_fine
            best_points = points_fine.copy()
    
    # Final check: ensure all points are inside the triangle
    if best_points is not None:
        best_points = np.array([_clamp_to_triangle(p) for p in best_points])
    
    return best_points


# EVOLVE-BLOCK-START
# (The construct function is defined above, but we keep this block for compatibility)
# EVOLVE-BLOCK-END
