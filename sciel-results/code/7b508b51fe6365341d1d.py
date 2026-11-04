import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 11
S3 = np.sqrt(3)
H = S3 / 2  # height of unit-base equilateral triangle
A_AREA = 0.5 * 1 * H  # area of the big triangle = sqrt(3)/4

def _inside_triangle(pts):
    """Check if all points are inside the equilateral triangle."""
    for x, y in pts:
        if y < -1e-12 or y > H + 1e-12:
            return False
        if x < -1e-12 or x > 1 + 1e-12:
            return False
        # Left edge: y <= sqrt(3)*x
        if y > S3 * x + 1e-12:
            return False
        # Right edge: y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)
        if y > S3 * (1 - x) + 1e-12:
            return False
    return True

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    min_a = float('inf')
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
    return min_a

def _soft_min_objective(x, beta=100.0):
    """
    Negative soft-min of triangle areas using log-sum-exp trick.
    We want to maximize min area, so minimize -min area.
    Use -log(sum(exp(-beta * area_i))) / beta as a smooth approximation of -min(area_i).
    """
    pts = x.reshape(-1, 2)
    areas = []
    n = N
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                area = 0.5 * abs(cross)
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-beta * a))) / beta approximates min(a)
    # We want to maximize min, so minimize -soft_min
    # -soft_min = log(sum(exp(-beta*a))) / beta
    # To avoid overflow, use log-sum-exp trick
    min_idx = np.argmin(areas)
    shifted = -beta * (areas - areas[min_idx])
    log_sum = areas[min_idx] * beta + np.log(np.sum(np.exp(shifted)))
    soft_min_val = log_sum / beta
    return -soft_min_val  # negative because we minimize

def _project_to_triangle(pts):
    """Project points to be inside the triangle."""
    projected = []
    for x, y in pts:
        # Clamp to bounding box first
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, H)
        # Check left edge constraint: y <= sqrt(3)*x
        if y > S3 * x:
            # Project onto left edge: y = sqrt(3)*x
            # Move horizontally or vertically? Project perpendicularly is complex.
            # Simple: reduce y to sqrt(3)*x
            y = S3 * x
        # Check right edge constraint: y <= sqrt(3)*(1-x)
        if y > S3 * (1 - x):
            y = S3 * (1 - x)
        # After adjustments, re-check
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        projected.append([x, y])
    return np.array(projected)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s, 10s safety margin
    
    best_pts = None
    best_score = -float('inf')
    
    # Constraints for scipy: keep points inside triangle
    # We'll use penalty in objective instead for simplicity
    
    def penalty_objective(x, beta=50.0, pen_weight=1000.0):
        pts = x.reshape(-1, 2)
        # Check if points are inside triangle
        penalty = 0.0
        for px, py in pts:
            if py < -1e-9:
                penalty += pen_weight * (-py) ** 2
            if py > H + 1e-9:
                penalty += pen_weight * (py - H) ** 2
            if px < -1e-9:
                penalty += pen_weight * (-px) ** 2
            if px > 1 + 1e-9:
                penalty += pen_weight * (px - 1) ** 2
            if py > S3 * px + 1e-9:
                penalty += pen_weight * (py - S3 * px) ** 2
            if py > S3 * (1 - px) + 1e-9:
                penalty += pen_weight * (py - S3 * (1 - px)) ** 2
        obj = _soft_min_objective(x, beta=beta)
        return obj + penalty
    
    # Multiple random restarts
    np.random.seed(42)
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization inside triangle
        pts = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = H * v
            pts.append([x, y])
        x0 = np.array(pts).flatten()
        
        # Optimize
        try:
            result = minimize(
                penalty_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if result.success or result.fun < best_score:
                pts_opt = result.x.reshape(-1, 2)
                # Project to ensure validity
                pts_opt = _project_to_triangle(pts_opt)
                if _inside_triangle(pts_opt):
                    score = _min_triangle_area(pts_opt)
                    if score > best_score:
                        best_score = score
                        best_pts = pts_opt.copy()
        except Exception:
            continue
        
        # Also try a simpler local search with direct min
        if time.time() > deadline:
            break
        
        # Final polish: use exact min with small perturbations
        if best_pts is not None:
            if time.time() < deadline - 5.0:
                # Try to improve with small random perturbations
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    perturbed = best_pts + np.random.randn(N, 2) * 0.01
                    perturbed = _project_to_triangle(perturbed)
                    if _inside_triangle(perturbed):
                        s = _min_triangle_area(perturbed)
                        if s > best_score:
                            best_score = s
                            best_pts = perturbed.copy()
    
    if best_pts is None:
        # Fallback: random points
        pts = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = H * v
            pts.append([x, y])
        best_pts = np.array(pts)
    
    return best_pts
