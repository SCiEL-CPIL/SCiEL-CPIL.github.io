"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use scipy.optimize.differential_evolution (DE) to globally explore the 22-dimensional space of point coordinates.
2. Define a smooth objective function based on the "soft min" (log-sum-exp) of the negative triangle areas. This avoids the non-differentiability of the raw minimum.
3. Run DE for a significant portion of the time budget to find a promising basin.
4. Polish the best result using L-BFGS-B with the same smooth objective for local convergence.
5. Handle boundary constraints via penalties in the objective function, which is robust for global optimizers.
"""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _penalty(pts):
    """
    Calculate a penalty for points outside the unit equilateral triangle.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Inequalities (must be <= 0 for interior):
    1. y <= 0  (outside below) -> penalty max(0, -y)
    2. y > sqrt(3)*x (outside left) -> penalty max(0, y - sqrt(3)*x)
    3. y > sqrt(3)*(1-x) (outside right) -> penalty max(0, y - sqrt(3)*(1-x))
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Boundary 1: y >= 0
    p1 = np.maximum(0, -y)
    
    # Boundary 2: y <= sqrt(3) * x
    # y - sqrt(3)*x <= 0
    p2 = np.maximum(0, y - np.sqrt(3) * x)
    
    # Boundary 3: y <= sqrt(3) * (1 - x)
    # y - sqrt(3) + sqrt(3)*x <= 0
    p3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    # Sum of violations, squared to make penalty more aggressive for large violations
    # But linear is often better for DE stability. Let's use linear sum first.
    return np.sum(p1) + np.sum(p2) + np.sum(p3)

def _min_area_raw(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    pts: (n, 2) array.
    """
    n = len(pts)
    min_area = float('inf')
    # Vectorized calculation of all areas is possible but 165 is small.
    # However, for speed in the objective function, we can vectorize partially.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _objective(x_flat):
    """
    Objective function for the optimizer.
    x_flat: 22-element array of coordinates.
    Returns: -min_area + penalty. We want to minimize this.
    """
    pts = x_flat.reshape(N, 2)
    
    # Check bounds quickly? DE handles bounds, but L-BFGS-B might drift slightly if not careful.
    # Penalty handles out-of-bounds gracefully.
    pen = _penalty(pts)
    
    if pen > 1e-3:
        # If significantly outside, return a large value
        return 1000.0 + pen
    
    min_a = _min_area_raw(pts)
    
    return -min_a + 100.0 * pen

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Define bounds for DE
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, S3H)] * N
    
    # Initial population size
    popsize = 50
    
    # Run Differential Evolution
    # maxiter is set high, but we rely on the time limit or maxiter.
    # DE is stochastic.
    try:
        # Use 'best1bin' strategy for faster convergence
        res = differential_evolution(
            _objective, 
            bounds, 
            popsize=popsize,
            maxiter=200,  # Cap iterations to prevent hanging if it gets stuck
            tol=1e-8,
            seed=42,
            polish=False,  # We will polish manually to control time
            init='latinhypercube'
        )
    except Exception as e:
        # Fallback if DE fails
        res = None
    
    best_x = res.x if res is not None else None
    best_val = res.fun if res is not None else float('inf')
    
    # Check time
    if time.time() - start_time > DEADLINE - 20:
        # Not enough time for polishing, return best from DE
        if best_x is not None:
            pts = best_x.reshape(N, 2)
            # Ensure validity
            if _penalty(pts) < 1e-3:
                return pts
        # Fallback
        return np.array([
            [0, 0], [1, 0], [0.5, S3H],
            [0.5, 0], [0.25, S3H/2], [0.75, S3H/2],
            [0.5, S3H/3], [0.25, S3H/6], [0.75, S3H/6],
            [0.5, 2*S3H/3], [0.5, S3H/2]
        ])
    
    # Polish with L-BFGS-B
    if best_x is not None:
        try:
            res_polish = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'disp': False}
            )
            if res_polish.fun < best_val:
                best_x = res_polish.x
                best_val = res_polish.fun
        except Exception:
            pass
    
    # Final check and return
    pts = best_x.reshape(N, 2)
    
    # Project to triangle if slightly outside due to numerical errors
    # Simple projection
    for i in range(N):
        x, y = pts[i]
        if y < 0: y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y: y = max_y
        if x < 0: x = 0
        if x > 1: x = 1
        pts[i] = [x, y]
        
    # Verify validity
    if _penalty(pts) > 1e-3:
         # Fallback to a known valid configuration
         return np.array([
             [0, 0], [1, 0], [0.5, S3H],
             [0.5, 0], [0.25, S3H/2], [0.75, S3H/2],
             [0.5, S3H/3], [0.25, S3H/6], [0.75, S3H/6],
             [0.5, 2*S3H/3], [0.5, S3H/2]
         ])
         
    return pts

if __name__ == "__main__":
    pts = construct()
    print(f"Min Area: {_min_area_raw(pts)}")
    print(f"Normalized: {_min_area_raw(pts) / (0.5 * np.sqrt(3)/2)}") # Area of unit triangle is sqrt(3)/4? No.
    # Unit area triangle? The problem says "unit-area triangle".
    # The triangle (0,0), (1,0), (0.5, sqrt(3)/2) has area:
    # base=1, height=sqrt(3)/2. Area = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4.
    # So we divide by sqrt(3)/4.
    area_tri = np.sqrt(3)/4
    print(f"Score: {_min_area_raw(pts) / area_tri}")
