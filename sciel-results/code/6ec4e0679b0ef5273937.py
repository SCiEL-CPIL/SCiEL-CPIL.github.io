import numpy as np
from scipy.optimize import minimize
import time

N = 14

def _calculate_score_from_angles(angles):
    """
    Calculate the score given a set of angles.
    The first angle is fixed at 0. The rest are optimized.
    Returns the score (min triangle area / hull area).
    """
    # angles is an array of shape (N-1,)
    # Full angles: 0, angles[0], ..., angles[N-2]
    all_angles = np.concatenate(([0.0], angles))
    
    # Sort angles to ensure convex order
    all_angles = np.sort(all_angles)
    
    # Ensure they are in [0, 2*pi)
    # Since we start with 0 and others are positive, sort is fine.
    # But we must handle the wrap-around if the last angle is close to 2*pi.
    # Actually, if we fix 0, and optimize others in (0, 2*pi), the polygon is closed.
    # The convex hull is the polygon formed by these points in sorted order.
    
    # Calculate points
    pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
    
    # Calculate Hull Area (Shoelace formula for sorted points on convex hull)
    # Since all points are on the circle and sorted, they are all on the hull.
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Shoelace formula
    # Sum over i of (x_i * y_{i+1} - x_{i+1} * y_i)
    # Note: for i=N-1, i+1 is 0
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area_sum = np.sum(x * y_next - x_next * y)
    hull_area = 0.5 * np.abs(area_sum)
    
    if hull_area < 1e-10:
        return 0.0
        
    # Calculate Min Triangle Area
    # We need min over all i < j < k of 0.5 * |det(p_j - p_i, p_k - p_i)|
    # Vectorized calculation for all triples is O(N^3). N=14, so 14*13*12/6 = 364 triples.
    # This is very fast.
    
    min_area = np.inf
    # We can just loop, it's small enough
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * np.abs(cross)
                
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def _objective(angles):
    """
    Objective function for minimization (negative score).
    Handles sorting and boundary conditions.
    """
    # Ensure angles are within (0, 2*pi)
    # The optimizer might push them out, but we sort anyway.
    # We don't strictly need to bound them if we sort, but it helps convergence.
    # Let's just sort.
    try:
        score = _calculate_score_from_angles(angles)
        if not np.isfinite(score):
            return 1.0
        return -score
    except Exception:
        return 1.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_score = -np.inf
    best_angles = None
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Regular polygon
    # Angles for regular N-gon: 0, 2pi/N, 4pi/N, ...
    # We fix 0, so we need the other N-1 angles.
    regular_angles = np.linspace(2*np.pi/N, 2*np.pi, N-1, endpoint=False)
    
    initial_guesses = [regular_angles]
    
    # Add some random perturbations
    for _ in range(20):
        perturbation = rng.normal(0, 0.1, N-1)
        guess = regular_angles + perturbation
        initial_guesses.append(guess)
    
    # Add some "clustered" guesses to explore different local minima
    for _ in range(10):
        guess = rng.uniform(0.1, 2*np.pi-0.1, N-1)
        initial_guesses.append(guess)
        
    for guess in initial_guesses:
        if time.time() > deadline - 5:
            break
            
        # Use Powell's method
        # Powell is good for this because it doesn't require gradients
        # and can handle the non-smoothness of the min operator.
        try:
            res = minimize(_objective, guess, method='Powell', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12, 'disp': False})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_angles = res.x.copy()
                
        except Exception:
            pass
            
    # If no good result, fallback to regular polygon
    if best_angles is None:
        best_angles = regular_angles
        best_score = _calculate_score_from_angles(best_angles)
        
    # Convert best angles to points
    all_angles = np.concatenate(([0.0], best_angles))
    all_angles = np.sort(all_angles)
    pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
    
    return pts
