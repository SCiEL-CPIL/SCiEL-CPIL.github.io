import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 14
TIME_LIMIT = 115.0  # Safety margin against 150s kill

def _vectorized_min_triangle_area(points):
    """
    Compute the minimum triangle area for a set of points using vectorization.
    points: (n, 2) array
    Returns: minimum area
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Vectorize triangle area calculation
    # We need to compute area for all (i, j, k) with i < j < k
    # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    
    # To save memory and time, we can iterate over i, but vectorize j and k
    # Or fully vectorize if n is small enough (n=14 is small)
    
    # Fully vectorized approach:
    # Create indices
    idx_i, idx_j, idx_k = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    
    # Filter for i < j < k
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    
    i_idx = idx_i[mask]
    j_idx = idx_j[mask]
    k_idx = idx_k[mask]
    
    if len(i_idx) == 0:
        return 0.0
        
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _score(points):
    """
    Compute the score: min_triangle_area / convex_hull_area
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
        min_area = _vectorized_min_triangle_area(points)
        return min_area / hull_area
    except Exception:
        return 0.0

def _support_to_points(coeffs):
    """
    Convert support function coefficients to points.
    coeffs[0] = a0 (base radius)
    coeffs[1] = a1
    coeffs[2] = a2
    coeffs[3] = a3
    coeffs[4] = a4
    coeffs[5] = a5
    """
    a0, a1, a2, a3, a4, a5 = coeffs
    
    # Ensure convexity: a0 must be large enough
    # A sufficient condition for h(theta) = a0 + sum a_m cos(2m theta) to be convex
    # is roughly a0 > sum |a_m| * (2m)^2 / 2 ? No, that's for curvature.
    # For a support function to define a convex body, h must be such that the boundary is convex.
    # A simple heuristic: a0 > sum(|a_m|) * 2m
    # Let's enforce a0 >= 0.5 + sum(|a_m| * 2m)
    min_a0 = 0.5 + (abs(a1)*2 + abs(a2)*4 + abs(a3)*6 + abs(a4)*8 + abs(a5)*10)
    if a0 < min_a0:
        a0 = min_a0
    
    # We sample 14 points evenly by angle
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # h(theta) = a0 + a1*cos(2theta) + a2*cos(4theta) + ...
    h = np.zeros(N)
    for m in range(1, 6):
        h += coeffs[m] * np.cos(2 * m * theta)
    h += a0
    
    # h'(theta) = sum -2m * a_m * sin(2m theta)
    hp = np.zeros(N)
    for m in range(1, 6):
        hp += -2 * m * coeffs[m] * np.sin(2 * m * theta)
        
    # Point on boundary:
    # x = h cos(theta) - hp sin(theta)
    # y = h sin(theta) + hp cos(theta)
    x = h * np.cos(theta) - hp * np.sin(theta)
    y = h * np.sin(theta) + hp * np.cos(theta)
    
    return np.column_stack([x, y])

def _objective(coeffs):
    """
    Objective function for optimization (minimization).
    Returns -score.
    """
    points = _support_to_points(coeffs)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    
    # Define bounds for the support function coefficients
    # a0 is the base radius. It should be positive and large enough.
    # a1..a5 are the harmonic coefficients.
    # We allow them to be in [-1, 1] to explore various shapes.
    # a0 in [0.5, 5.0]
    
    bounds = [
        (0.5, 5.0),   # a0
        (-1.0, 1.0),  # a1
        (-1.0, 1.0),  # a2
        (-1.0, 1.0),  # a3
        (-1.0, 1.0),  # a4
        (-1.0, 1.0)   # a5
    ]
    
    best_score = 0.0
    best_coeffs = None
    
    # Strategy: Differential Evolution
    # DE is a global optimizer. It maintains a population of candidate solutions.
    # This helps escape local optima better than Powell/Nelder-Mead.
    
    # We use a time-based stop condition or maxiter.
    # Let's run DE for a fixed number of generations, then refine.
    
    try:
        # Use differential evolution
        # popsize=20 (default is 15), maxiter=100 (might be too slow, let's check)
        # tol=1e-4
        # We need to be careful with time.
        
        # Let's run DE with a limited number of iterations first.
        # The objective function is vectorized, so it's fast.
        # Let's estimate: 14 points -> 364 triangles. Vectorized area calc is very fast.
        # ConvexHull is the bottleneck.
        
        # Let's run DE for 50 iterations.
        # If time allows, we can run more.
        
        # We'll use a callback to monitor time.
        def callback(xk, convergence):
            if time.time() - start_time > TIME_LIMIT - 10:
                return True
            return False
        
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=20, 
            tol=1e-4, 
            seed=42,
            callback=callback,
            polish=True  # Polish the result with L-BFGS-B or similar? 
                         # Actually, DE's polish uses minimize. 
                         # We might want to control this.
        )
        
        if res.success and res.fun < -best_score:
            best_score = -res.fun
            best_coeffs = res.x
            
    except Exception as e:
        pass
        
    # Refinement Phase
    # Use local optimization (Powell) to refine the best solution found by DE.
    if best_coeffs is not None and time.time() - start_time < TIME_LIMIT - 5:
        try:
            # Powell is robust for this kind of problem
            res_refine = minimize(
                _objective, 
                best_coeffs, 
                method='Powell', 
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            if res_refine.fun < -best_score:
                best_score = -res_refine.fun
                best_coeffs = res_refine.x
        except Exception:
            pass
            
    # If DE failed or didn't find anything, try a few random starts with Powell
    if best_coeffs is None or best_score < 1.0:
        rng = np.random.RandomState(123)
        for _ in range(10):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            # Random init
            c_init = np.array([
                rng.uniform(0.5, 2.0),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5)
            ])
            try:
                res = minimize(
                    _objective, 
                    c_init, 
                    method='Powell', 
                    options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
                )
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_coeffs = res.x
            except Exception:
                pass
                
    # Final fallback: Regular 14-gon
    if best_coeffs is None:
        # Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        # Normalize to unit area? 
        # The score is min_area / hull_area. 
        # Scaling doesn't change the score.
        # But the task says "normalized to unit area". 
        # Wait, the task says "Place 14 points in a planar convex region normalized to unit area".
        # And "score is the minimum triangle area divided by the convex hull area".
        # So the scale cancels out.
        return points
        
    points = _support_to_points(best_coeffs)
    
    # Ensure we return a valid array
    return points
