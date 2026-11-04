import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13
# Precompute triangle indices
TRIPLES = np.array(list(itertools.combinations(range(N), 3)))
NUM_TRIPLES = len(TRIPLES)

def _hull_area(points):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10 # Avoid division by zero, effectively 0 score

def _min_triangle_area(points):
    """Compute minimum triangle area."""
    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _score(points):
    """Compute the actual score: min area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _radial_to_points(radii, base_angles):
    """Convert radial offsets to Cartesian coordinates.
    base_angles: fixed angles of the regular N-gon.
    radii: array of N radii.
    """
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    return np.column_stack((x, y))

def _soft_min_score(radii, base_angles, temp=0.01):
    """
    Smooth surrogate for the score.
    We want to maximize min_area / hull_area.
    Equivalent to minimizing -log(min_area) + log(hull_area).
    Using log-sum-exp for min_area:
    log(min(areas)) approx -log(sum(exp(-areas/temp))) + log(temp) ... wait.
    
    Let's use the standard soft-min:
    soft_min(areas) = -temp * log(sum(exp(-areas/temp)))
    As temp -> 0, this approaches min(areas).
    
    Objective to minimize: -soft_min(areas) + log(hull_area)
    Or simply: -soft_min(areas) / hull_area? No, division is non-smooth if hull changes sign (it doesn't) but gradient is complex.
    
    Better: Minimize f(r) = -soft_min_area(r) + log(hull_area(r))
    This is equivalent to maximizing exp(-f) = exp(soft_min) / hull_area, which is a smooth approximation to min_area / hull_area.
    """
    points = _radial_to_points(radii, base_angles)
    
    # Calculate areas
    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    # To avoid overflow/underflow, shift by max
    max_area = np.max(areas)
    # soft_min = max_area - temp * log(sum(exp(-(areas - max_area)/temp)))
    # Note: min(a) = max - temp*log(sum(exp(-(a-max)/temp)))
    exp_terms = np.exp(-(areas - max_area) / temp)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_area = max_area - temp * log_sum
    
    # Hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10 # Large penalty
    
    # Objective: -soft_min_area + log(hull_area)
    # Minimizing this maximizes soft_min_area / hull_area (roughly)
    return -soft_min_area + np.log(hull_area)

def _soft_min_grad(radii, base_angles, temp=0.01):
    """
    Numerical gradient is safer and fast enough for N=13 parameters.
    Central difference.
    """
    grad = np.zeros_like(radii)
    h = 1e-5
    for i in range(len(radii)):
        radii_plus = radii.copy()
        radii_plus[i] += h
        radii_minus = radii.copy()
        radii_minus[i] -= h
        
        f_plus = _soft_min_score(radii_plus, base_angles, temp)
        f_minus = _soft_min_score(radii_minus, base_angles, temp)
        
        grad[i] = (f_plus - f_minus) / (2 * h)
        
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Base angles for regular 13-gon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Initial radii: 1.0 (regular polygon)
    # We can jitter them slightly to break symmetry if needed, but L-BFGS should handle it.
    # Let's try starting with uniform radii.
    best_radii = np.ones(N)
    best_score = _score(_radial_to_points(best_radii, base_angles))
    
    # Optimization parameters
    # Bounds for radii: [0.5, 1.5] to keep points in a reasonable range and ensure convexity (mostly)
    # If radii vary too much, the polygon might self-intersect or become non-convex.
    # For N=13, small variations maintain convexity.
    bounds = [(0.5, 1.5) for _ in range(N)]
    
    # Multiple restarts with different temperatures or initial jitters
    # Strategy: 
    # 1. Try regular polygon.
    # 2. Try small random jitters in radii.
    
    num_restarts = 5
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < num_restarts:
        # Generate initial radii
        if restarts_done == 0:
            radii_init = np.ones(N)
        else:
            # Small random jitter
            radii_init = np.ones(N) + np.random.normal(0, 0.05, N)
            # Clip to bounds
            radii_init = np.clip(radii_init, 0.5, 1.5)
        
        # Annealing temperature for soft-min
        # Start with higher temp (smoother), reduce it? 
        # L-BFGS works best with smooth functions.
        # Let's try a few temperatures.
        temps = [0.05, 0.02, 0.01, 0.005]
        
        for temp in temps:
            if time.time() > deadline:
                break
            
            # Run L-BFGS-B
            try:
                result = minimize(
                    _soft_min_score,
                    radii_init,
                    args=(base_angles, temp),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success:
                    radii_opt = result.x
                    # Calculate actual score
                    points_opt = _radial_to_points(radii_opt, base_angles)
                    score_opt = _score(points_opt)
                    
                    if score_opt > best_score:
                        best_score = score_opt
                        best_radii = radii_opt.copy()
                        
            except Exception:
                pass
        
        restarts_done += 1
        
        # If we have plenty of time, do more restarts with larger jitters
        if time.time() < deadline - 20:
             num_restarts += 5 # Allow more restarts if fast

    # Final polish: Nelder-Mead on the actual (non-smooth) score using the best radii found
    # This helps escape the soft-min approximation error near the optimum.
    if time.time() < deadline - 5:
        try:
            # Use Nelder-Mead on the 13-dim space
            # Objective: -score(points)
            def neg_score(radii):
                points = _radial_to_points(radii, base_angles)
                return -_score(points)
            
            res_nm = minimize(
                neg_score,
                best_radii,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res_nm.success:
                radii_nm = res_nm.x
                points_nm = _radial_to_points(radii_nm, base_angles)
                score_nm = _score(points_nm)
                
                if score_nm > best_score:
                    best_score = score_nm
                    best_radii = radii_nm.copy()
        except Exception:
            pass

    # Convert best radii to points
    final_points = _radial_to_points(best_radii, base_angles)
    
    return final_points

if __name__ == "__main__":
    pts = construct()
    score = _score(pts)
    print(f"Score: {score}")
