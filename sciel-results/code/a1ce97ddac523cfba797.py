import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate all triangle areas
    min_area = float('inf')
    
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                # Area is 0.5 * |cross product|
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _points_from_angles(angles):
    """
    Convert a set of angles to points on the unit circle.
    Sorts angles to ensure convex ordering.
    """
    angles = np.sort(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def _objective(angles):
    """
    Objective function for differential evolution.
    Minimizes the negative score.
    """
    points = _points_from_angles(angles)
    
    # Ensure points are distinct enough
    # If any two points are too close, penalize heavily
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                return 1.0 # High penalty (since we minimize -score, high value is bad)

    score = _compute_score(points)
    return -score

def _local_refinement(angles):
    """
    Perform local refinement using Nelder-Mead on the angles.
    """
    res = minimize(
        _objective,
        angles,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    return res.x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial best: regular polygon
    initial_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_points = _points_from_angles(initial_angles)
    best_score = _compute_score(initial_points)
    best_angles = initial_angles.copy()
    
    # Define bounds for angles: [0, 2*pi]
    # Differential evolution handles the periodicity implicitly if we just sort,
    # but it's better to let it search in a wide range and sort.
    # However, DE works best with bounded spaces.
    # Since we sort the angles, the order doesn't matter for the geometry, only the set.
    # We can constrain angles to [0, 2*pi].
    bounds = [(0, 2*np.pi) for _ in range(N)]
    
    # Run Differential Evolution
    # maxiter is the number of generations
    # popsize is the population size
    # tol is the relative tolerance
    # disp controls output
    # init is the initialization method
    # seed is for reproducibility
    
    # We have ~140 seconds.
    # DE can be slow. Let's try a moderate population size and iterations.
    # N=13, so 13 dimensions.
    # popsize=15 -> 195 points per generation.
    # maxiter=100 -> 19500 evaluations.
    # Each evaluation is fast (286 triangles).
    
    try:
        if time.time() < deadline - 20.0:
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=20,
                tol=1e-6,
                seed=42,
                polish=False, # We will do our own polish
                disp=False,
                init='sobol' # Use Sobol for better coverage in high dims
            )
            
            if de_result.success or de_result.fun < -best_score:
                candidate_angles = de_result.x
                candidate_points = _points_from_angles(candidate_angles)
                candidate_score = _compute_score(candidate_points)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_angles = candidate_angles.copy()
                    
    except Exception as e:
        pass # Fallback to initial if DE fails
    
    # Local Refinement
    if time.time() < deadline - 10.0:
        refined_angles = _local_refinement(best_angles)
        refined_points = _points_from_angles(refined_angles)
        refined_score = _compute_score(refined_points)
        
        if refined_score > best_score:
            best_score = refined_score
            best_angles = refined_angles.copy()
            
    # Try a few random restarts if time permits
    if time.time() < deadline - 30.0:
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            random_angles = np.random.uniform(0, 2*np.pi, N)
            # Local refinement from random start
            ref_angles = _local_refinement(random_angles)
            ref_points = _points_from_angles(ref_angles)
            ref_score = _compute_score(ref_points)
            
            if ref_score > best_score:
                best_score = ref_score
                best_angles = ref_angles.copy()

    final_points = _points_from_angles(best_angles)
    return final_points
