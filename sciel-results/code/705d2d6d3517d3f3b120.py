"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def _triangle_area_on_circle(angles, i, j, k):
    """Area of triangle formed by points i,j,k on unit circle."""
    a, b, c = angles[i], angles[j], angles[k]
    val = np.sin(b - a) + np.sin(c - b) + np.sin(a - c)
    return abs(val)

def _hull_area_on_circle(angles):
    """Convex hull area for points on unit circle, angles in [0, 2*pi)."""
    # Sort angles
    sorted_angles = np.sort(angles)
    # Add wrap-around
    diffs = np.diff(sorted_angles)
    diffs = np.append(diffs, 2 * np.pi - sorted_angles[-1] + sorted_angles[0])
    return 0.5 * np.sum(np.sin(diffs))

def _min_triangle_area_on_circle(angles):
    """Minimum triangle area for points on unit circle."""
    n = len(angles)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area_on_circle(angles, i, j, k)
                if area < min_area:
                    min_area = area
    return min_area

def _score_angles(angles):
    """Score = min triangle area / hull area for points on unit circle."""
    min_ta = _min_triangle_area_on_circle(angles)
    hull_a = _hull_area_on_circle(angles)
    if hull_a <= 1e-12:
        return -1.0
    return min_ta / hull_a

def _neg_score_angles(angles):
    return -_score_angles(angles)

def _precompute_indices(n):
    """Precompute all triple indices."""
    return list(itertools.combinations(range(n), 3))

def _min_triangle_area_fast(angles, triples):
    min_area = float('inf')
    for i, j, k in triples:
        a, b, c = angles[i], angles[j], angles[k]
        val = np.sin(b - a) + np.sin(c - b) + np.sin(a - c)
        area = abs(val)
        if area < min_area:
            min_area = area
    return min_area

def _neg_score_fast(angles, triples):
    min_ta = _min_triangle_area_fast(angles, triples)
    hull_a = _hull_area_on_circle(angles)
    if hull_a <= 1e-12:
        return 1.0
    return -min_ta / hull_a

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = N
    triples = _precompute_indices(n)
    
    best_score = -1.0
    best_angles = None
    
    # Regular polygon as initial guess
    regular = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = _score_angles(regular)
    if s > best_score:
        best_score = s
        best_angles = regular.copy()
    
    # Try multiple random restarts
    np.random.seed(42)
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation of regular polygon
        if r == 0:
            angles = regular.copy()
        else:
            # Mix of different strategies
            if r % 3 == 0:
                # Small random perturbation
                angles = regular + np.random.normal(0, 0.1, n)
            elif r % 3 == 1:
                # Larger perturbation
                angles = regular + np.random.normal(0, 0.3, n)
            else:
                # Random angles (will be sorted effectively)
                angles = np.sort(np.random.uniform(0, 2 * np.pi, n))
        
        # Fix rotational symmetry: set first angle to 0
        angles = angles - angles[0]
        angles = np.sort(angles)
        # Ensure in [0, 2*pi)
        angles = angles % (2 * np.pi)
        angles = np.sort(angles)
        
        # Local optimization
        try:
            res = minimize(
                _neg_score_fast,
                angles,
                args=(triples,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() > deadline - 3.0:
                break
            
            opt_angles = res.x
            opt_angles = np.sort(opt_angles)
            opt_score = _score_angles(opt_angles)
            
            if opt_score > best_score:
                best_score = opt_score
                best_angles = opt_angles.copy()
        except Exception:
            continue
    
    # If no valid angles found, use regular polygon
    if best_angles is None:
        best_angles = regular
    
    # Convert angles to 2D points on unit circle
    points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return points
# EVOLVE-BLOCK-END
