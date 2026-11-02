"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_score_from_gaps(gaps):
    """
    Compute the score (min triangle area / hull area) given a vector of gaps.
    gaps: (N-1,) array of positive numbers representing relative angle gaps.
    """
    # Normalize gaps to sum to 1, then scale to 2*pi
    total_gap = np.sum(gaps)
    if total_gap < 1e-10:
        return 0.0
    normalized_gaps = gaps / total_gap
    angles = np.zeros(N)
    angles[1:] = 2 * np.pi * np.cumsum(normalized_gaps[:N-1])
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Compute Convex Hull Area
    # Area of polygon on unit circle is 0.5 * sum(sin(theta_{i+1} - theta_i))
    # Since points are sorted by angle, we just sum the sine of the gaps
    hull_area = 0.5 * np.sum(np.sin(angles[1:] - angles[:-1]) + np.sin(2*np.pi - angles[-1]))
    # Note: angles[0] is 0. The last gap is 2*pi - angles[-1].
    # Actually, angles[1:] - angles[:-1] gives gaps 1 to N-2.
    # The gap from last point to first (0) is 2*pi - angles[-1].
    # The gap from 0 to first non-zero is angles[1] - 0 = angles[1].
    # My construction:
    # angles[0] = 0
    # angles[1] = 2*pi * g0
    # angles[2] = 2*pi * (g0+g1)
    # ...
    # angles[N-1] = 2*pi * (g0+...+g[N-2])
    # Gaps are:
    # g0: angles[1]-angles[0]
    # g1: angles[2]-angles[1]
    # ...
    # g[N-2]: angles[N-1]-angles[N-2]
    # g[N-1]: 2*pi - angles[N-1]
    
    # Let's re-calculate hull area properly from the generated angles
    # The vertices are in circular order.
    # Area = 0.5 * sum_{i=0}^{N-1} (x_i y_{i+1} - x_{i+1} y_i)
    # For unit circle, x_i y_{i+1} - x_{i+1} y_i = sin(theta_{i+1} - theta_i)
    # So Area = 0.5 * sum_{i=0}^{N-1} sin(theta_{i+1} - theta_i)
    
    thetas = np.append(angles, 2*np.pi)
    diffs = np.diff(thetas)
    # Ensure diffs are positive and < pi (which they are for N=14)
    hull_area = 0.5 * np.sum(np.sin(diffs))
    
    if hull_area <= 1e-12:
        return 0.0
        
    # Compute Min Triangle Area
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _objective(params):
    """
    Objective function for optimization.
    params: (N-1,) array of positive gap weights.
    """
    # Ensure positivity to avoid numerical issues in log or division
    # We just use the raw params, assuming they stay positive.
    # To be safe, we can map them to positive reals via exp, but Nelder-Mead
    # works in the original space. Let's just return 0 if any are <= 0.
    if np.any(params <= 0):
        return 100.0 # Large penalty
    
    score = _compute_score_from_gaps(params)
    return -score


def construct():
    """
    Construct N points in a planar convex region to maximize the minimum
    triangle area normalized by convex hull area.
    
    Returns: (N, 2) array of points
    """
    # Initial guess: equal gaps
    initial_gaps = np.ones(N - 1)
    
    # Optimize the gaps
    result = minimize(
        _objective,
        initial_gaps,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
    )
    
    optimal_gaps = result.x
    
    # Convert optimal gaps to angles and points
    total_gap = np.sum(optimal_gaps)
    normalized_gaps = optimal_gaps / total_gap
    angles = np.zeros(N)
    angles[1:] = 2 * np.pi * np.cumsum(normalized_gaps)
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
# EVOLVE-BLOCK-END
