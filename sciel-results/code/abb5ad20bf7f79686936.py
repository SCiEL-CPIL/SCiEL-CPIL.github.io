# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area computation
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
                if min_area == 0:
                    return 0.0
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _generate_fourier_points(n: int, radius: float, center: np.ndarray, 
                              amplitudes: np.ndarray, phases: np.ndarray,
                              frequencies: list) -> np.ndarray:
    """Generate points on a perturbed regular n-gon using Fourier modes."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.zeros((n, 2))
    
    for i in range(n):
        # Base radius with Fourier perturbations
        r = radius
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            r += amp * np.sin(freq * angles[i] + phase)
        
        points[i, 0] = center[0] + r * np.cos(angles[i])
        points[i, 1] = center[1] + r * np.sin(angles[i])
    
    return points


def _local_search_fourier(n: int, deadline: float) -> np.ndarray:
    """Local search in Fourier parameter space."""
    # Parameters: radius, center_x, center_y, and Fourier amplitudes/phases
    # Frequencies to use: 1, 2, 3 (low-order harmonics)
    frequencies = [1, 2, 3]
    n_params = 3 + 2 * len(frequencies)  # radius, cx, cy, + amp/phase per freq
    
    best_points = None
    best_score = 0.0
    
    # Initial parameters: regular 13-gon
    init_params = np.array([0.45, 0.5, 0.5] + [0.0] * (2 * len(frequencies)))
    
    # Coarse grid search over key parameters
    radius_range = np.linspace(0.3, 0.5, 10)
    cx_range = np.linspace(0.35, 0.65, 7)
    cy_range = np.linspace(0.35, 0.65, 7)
    
    # First pass: grid search over radius, cx, cy with zero amplitudes
    for r in radius_range:
        for cx in cx_range:
            for cy in cy_range:
                if time.time() > deadline - 60:
                    break
                
                params = np.array([r, cx, cy] + [0.0] * (2 * len(frequencies)))
                points = _generate_fourier_points(n, params[0], 
                                                  np.array([params[1], params[2]]),
                                                  params[3:6], params[6:9],
                                                  frequencies)
                
                score = _compute_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_params = params.copy()
    
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([0.5 + 0.45 * np.cos(angles),
                                       0.5 + 0.45 * np.sin(angles)])
        best_params = np.array([0.45, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Second pass: local search in Fourier parameter space
    current_params = best_params.copy()
    current_score = best_score
    step_size = 0.02
    learning_rate = 0.95
    
    for iteration in range(200):
        if time.time() > deadline - 30:
            break
        
        improved = False
        for p_idx in range(n_params):
            for direction in [-1, 1]:
                candidate_params = current_params.copy()
                candidate_params[p_idx] += direction * step_size
                
                # Constraints: radius in [0.2, 0.6], center in [0.2, 0.8], amplitudes in [-0.15, 0.15]
                if p_idx == 0:
                    if not (0.2 <= candidate_params[0] <= 0.6):
                        continue
                elif p_idx in [1, 2]:
                    if not (0.2 <= candidate_params[p_idx] <= 0.8):
                        continue
                elif p_idx >= 3:
                    if not (-0.15 <= candidate_params[p_idx] <= 0.15):
                        continue
                
                points = _generate_fourier_points(n, candidate_params[0],
                                                  np.array([candidate_params[1], candidate_params[2]]),
                                                  candidate_params[3:6], candidate_params[6:9],
                                                  frequencies)
                
                score = _compute_score(points)
                if score > current_score:
                    current_params = candidate_params
                    current_score = score
                    best_score = score
                    best_points = points.copy()
                    improved = True
        
        if not improved:
            step_size *= learning_rate
            if step_size < 1e-5:
                break
        else:
            step_size = max(step_size, 1e-5)
    
    return best_points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 600
    
    # Use Fourier parametrization approach
    points = _local_search_fourier(n, deadline)
    
    # Validate
    if points is None or points.shape != (n, 2) or not np.all(np.isfinite(points)):
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([0.5 + 0.45 * np.cos(angles),
                                  0.5 + 0.45 * np.sin(angles)])
    
    return points


# EVOLVE-BLOCK-END
