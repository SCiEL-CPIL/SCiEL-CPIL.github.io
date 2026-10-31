import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _fourier_to_points(params):
    """
    Convert Fourier parameters to points.
    params: array of shape (2 * K + 1, )
    The first parameter is the base radius (offset from 1).
    The next 2*K parameters are coefficients for cos and sin of harmonics 1..K.
    
    r_k = base_r + sum_{m=1}^K (a_m * cos(m * theta_k) + b_m * sin(m * theta_k))
    x_k = r_k * cos(theta_k)
    y_k = r_k * sin(theta_k)
    """
    K = (len(params) - 1) // 2
    base_r = params[0]
    coeffs = params[1:]
    
    theta = 2 * np.pi * np.arange(N) / N
    
    r = np.full(N, base_r)
    
    for m in range(1, K + 1):
        a_m = coeffs[2*(m-1)]
        b_m = coeffs[2*(m-1) + 1]
        r += a_m * np.cos(m * theta) + b_m * np.sin(m * theta)
        
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack([x, y])

def _neg_fourier_score(params):
    """
    Objective for Fourier-parameterized optimization.
    """
    points = _fourier_to_points(params)
    s = _score(points)
    return -s

def _optimize_fourier(deadline, rng, K=3, n_restarts=10):
    """
    Optimize the Fourier coefficients to find a good configuration.
    """
    best_params = None
    best_score = -np.inf
    best_points = None
    
    # Parameters: 1 (base radius) + 2*K (harmonics)
    n_params = 1 + 2 * K
    
    time_per_restart = (deadline - time.time()) / (n_restarts + 2) # +2 for buffer and polish
    
    for i in range(n_restarts):
        if time.time() > deadline - 10: # Reserve 10s for final polish
            break
            
        # Initialize parameters
        # Start with regular polygon (base_r=1, others 0)
        # Add small random perturbations to escape local minima
        init_params = np.zeros(n_params)
        init_params[0] = 1.0
        # Small perturbations for harmonics
        # Higher harmonics should be smaller to maintain convexity/regularity
        for m in range(1, K + 1):
            scale = 0.05 / m
            init_params[2*m] = rng.normal(0, scale)
            init_params[2*m + 1] = rng.normal(0, scale)
            
        # Also try some structured initializations
        if i == 1:
            init_params[2] = 0.1 # First harmonic
        elif i == 2:
            init_params[4] = 0.1 # Second harmonic
            
        # Optimize
        try:
            res = minimize(
                _neg_fourier_score,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            cand_params = res.x
            cand_points = _fourier_to_points(cand_params)
            cand_score = _score(cand_points)
            
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand_params
                best_points = cand_points
                
        except Exception:
            pass
            
    return best_points, best_score

def _polish(points, deadline, rng):
    """
    Final polishing using Nelder-Mead on the raw coordinates.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Nelder-Mead on hard score
    if time.time() < deadline - 1:
        try:
            # Use a tighter tolerance for the final polish
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Fourier Optimization
    # Allocate most of the time here
    fourier_deadline = deadline - 30.0 # Reserve 30s for polishing
    
    if time.time() < fourier_deadline:
        pts, score = _optimize_fourier(fourier_deadline, rng, K=4, n_restarts=15)
        if pts is not None:
            best_points = pts
            best_score = score
            
    # If Fourier optimization failed or didn't run, use a regular polygon
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
        
    # Phase 2: Polishing
    # Use remaining time to polish the best found configuration
    polish_deadline = deadline - 2.0
    
    if time.time() < polish_deadline:
        polished_pts, polished_score = _polish(best_points, polish_deadline, rng)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
            
    # Final fallback safety check
    if best_points is None or np.isnan(best_points).any():
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
