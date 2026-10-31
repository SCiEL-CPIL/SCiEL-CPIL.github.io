import time
import numpy as np
from scipy.optimize import differential_evolution
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
_NUM_TRI = len(_TRI_INDICES)

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
        # ConvexHull requires at least 3 non-collinear points
        if len(points) < 3:
            return 0.0
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

def _objective(params):
    """
    Objective function for minimization (maximizing score).
    Returns -score.
    """
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _simulated_annealing(params, start_time, deadline, rng):
    """
    Simulated Annealing refinement.
    """
    current_params = params.copy()
    current_score = _score(current_params.reshape(N, 2))
    best_params = current_params.copy()
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling = 0.95
    
    while time.time() < deadline - 2.0:
        if T < 1e-6:
            break
            
        # Generate a neighbor
        # Scale step size with temperature
        step_size = T * 0.5
        noise = rng.normal(0, step_size, (N, 2))
        new_params = current_params + noise.flatten()
        new_points = new_params.reshape(N, 2)
        
        new_score = _score(new_points)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_params = new_params
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        T *= cooling
        
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    rng = np.random.RandomState(42)
    
    # Bounds for points: [-2, 2] to allow hull to grow/shrink, ratio is scale invariant
    bounds = [(-2.0, 2.0)] * (N * 2)
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    init_poly = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Phase 1: Differential Evolution
    # DE is stochastic, so we run it with a reasonable number of iterations.
    # maxiter=100, popsize=15 (default) -> 1500 evaluations.
    # Each eval is ~fast (vectorized).
    
    try:
        # Use a specific seed for reproducibility in DE if needed, but we have rng
        # differential_evolution doesn't take a rng directly in older scipy, but we can set seed
        # Let's just run it.
        res = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-5,
            init='sobol' if hasattr(differential_evolution, '__dict__') and 'sobol' in str(differential_evolution.__doc__) else 'latinhypercube',
            # Note: 'sobol' init might not be available in all scipy versions, falling back to latinhypercube
            # Actually, just use default or 'latinhypercube' which is standard
            init='latinhypercube'
        )
        
        if res.success or res.fun < -best_score:
            cand_points = res.x.reshape(N, 2)
            score = _score(cand_points)
            if score > best_score:
                best_score = score
                best_points = cand_points
    except Exception as e:
        # Fallback to initial polygon if DE fails
        best_points = init_poly.reshape(N, 2)
        best_score = _score(best_points)
    
    # If DE didn't run or found nothing, use the polygon
    if best_points is None:
        best_points = init_poly.reshape(N, 2)
        best_score = _score(best_points)
    
    # Phase 2: Simulated Annealing Refinement
    # Run SA starting from the DE result
    if time.time() < deadline - 5.0:
        try:
            sa_points, sa_score = _simulated_annealing(
                best_points.flatten(), 
                start_time, 
                deadline, 
                rng
            )
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points.reshape(N, 2)
        except Exception:
            pass
            
    # Phase 3: Additional SA restarts if time permits
    if time.time() < deadline - 10.0:
        # Try a few restarts with different temperatures or perturbations
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            # Perturb the best solution
            noise_scale = 0.1
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            try:
                sa_points, sa_score = _simulated_annealing(
                    perturbed.flatten(), 
                    start_time, 
                    deadline, 
                    rng
                )
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points.reshape(N, 2)
            except Exception:
                continue

    return best_points

