"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute triple indices for vectorized area computation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """Compute all triangle areas vectorized."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute score: min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_value(values, temperature):
    """Compute soft minimum using log-sum-exp trick for numerical stability."""
    # Use the standard softmin: -T * log(sum(exp(-values/T)))
    # For stability: let m = min(values), then softmin = m - T * log(sum(exp(-(values - m)/T)))
    m = np.min(values)
    exp_vals = np.exp(-(values - m) / temperature)
    return m - temperature * np.log(np.sum(exp_vals))

def _soft_score(points, temperature):
    """Compute soft score: soft_min(triangle areas) / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    soft_min = _soft_min_value(areas, temperature)
    return soft_min / hull_area

def _neg_soft_score(params, temperature):
    """Negative soft score for minimization."""
    points = _generate_points_from_params(params)
    score = _soft_score(points, temperature)
    if score <= 0:
        return 1.0
    return -score

def _neg_score(params):
    """Negative exact score for minimization."""
    points = _generate_points_from_params(params)
    score = _score(points)
    if score <= 0:
        return 1.0
    return -score

def _generate_points_from_params(params):
    """
    Generate 13 points from parameters.
    Params:
    - 13 angular deviations (delta_i) from regular 13-gon
    - 2 scaling factors (sx, sy)
    - 1 rotation angle (phi)
    """
    n = 13
    deltas = params[:n]
    sx = params[n]
    sy = params[n+1]
    phi = params[n+2]
    
    # Base angles for regular 13-gon
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = base_angles + deltas
    
    # Points on unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    # Apply scaling
    points[:, 0] *= sx
    points[:, 1] *= sy
    
    # Apply rotation
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    R = np.array([[cos_phi, -sin_phi],
                  [sin_phi, cos_phi]])
    
    points = points @ R.T
    
    return points

def _seed_regular():
    """Regular 13-gon."""
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(base_angles)
    y = np.sin(base_angles)
    points = np.column_stack((x, y))
    # Convert to params: deltas=0, sx=1, sy=1, phi=0
    params = np.zeros(N + 3)
    return params

def _seed_two_ring():
    """Two rings: inner and outer."""
    # 7 outer, 6 inner
    n_outer = 7
    n_inner = 6
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    
    outer_points = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    inner_points = np.column_stack((0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)))
    
    points = np.vstack([outer_points, inner_points])
    
    # This doesn't fit the radial model well, so we'll just return a default
    # The DE will handle exploration
    return np.zeros(N + 3)

def _seed_three_ring():
    """Three rings."""
    return np.zeros(N + 3)

def _seed_quasi_regular():
    """Quasi-regular with small perturbations."""
    rng = np.random.RandomState(42)
    deltas = rng.normal(0, 0.05, N)
    params = np.zeros(N + 3)
    params[:N] = deltas
    return params

def _annealed_soft_minimize(params_init, temperature_start, temperature_end, n_steps):
    """Anneal the temperature and minimize soft score."""
    current_params = params_init.copy()
    temps = np.geomspace(temperature_start, temperature_end, n_steps)
    
    for i, T in enumerate(temps):
        # Use L-BFGS-B for local optimization
        try:
            result = minimize(
                _neg_soft_score,
                current_params,
                args=(T,),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            if result.fun < _neg_soft_score(current_params, T):
                current_params = result.x
        except Exception:
            pass
    
    return current_params

def construct():
    """Construct 13 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 13
    # Define bounds for parameters
    angle_bounds = [(-np.pi/3, np.pi/3) for _ in range(n)]
    scale_bounds = [(0.1, 5.0), (0.1, 5.0)]
    rotation_bounds = [(0, np.pi/2)]
    
    bounds = angle_bounds + scale_bounds + rotation_bounds
    
    best_params = None
    best_score = -np.inf
    
    # Strategy 1: Differential Evolution with soft objective
    try:
        # Use a moderate temperature for soft objective
        T_de = 0.01
        
        def _de_objective(params):
            return _neg_soft_score(params, T_de)
        
        result = differential_evolution(
            _de_objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-5,
            polish=False,
            init='latinhypercube'
        )
        
        if result.x is not None:
            # Refine with annealing
            refined_params = _annealed_soft_minimize(
                result.x,
                temperature_start=0.01,
                temperature_end=0.001,
                n_steps=5
            )
            
            score1 = _score(_generate_points_from_params(refined_params))
            if score1 > best_score:
                best_score = score1
                best_params = refined_params.copy()
                
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        if best_params is not None:
            best_points = _generate_points_from_params(best_params)
            return best_points
        else:
            return _generate_points_from_params(_seed_regular())
    
    # Strategy 2: Start from multiple seeds with annealing
    seeds = [_seed_regular(), _seed_quasi_regular(), np.zeros(N + 3)]
    
    for seed_params in seeds:
        if time.time() > deadline - 20:
            break
        
        try:
            refined = _annealed_soft_minimize(
                seed_params,
                temperature_start=0.05,
                temperature_end=0.001,
                n_steps=8
            )
            
            score = _score(_generate_points_from_params(refined))
            if score > best_score:
                best_score = score
                best_params = refined.copy()
                
        except Exception:
            pass
    
    # Strategy 3: If we have time, do more DE restarts
    if time.time() < deadline - 40 and best_params is not None:
        try:
            # Perturb best params and run more DE
            rng = np.random.RandomState(123)
            perturbed = best_params + rng.normal(0, 0.1, N + 3)
            
            # Clip to bounds
            for i in range(N):
                perturbed[i] = np.clip(perturbed[i], -np.pi/3, np.pi/3)
            perturbed[N] = np.clip(perturbed[N], 0.1, 5.0)
            perturbed[N+1] = np.clip(perturbed[N+1], 0.1, 5.0)
            perturbed[N+2] = np.clip(perturbed[N+2], 0, np.pi/2)
            
            T_de2 = 0.005
            
            def _de_objective2(params):
                return _neg_soft_score(params, T_de2)
            
            result2 = differential_evolution(
                _de_objective2,
                bounds=bounds,
                seed=123,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                polish=False,
                init='latinhypercube'
            )
            
            if result2.x is not None:
                refined2 = _annealed_soft_minimize(
                    result2.x,
                    temperature_start=0.005,
                    temperature_end=0.0005,
                    n_steps=3
                )
                
                score2 = _score(_generate_points_from_params(refined2))
                if score2 > best_score:
                    best_score = score2
                    best_params = refined2.copy()
                    
        except Exception:
            pass
    
    # Final result
    if best_params is not None:
        best_points = _generate_points_from_params(best_params)
    else:
        best_points = _generate_points_from_params(_seed_regular())
    
    return best_points
