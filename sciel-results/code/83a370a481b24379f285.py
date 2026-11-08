"""Heilbronn triangles: maximize minimum triangle area among 11 points in unit equilateral triangle."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

# Precompute triangle index combinations
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])
_n_combos = len(_combos)

def _barycentric_to_cartesian(params):
    """Convert barycentric params (u, v) for each point to Cartesian (x, y).
    Point = u * A + v * B + (1-u-v) * C where A=(0,0), B=(1,0), C=(0.5, H)
    So x = u + 0.5*v, y = H*v
    """
    u = params[0::2]
    v = params[1::2]
    x = u + 0.5 * v
    y = H * v
    return np.column_stack([x, y])

def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _soft_min_objective(params, beta):
    """Soft-min surrogate: -log(sum(exp(-beta * area_i))) / beta approximates -min(area_i).
    We minimize the negative, so maximize the soft-min.
    Returns the negative of the soft-min (to minimize).
    """
    points = _barycentric_to_cartesian(params)
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    # Soft minimum: -(1/beta) * log(sum(exp(-beta * area)))
    # For stability, subtract max
    max_area = np.max(areas)
    exp_vals = np.exp(-beta * (areas - max_area))
    soft_min = max_area - (1.0 / beta) * np.log(np.sum(exp_vals))
    return -soft_min

def _objective(params, beta):
    return _soft_min_objective(params, beta)

def _gradient(params, beta):
    """Numerical gradient for L-BFGS-B (since analytical gradient is complex with abs)."""
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _soft_min_objective(params, beta)
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += eps
        grad[i] = (_soft_min_objective(params_p, beta) - f0) / eps
    return grad

def _random_initial_params(rng, mode='random'):
    """Generate random initial barycentric parameters."""
    if mode == 'random':
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        # Clip to stay strictly inside
        u = np.clip(u, 0.01, 0.98)
        v = np.clip(v, 0.01, 0.98 - u)
    elif mode == 'grid':
        # Try a structured initial guess
        u = np.linspace(0.1, 0.8, N)
        v = np.linspace(0.1, 0.5, N)
        # Perturb slightly
        u += rng.uniform(-0.05, 0.05, N)
        v += rng.uniform(-0.05, 0.05, N)
        u = np.clip(u, 0.01, 0.98)
        v = np.clip(v, 0.01, 0.98 - u)
    else:
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        u = np.clip(u, 0.01, 0.98)
        v = np.clip(v, 0.01, 0.98 - u)
    
    params = np.zeros(2 * N)
    params[0::2] = u
    params[1::2] = v
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_params = None
    best_score = -np.inf
    best_points = None
    
    # Start with a reasonable initial guess
    initial_params = _random_initial_params(rng, mode='random')
    
    # Phase 1: Multiple restarts with L-BFGS-B
    beta = 50.0  # Soft-min sharpness
    
    n_restarts = 30  # Max restarts, but will stop based on time
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial parameters
        if restart == 0:
            params0 = initial_params
        elif restart % 3 == 1:
            params0 = _random_initial_params(rng, mode='grid')
        else:
            params0 = _random_initial_params(rng, mode='random')
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective,
                params0,
                args=(beta,),
                method='L-BFGS-B',
                jac=_gradient,
                bounds=[(0.01, 0.98)] * N + [(0.01, 0.98)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        params = result.x
        # Ensure feasibility
        u = params[0::2]
        v = params[1::2]
        u = np.clip(u, 0.01, 0.98)
        v = np.clip(v, 0.01, 0.98 - u)
        params[0::2] = u
        params[1::2] = v
        
        points = _barycentric_to_cartesian(params)
        score = _min_triangle_area(points)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_points = points.copy()
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Local refinement with higher beta (sharper soft-min)
    if best_params is not None and time.time() < deadline - 5:
        for beta_refine in [100.0, 200.0]:
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    _objective,
                    best_params,
                    args=(beta_refine,),
                    method='L-BFGS-B',
                    jac=_gradient,
                    bounds=[(0.01, 0.98)] * N + [(0.01, 0.98)] * N,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
            except Exception:
                continue
            
            if time.time() > deadline:
                break
            
            params = result.x
            u = params[0::2]
            v = params[1::2]
            u = np.clip(u, 0.01, 0.98)
            v = np.clip(v, 0.01, 0.98 - u)
            params[0::2] = u
            params[1::2] = v
            
            points = _barycentric_to_cartesian(params)
            score = _min_triangle_area(points)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_points = points.copy()
    
    if best_points is None:
        # Fallback: return random points
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        u = np.clip(u, 0.01, 0.98)
        v = np.clip(v, 0.01, 0.98 - u)
        x = u + 0.5 * v
        y = H * v
        best_points = np.column_stack([x, y])
    
    return best_points
