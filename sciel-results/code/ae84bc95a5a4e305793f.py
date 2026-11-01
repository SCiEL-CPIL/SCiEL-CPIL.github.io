"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

# EVOLVE-BLOCK-START

def _pairwise_sq_dists(points):
    """Compute squared pairwise distances as flat array of length N*(N-1)/2."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    sq_dists = _pairwise_sq_dists(points)
    d_min_sq = np.min(sq_dists)
    d_max_sq = np.max(sq_dists)
    if d_min_sq < 1e-24:
        return 1e10
    return d_max_sq / d_min_sq

def _soft_max_sq(x, beta=50.0):
    """Soft max of squared distances using log-sum-exp."""
    # x is array of squared distances
    x_max = np.max(x)
    if x_max < 1e-15:
        return x_max
    # Use log-sum-exp trick for numerical stability
    shifted = x - x_max
    log_sum = x_max + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum

def _soft_min_sq(x, beta=50.0):
    """Soft min of squared distances using log-sum-exp (negative log-sum-exp)."""
    # x is array of squared distances
    x_min = np.min(x)
    if x_min < 1e-15:
        return x_min
    shifted = x - x_min
    log_sum = x_min - np.log(np.sum(np.exp(-beta * shifted))) / beta
    return log_sum

def _objective_smooth(params, beta=50.0):
    """Smooth objective: (soft_max / soft_min)^2."""
    points = params.reshape(N, D)
    sq_dists = _pairwise_sq_dists(points)
    smax = _soft_max_sq(sq_dists, beta)
    smin = _soft_min_sq(sq_dists, beta)
    if smin < 1e-15:
        return 1e10
    return (smax / smin) ** 2

def _objective_true(params):
    """True objective: (max / min)^2."""
    points = params.reshape(N, D)
    return _score(points)

def _gradient_smooth(params, beta=50.0):
    """Numerical gradient of the smooth objective."""
    eps = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (_objective_smooth(params_plus, beta) - _objective_smooth(params_minus, beta)) / (2 * eps)
    return grad

def _tetrahedron_initial():
    """
    Generate initial configuration based on regular tetrahedron vertices
    plus midpoints of edges, then add additional points.
    
    A regular tetrahedron has 4 vertices and 6 edges.
    4 vertices + 6 midpoints = 10 points.
    We need 14, so we add 4 more points.
    """
    # Regular tetrahedron vertices (centered at origin)
    a = 1.0
    verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ], dtype=float)
    
    # Center at origin
    verts -= np.mean(verts, axis=0)
    
    # Scale to unit edge length for consistency
    edge_len = np.linalg.norm(verts[0] - verts[1])
    verts /= edge_len
    
    # Edge midpoints
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    midpoints = []
    for i, j in edges:
        midpoints.append((verts[i] + verts[j]) / 2)
    midpoints = np.array(midpoints)
    
    # We have 4 + 6 = 10 points. Need 4 more.
    # Add points at face centers (4 faces)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Combine: 4 vertices + 6 midpoints + 4 face centers = 14 points
    points = np.vstack([verts, midpoints, face_centers])
    
    # Center the whole configuration
    points -= np.mean(points, axis=0)
    
    return points

def _polish(points, start_time, deadline, beta=100.0):
    """Final high-precision polish pass with tighter tolerances."""
    if time.time() > deadline - 5.0:
        return points
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Try L-BFGS-B with smooth objective and tight tolerances
    params0 = best_points.flatten()
    
    try:
        result = minimize(
            _objective_smooth,
            params0,
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 10000,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 50
            }
        )
        
        if result.fun < 1e9:
            candidate = result.x.reshape(N, D)
            candidate_score = _score(candidate)
            if candidate_score < best_score:
                best_score = candidate_score
                best_points = candidate.copy()
    except Exception:
        pass
    
    # Try Nelder-Mead with very tight tolerances
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective_smooth,
                best_points.flatten(),
                args=(beta,),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result.fun < 1e9:
                candidate = result.x.reshape(N, D)
                candidate_score = _score(candidate)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    return best_points

def construct():
    """Construct 14 points in 3D to minimize squared max/min distance ratio."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Initial configuration from tetrahedron-based construction
    points0 = _tetrahedron_initial()
    best_points = points0.copy()
    best_score = _score(best_points)
    
    # Try multiple random initializations and local optimizations
    np.random.seed(42)
    
    # Multi-stage beta annealing: start with smooth, get sharper
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    n_attempts = 0
    while time.time() < deadline - 20.0:
        if n_attempts == 0:
            init_points = points0
        else:
            # Random initialization with some structure
            noise_scale = 0.05 + 0.03 * n_attempts
            init_points = points0 + np.random.randn(N, D) * noise_scale
            init_points -= np.mean(init_points, axis=0)
        
        params0 = init_points.flatten()
        
        # Use Nelder-Mead with smooth objective
        beta = beta_schedule[min(n_attempts // 2, len(beta_schedule) - 1)]
        
        result = minimize(
            _objective_smooth,
            params0,
            args=(beta,),
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.fun < 1e9:
            candidate = result.x.reshape(N, D)
            # Ensure distinctness
            sq_dists = _pairwise_sq_dists(candidate)
            if np.min(sq_dists) > 1e-16:
                candidate_score = _score(candidate)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        
        n_attempts += 1
        
        # Check time
        if time.time() > deadline - 25.0:
            break
        
        # After some attempts, try L-BFGS-B with gradient
        if n_attempts == 2 or n_attempts == 4:
            for trial in range(2):
                if time.time() > deadline - 20.0:
                    break
                noise_scale = 0.03 * (trial + 1)
                init_points = points0 + np.random.randn(N, D) * noise_scale
                init_points -= np.mean(init_points, axis=0)
                params0 = init_points.flatten()
                
                beta = beta_schedule[min(n_attempts // 2, len(beta_schedule) - 1)]
                
                result = minimize(
                    _objective_smooth,
                    params0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 3000,
                        'ftol': 1e-12,
                        'gtol': 1e-10
                    }
                )
                
                if result.fun < 1e9:
                    candidate = result.x.reshape(N, D)
                    sq_dists = _pairwise_sq_dists(candidate)
                    if np.min(sq_dists) > 1e-16:
                        candidate_score = _score(candidate)
                        if candidate_score < best_score:
                            best_score = candidate_score
                            best_points = candidate.copy()
        
        if n_attempts >= 6:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        best_points = _polish(best_points, start_time, deadline, beta=100.0)
    
    # Final check: ensure all points are distinct
    sq_dists = _pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-16:
        # Fallback to initial
        best_points = points0
    
    return best_points

# EVOLVE-BLOCK-END
