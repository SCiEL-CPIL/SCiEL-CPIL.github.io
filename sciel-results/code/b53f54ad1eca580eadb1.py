"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def _score(c):
    """Compute the radius for a given set of centers."""
    if not np.all(np.isfinite(c)):
        return 0.0
    if c.shape != (N, 2):
        return 0.0
    # Check bounds
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return 0.0
    # Compute min distance between centers / 2
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(c[i] - c[j])
            if d < min_pair_dist:
                min_pair_dist = d
    min_pair_r = min_pair_dist / 2.0
    # Compute min distance to boundary
    min_boundary = np.inf
    for i in range(N):
        for k in range(2):
            d = 0.5 - abs(c[i, k])
            if d < min_boundary:
                min_boundary = d
    return min(min_pair_r, min_boundary)


def _initial_grid():
    """Create an initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _slsqp_optimize(c0, max_time=30.0, ftol=1e-14):
    """
    Optimize circle packing using SLSQP with linear epigraph formulation.
    
    We maximize t subject to:
      - Pairwise distances: ||c_i - c_j|| >= 2t for all i < j
      - Boundary: 0.5 - |c_ik| >= t for all i, k
      
    Using epigraph: minimize -t
    Constraints:
      - ||c_i - c_j||^2 - (2t)^2 >= 0  (nonlinear, but we linearize or use direct)
    
    Actually, SLSQP can handle nonlinear constraints directly.
    """
    start_time = time.time()
    n = len(c0)
    
    # Variables: [c_0x, c_0y, c_1x, c_1y, ..., c_{n-1}x, c_{n-1}y, t]
    # Total: 2*n + 1 variables
    
    def flatten(c, t):
        return np.concatenate([c.flatten(), [t]])
    
    def unflatten(v):
        c = v[:2*n].reshape(n, 2)
        t = v[2*n]
        return c, t
    
    def objective(v):
        c, t = unflatten(v)
        return -t
    
    # Constraints: pairwise distance >= 2t
    # For SLSQP, we need constraints of form g(v) >= 0
    # ||c_i - c_j|| - 2t >= 0
    
    # Boundary constraints: 0.5 - |c_ik| - t >= 0
    
    def constraint_func(v):
        c, t = unflatten(v)
        cons = []
        # Pairwise
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                cons.append(d - 2 * t)
        # Boundary
        for i in range(n):
            for k in range(2):
                cons.append(0.5 - abs(c[i, k]) - t)
        return cons
    
    # Initial value
    c_init = c0.copy()
    t_init = _score(c_init)
    v0 = flatten(c_init, t_init)
    
    # Bounds: centers in [-0.5, 0.5], t >= 0
    bounds = []
    for i in range(n):
        bounds.append((-0.5, 0.5))
        bounds.append((-0.5, 0.5))
    bounds.append((0, 0.5))
    
    # Use SLSQP
    result = None
    best_c = c0.copy()
    best_t = t_init
    
    def callback(xk):
        nonlocal best_c, best_t
        c, t = unflatten(xk)
        if np.all(np.isfinite(c)) and np.all(np.abs(c) <= 0.5 + 1e-10):
            actual_t = _score(c)
            if actual_t > best_t:
                best_t = actual_t
                best_c = c.copy()
    
    try:
        result = minimize(
            objective,
            v0,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraint_func}],
            options={'maxiter': 500, 'ftol': ftol, 'disp': False},
            callback=callback
        )
    except Exception as e:
        pass
    
    # Check result
    if result is not None and result.success:
        c_res, t_res = unflatten(result.x)
        if np.all(np.isfinite(c_res)) and np.all(np.abs(c_res) <= 0.5 + 1e-10):
            actual_t = _score(c_res)
            if actual_t > best_t:
                best_t = actual_t
                best_c = c_res.copy()
    
    return best_c, best_t


def _local_search(c0, max_time=20.0):
    """
    Perform local search by pushing circles apart.
    """
    start_time = time.time()
    c = c0.copy()
    n = len(c)
    
    def score(c):
        return _score(c)
    
    current_score = score(c)
    
    # Simple iterative relaxation
    lr = 0.01
    for iteration in range(10000):
        if time.time() - start_time > max_time:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random small push
                    d = np.random.randn(2)
                    dist = np.linalg.norm(d)
                # If too close, push apart
                if dist < 2 * current_score + 0.01:
                    force = d / dist * (2 * current_score + 0.01 - dist)
                    forces[i] += force
                    forces[j] -= force
            # Boundary forces
            for k in range(2):
                if c[i, k] > 0.5 - current_score:
                    forces[i, k] -= (c[i, k] - (0.5 - current_score))
                elif c[i, k] < -0.5 + current_score:
                    forces[i, k] += (-0.5 + current_score - c[i, k])
        
        # Update
        c_new = c + lr * forces
        # Project to bounds
        c_new = np.clip(c_new, -0.5, 0.5)
        
        new_score = score(c_new)
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            lr *= 0.99
        else:
            lr *= 0.9
    
    return c, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with grid
    c = _initial_grid()
    best_c = c.copy()
    best_score = _score(c)
    
    # Phase 1: SLSQP optimization
    if time.time() < deadline - 40:
        c1, s1 = _slsqp_optimize(c, max_time=30.0, ftol=1e-14)
        if s1 > best_score:
            best_score = s1
            best_c = c1.copy()
        c = c1
    
    # Phase 2: Local search polish
    if time.time() < deadline - 20:
        c2, s2 = _local_search(c, max_time=15.0)
        if s2 > best_score:
            best_score = s2
            best_c = c2.copy()
        c = c2
    
    # Phase 3: Final SLSQP polish with tighter tolerance
    if time.time() < deadline - 5:
        c3, s3 = _slsqp_optimize(c, max_time=10.0, ftol=1e-15)
        if s3 > best_score:
            best_score = s3
            best_c = c3.copy()
    
    # Ensure validity
    if not np.all(np.isfinite(best_c)) or np.any(np.abs(best_c) > 0.5):
        best_c = _initial_grid()
    
    return best_c
# EVOLVE-BLOCK-END
