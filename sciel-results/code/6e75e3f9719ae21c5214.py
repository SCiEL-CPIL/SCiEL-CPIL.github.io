"""Optimize 50 equal circles in a unit square using SLSQP with epigraph formulation."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _pairwise_distances(c):
    """Return matrix of pairwise distances (N, N)."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(c):
    """Compute the radius for a given configuration."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(c < -0.5) or np.any(c > 0.5):
        # Allow small tolerance for numerical issues
        if np.any(c < -0.5 - 1e-10) or np.any(c > 0.5 + 1e-10):
            return 0.0
    c = np.clip(c, -0.5, 0.5)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # Distance to walls
    wall_dists = np.minimum(np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1]))
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _initial_grid():
    """Create a grid-like initial guess."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a slightly perturbed grid to avoid symmetry issues
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    # Add small random perturbation
    rng = np.random.default_rng(42)
    pts += rng.normal(0, 0.001, size=pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    return pts


def _slsqp_objective(x, t):
    """
    Objective: minimize -t (i.e., maximize t = radius).
    x: flattened (N, 2) positions
    t: radius variable
    """
    return -t


def _slsqp_constraints(x, t):
    """
    Constraints for SLSQP:
    For each pair i<j: ||c_i - c_j|| >= 2*t
    For each i, each wall: distance >= t
    """
    c = x[:2 * N].reshape(N, 2)
    constraints = []
    
    # Pairwise constraints: ||c_i - c_j|| - 2*t >= 0
    for i in range(N):
        for j in range(i + 1, N):
            diff = c[i] - c[j]
            d = np.sqrt(np.sum(diff ** 2))
            constraints.append(d - 2 * t)
    
    # Wall constraints
    for i in range(N):
        # Right wall: 0.5 - x_i >= t
        constraints.append(0.5 - c[i, 0] - t)
        # Left wall: x_i + 0.5 >= t
        constraints.append(c[i, 0] + 0.5 - t)
        # Top wall: 0.5 - y_i >= t
        constraints.append(0.5 - c[i, 1] - t)
        # Bottom wall: y_i + 0.5 >= t
        constraints.append(c[i, 1] + 0.5 - t)
    
    return constraints


def _run_slsqp(c0, t0, time_limit):
    """Run SLSQP optimization with time limit."""
    start_time = time.time()
    
    # Flatten: x = [c_0, c_1, ..., c_{N-1}, t]
    x0 = np.concatenate([c0.flatten(), [t0]])
    
    # Constraints in scipy format
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            def pair_constraint(x, t, i=i, j=j):
                c = x[:2 * N].reshape(N, 2)
                diff = c[i] - c[j]
                return np.sqrt(np.sum(diff ** 2)) - 2 * x[2 * N]
            cons.append({'type': 'ineq', 'fun': pair_constraint})
    
    for i in range(N):
        # Right wall
        def right_wall(x, i=i):
            c = x[:2 * N].reshape(N, 2)
            return 0.5 - c[i, 0] - x[2 * N]
        cons.append({'type': 'ineq', 'fun': right_wall})
        
        # Left wall
        def left_wall(x, i=i):
            c = x[:2 * N].reshape(N, 2)
            return c[i, 0] + 0.5 - x[2 * N]
        cons.append({'type': 'ineq', 'fun': left_wall})
        
        # Top wall
        def top_wall(x, i=i):
            c = x[:2 * N].reshape(N, 2)
            return 0.5 - c[i, 1] - x[2 * N]
        cons.append({'type': 'ineq', 'fun': top_wall})
        
        # Bottom wall
        def bottom_wall(x, i=i):
            c = x[:2 * N].reshape(N, 2)
            return c[i, 1] + 0.5 - x[2 * N]
        cons.append({'type': 'ineq', 'fun': bottom_wall})
    
    # Bounds for positions
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    result = minimize(
        lambda x: _slsqp_objective(x, x[2 * N]),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
    )
    
    elapsed = time.time() - start_time
    return result, elapsed


def _polish(c, t, time_limit):
    """High-precision polish pass."""
    start_time = time.time()
    x0 = np.concatenate([c.flatten(), [t]])
    
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            def pair_constraint(x, i=i, j=j):
                c = x[:2 * N].reshape(N, 2)
                diff = c[i] - c[j]
                return np.sqrt(np.sum(diff ** 2)) - 2 * x[2 * N]
            cons.append({'type': 'ineq', 'fun': pair_constraint})
    
    for i in range(N):
        def right_wall(x, i=i):
            c = x[:2 * N].reshape(N, 2)
            return 0.5 - c[i, 0] - x[2 * N]
        cons.append({'type': 'ineq', 'fun': right_wall})
        
        def left_wall(x, i=i):
            c = x[:2 * N].reshape(N, 2)
            return c[i, 0] + 0.5 - x[2 * N]
        cons.append({'type': 'ineq', 'fun': left_wall})
        
        def top_wall(x, i=i):
            c = x[:2 * N].reshape(N, 2)
            return 0.5 - c[i, 1] - x[2 * N]
        cons.append({'type': 'ineq', 'fun': top_wall})
        
        def bottom_wall(x, i=i):
            c = x[:2 * N].reshape(N, 2)
            return c[i, 1] + 0.5 - x[2 * N]
        cons.append({'type': 'ineq', 'fun': bottom_wall})
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    result = minimize(
        lambda x: -x[2 * N],
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 5000, 'ftol': 1e-18, 'disp': False}
    )
    
    elapsed = time.time() - start_time
    return result, elapsed


def construct():
    """Construct 50 circle centers maximizing the minimum radius."""
    global_start = time.time()
    deadline = global_start + 140.0  # Safety margin
    
    best_c = None
    best_r = 0.0
    
    # Initial grid
    c0 = _initial_grid()
    r0 = _score(c0)
    best_c = c0.copy()
    best_r = r0
    
    # Run SLSQP
    try:
        result, elapsed = _run_slsqp(c0, r0, 30.0)
        if result.success or result.fun is not None:
            x_opt = result.x
            c_opt = x_opt[:2 * N].reshape(N, 2)
            t_opt = x_opt[2 * N]
            # Clip to valid range
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = _score(c_opt)
            if r_opt > best_r:
                best_c = c_opt
                best_r = r_opt
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_c
    
    # Polish pass
    try:
        result2, elapsed2 = _polish(best_c, best_r, 15.0)
        if result2.success or result2.fun is not None:
            x_opt2 = result2.x
            c_opt2 = x_opt2[:2 * N].reshape(N, 2)
            t_opt2 = x_opt2[2 * N]
            c_opt2 = np.clip(c_opt2, -0.5, 0.5)
            r_opt2 = _score(c_opt2)
            if r_opt2 > best_r:
                best_c = c_opt2
                best_r = r_opt2
    except Exception as e:
        pass
    
    return best_c
# EVOLVE-BLOCK-END
