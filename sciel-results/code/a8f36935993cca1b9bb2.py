"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Deadline guard
DEADLINE = 145.0  # seconds, safety margin before 150s kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _radius(c):
    """Compute the largest common radius for centres c (N,3)."""
    if c.shape[0] < 2:
        return 0.0
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Distance to walls
    wall = 0.5 - np.abs(c).max(axis=1)
    min_wall = wall.min()
    return min(min_pair / 2.0, min_wall)


def _objective(x, N):
    """Minimize -radius (i.e., maximize radius)."""
    c = x.reshape(N, 3)
    return -_radius(c)


def _constraints(c, N):
    """Return list of constraint dicts for SLSQP."""
    cons = []
    # Wall constraints: |c_ik| <= 0.5 - r, but we handle via objective.
    # We'll use a penalty approach instead for simplicity.
    return cons


def _penalty_objective(x, N, r_target):
    """Objective with penalty for constraint violations."""
    c = x.reshape(N, 3)
    # Wall violations
    wall_viol = np.maximum(0, np.abs(c) - (0.5 - r_target))
    wall_pen = np.sum(wall_viol**2)
    # Pairwise distance violations
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_viol = np.maximum(0, 2 * r_target - d)
    pair_pen = np.sum(pair_viol**2)
    return wall_pen + pair_pen


def _random_init(N, seed=42):
    rng = np.random.RandomState(seed)
    # Start near a grid but with random jitter
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    # Add small random jitter
    c += rng.normal(0, 0.01, c.shape)
    # Clip to cube
    c = np.clip(c, -0.48, 0.48)
    return c


def _simulated_annealing_step(c, r, T, rng):
    """Perform one SA step: perturb one or a few points."""
    c_new = c.copy()
    # Pick 1-3 random points to move
    n_move = rng.randint(1, 4)
    idx = rng.choice(N, size=n_move, replace=False)
    # Gaussian perturbation scaled by temperature
    scale = T * 0.05
    c_new[idx] += rng.normal(0, scale, (n_move, 3))
    # Clip to cube
    c_new = np.clip(c_new, -0.5, 0.5)
    r_new = _radius(c_new)
    # Accept if better or with probability exp((r_new - r)/T)
    if r_new >= r:
        return c_new, r_new, True
    else:
        accept_prob = np.exp((r_new - r) / max(T, 1e-10))
        if rng.uniform() < accept_prob:
            return c_new, r_new, True
        else:
            return c, r, False


def _local_polish(c, max_iter=500, tol=1e-10):
    """Polish using gradient-free local search (Nelder-Mead on flattened vector)."""
    x = c.flatten()
    # Use Nelder-Mead to maximize radius
    best_x = x.copy()
    best_r = _radius(c)
    
    # Nelder-Mead on -radius
    result = minimize(
        lambda v: -_radius(v.reshape(N, 3)),
        x,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
    )
    c_opt = result.x.reshape(N, 3)
    r_opt = _radius(c_opt)
    
    if r_opt > best_r:
        return c_opt, r_opt
    return c, best_r


def construct():
    global _start_time
    _start_time = time.time()
    
    rng = np.random.RandomState(12345)
    
    # Best solution found
    best_c = None
    best_r = -1.0
    
    # Initial solutions from different seeds
    n_starts = 5
    for s in range(n_starts):
        if _elapsed() > DEADLINE - 10:
            break
        c = _random_init(N, seed=42 + s * 7)
        r = _radius(c)
        if r > best_r:
            best_c = c.copy()
            best_r = r
    
    # Simulated Annealing phase
    if best_c is not None:
        c = best_c.copy()
        r = best_r
        T = 0.01
        cooling = 0.995
        n_sa_steps = 20000
        
        for step in range(n_sa_steps):
            if _elapsed() > DEADLINE - 30:
                break
            c, r, accepted = _simulated_annealing_step(c, r, T, rng)
            if r > best_r:
                best_r = r
                best_c = c.copy()
            T *= cooling
            if T < 1e-6:
                break
        
        # Local polish after SA
        if _elapsed() < DEADLINE - 15:
            c_polished, r_polished = _local_polish(best_c, max_iter=200)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished
    
    # Additional restarts with SA+polish if time remains
    n_extra = 3
    for s in range(n_extra):
        if _elapsed() > DEADLINE - 20:
            break
        c = _random_init(N, seed=200 + s * 13)
        r = _radius(c)
        
        # Short SA
        T = 0.005
        cooling = 0.99
        for step in range(5000):
            if _elapsed() > DEADLINE - 20:
                break
            c, r, accepted = _simulated_annealing_step(c, r, T, rng)
            T *= cooling
            if T < 1e-6:
                break
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Polish
        if _elapsed() < DEADLINE - 10:
            c_polished, r_polished = _local_polish(c, max_iter=100)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished
    
    # Final high-precision polish
    if best_c is not None and _elapsed() < DEADLINE - 5:
        c_final, r_final = _local_polish(best_c, max_iter=500, tol=1e-12)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final
    
    # Validate and return
    if best_c is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    # Ensure finite and within bounds
    best_c = np.nan_to_num(best_c, nan=0.0, posinf=0.49, neginf=-0.49)
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
