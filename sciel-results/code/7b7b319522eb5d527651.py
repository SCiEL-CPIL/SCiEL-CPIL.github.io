import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective(x):
    points = x.reshape(-1, D)
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals == 0):
        return 1e10
    
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    return d_max_sq / d_min_sq

def construct():
    # Seed 1: 4x4 Grid
    grid = np.array(list(itertools.product(range(4), repeat=2)), dtype=float)
    seed1 = grid - np.mean(grid, axis=0)
    
    # Seed 2: 16 points on a circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    seed2 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Seed 3: Two concentric circles (8 inner, 8 outer)
    a_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    a_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner = np.column_stack([np.cos(a_in), np.sin(a_in)])
    outer = 1.8 * np.column_stack([np.cos(a_out), np.sin(a_out)])
    seed3 = np.vstack([inner, outer])
    
    # Seed 4: Hexagonal
    hex_pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if (abs(q) + abs(r) + abs(q+r)) // 2 <= 2:
                hex_pts.append((1.5*q, np.sqrt(3)/2*q + np.sqrt(3)*r))
    hex_arr = np.array(hex_pts)
    d0 = np.sum(hex_arr**2, axis=1)
    idx = np.argsort(d0)[:N]
    seed4 = hex_arr[idx] - np.mean(hex_arr[idx], axis=0)
    
    # Seed 5: Rotated 4x4 grid
    seed5 = seed1.copy()
    angle = np.pi/8
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    seed5 = seed5 @ R.T
    
    seeds = [seed1, seed2, seed3, seed4, seed5]
    
    best_score = np.inf
    best_points = None
    
    for s in seeds:
        sc = objective(s.flatten())
        if sc < best_score:
            best_score = sc
            best_points = s.copy()
    
    if best_points is not None:
        try:
            res = minimize(objective, best_points.flatten(), method='Powell',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(-1, D)
        except Exception:
            pass
    
    try:
        bounds = [(-3, 3)] * (N*D)
        de_res = differential_evolution(objective, bounds, seed=42, maxiter=30,
                                       popsize=10, tol=1e-3, init='sobol')
        if de_res.fun < best_score:
            best_score = de_res.fun
            best_points = de_res.x.reshape(-1, D)
    except Exception:
        pass
    
    if best_points is not None:
        try:
            res = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(-1, D)
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective(x):
    points = x.reshape(-1, D)
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals == 0):
        return 1e10
    
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    return d_max_sq / d_min_sq

def construct():
    # Seed 1: 4x4 Grid
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    seed1 = np.array(grid, dtype=float)
    seed1 -= np.mean(seed1, axis=0)
    
    # Seed 2: Hexagonal subset (16 closest to center in radius 2 hex)
    hex_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= 2:
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                hex_candidates.append((x, y))
    hex_arr = np.array(hex_candidates)
    d_orig = np.sum(hex_arr**2, axis=1)
    sorted_indices = np.argsort(d_orig)
    seed2 = hex_arr[sorted_indices[:N]]
    seed2 -= np.mean(seed2, axis=0)

    # Seed 3: Random
    np.random.seed(42)
    seed3 = np.random.rand(N, D) * 2 - 1
    seed3 -= np.mean(seed3, axis=0)

    # Seed 4: 2x8 Grid
    grid2 = list(itertools.product(range(2), range(8)))[:N]
    seed4 = np.array(grid2, dtype=float)
    seed4 -= np.mean(seed4, axis=0)
    
    # Seed 5: 8x2 Grid
    grid5 = list(itertools.product(range(8), range(2)))[:N]
    seed5 = np.array(grid5, dtype=float)
    seed5 -= np.mean(seed5, axis=0)

    seeds = [seed1, seed2, seed3, seed4, seed5]
    
    best_score = np.inf
    best_points = None
    
    for seed in seeds:
        try:
            res = minimize(objective, seed.flatten(), method='Powell', options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(-1, D)
        except Exception:
            pass
            
    if best_points is None:
        for seed in seeds:
            s = objective(seed.flatten())
            if s < best_score:
                best_score = s
                best_points = seed
                
    return best_points
# EVOLVE-BLOCK-END
