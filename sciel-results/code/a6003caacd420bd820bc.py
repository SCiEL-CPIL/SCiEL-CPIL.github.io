import numpy as np
from scipy.optimize import minimize
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    # points: (N, 2)
    # returns (N, N) matrix of squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective(x):
    # x: flattened array of shape (N*D,)
    points = x.reshape(-1, D)
    d_sq = pairwise_distances_sq(points)
    # mask diagonal
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals == 0):
        return 1e10
    
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    # Minimize ratio of max to min distance, squared
    # (D_max / D_min)^2 = D_max^2 / D_min^2
    return d_max_sq / d_min_sq

def construct():
    # Generate seeds
    
    # Seed 1: 4x4 Grid
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    seed1 = np.array(grid, dtype=float)
    # Center it
    seed1 -= np.mean(seed1, axis=0)
    
    # Seed 2: Hexagonal subset
    # Generate 19 points of hex radius 2
    hex_points = []
    for j in range(-2, 3):
        for i in range(-2, 3):
            # Hexagonal lattice condition: |i| + |j| + |i+j| <= 2*2? No, simple bounding box check for hex shape
            # Actually, just generate all i,j in range and filter by distance from origin in hex metric or Euclidean
            pass
            
    # Simpler hex generation:
    # Axial coordinates (q, r)
    # Hex distance: (|q| + |r| + |q+r|) / 2
    # We want distance <= 2
    hex_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= 2:
                # Convert axial to Cartesian
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                hex_candidates.append((x, y))
                
    # This gives 19 points
    # We need 16. Let's remove 3 points that are "far" from center?
    # Or just take the first 16?
    # Better: Sort by distance from center, take the 16 closest?
    # The 19 points are: 1 center, 6 at dist 1, 12 at dist 2.
    # If we take all 13 (center + ring 1) and 3 from ring 2, we get 16.
    # Which 3 from ring 2? To maintain symmetry, we can't pick 3 symmetrically from 12 (12 is divisible by 3, 4, 6, 12).
    # If we pick 3, we break 6-fold symmetry.
    # Maybe pick 4? No, we need 16.
    # Let's just take the 16 points with smallest distance from origin.
    # There are 13 points with dist <= 1.
    # The next 3 points from ring 2.
    # This might be a good seed.
    
    hex_arr = np.array(hex_candidates)
    # Calculate distance from origin for each
    d_orig = np.sum(hex_arr**2, axis=1)
    # Sort indices by distance
    sorted_indices = np.argsort(d_orig)
    seed2 = hex_arr[sorted_indices[:N]]
    seed2 -= np.mean(seed2, axis=0)

    # Seed 3: Random
    np.random.seed(42)
    seed3 = np.random.rand(N, D) * 2 - 1
    seed3 -= np.mean(seed3, axis=0)

    # Seed 4: 2x8 Grid (rectangular)
    grid2 = list(itertools.product(range(2), range(8)))[:N]
    seed4 = np.array(grid2, dtype=float)
    seed4 -= np.mean(seed4, axis=0)

    seeds = [seed1, seed2, seed3, seed4]
    
    best_score = np.inf
    best_points = None
    
    for seed in seeds:
        # Optimize
        # Use Powell method, good for non-smooth
        # maxiter limit to ensure speed
        try:
            res = minimize(objective, seed.flatten(), method='Powell', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_points = res.x.reshape(-1, D)
        except Exception:
            pass
            
    # If optimization failed or didn't improve, return the best seed
    if best_points is None:
        # Evaluate seeds directly
        for seed in seeds:
            s = objective(seed.flatten())
            if s < best_score:
                best_score = s
                best_points = seed
                
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
