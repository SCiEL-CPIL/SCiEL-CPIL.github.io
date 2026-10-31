import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 16
D = 2

# EVOLVE-BLOCK-START

def hex_to_cartesian(q, r):
    x = q + r / 2.0
    y = r * np.sqrt(3) / 2.0
    return np.array([x, y])

def generate_hexagon_points(radius=2):
    points = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            # Hex distance
            dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
            if dist <= radius:
                points.append(hex_to_cartesian(q, r))
    return np.array(points)

def objective(x):
    pts = x.reshape(N, D)
    # Compute pairwise distances
    # Using broadcasting
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min < 1e-9:
        return 1e10
    return (d_max / d_min) ** 2

def optimize_from_seed(seed_pts):
    x0 = seed_pts.flatten()
    # Powell method is robust for non-smooth functions
    res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'xtol': 1e-6, 'ftol': 1e-6})
    return res.x.reshape(N, D), res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_score = np.inf
    best_pts = None
    
    # Seed 1: Hexagon subset
    hex_pts = generate_hexagon_points(2)
    # hex_pts has 19 points
    # Identify opposite pairs at distance 4
    # We will try all 8 combinations of removing one point from each opposite pair
    # Opposite pairs: (0,1), (2,3), (4,5) in some ordering? 
    # Let's find them programmatically.
    
    # Find points at max distance from origin (approx)
    # Or just hardcode the logic for R=2
    # The 6 corners are:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # In Cartesian:
    # (2,0) -> (2,0)
    # (0,2) -> (1, sqrt(3))
    # (-2,2) -> (-1, sqrt(3))
    # (-2,0) -> (-2,0)
    # (0,-2) -> (-1, -sqrt(3))
    # (2,-2) -> (1, -sqrt(3))
    
    # Let's just generate the 19 points and find the 6 furthest from centroid (0,0)
    # Actually, all 6 corners are at distance 2 from center in hex dist, but Euclidean dist 2?
    # No, Euclidean dist of (2,0) is 2.
    # Euclidean dist of (0,2) is 2.
    # So all 6 corners are at Euclidean distance 2 from origin.
    
    corners = []
    for p in hex_pts:
        if np.linalg.norm(p) > 1.9: # Close to 2
            corners.append(p)
            
    # There should be 6 corners
    # Pair them up: p and -p
    pairs = []
    used = [False] * len(corners)
    for i in range(len(corners)):
        if used[i]: continue
        for j in range(i+1, len(corners)):
            if used[j]: continue
            if np.allclose(corners[i] + corners[j], 0, atol=1e-6):
                pairs.append((i, j))
                used[i] = True
                used[j] = True
                break
                
    # We have 3 pairs. We want to remove one from each.
    # There are 2^3 = 8 choices.
    for mask in itertools.product([0, 1], repeat=3):
        remove_indices = []
        for k in range(3):
            i, j = pairs[k]
            if mask[k] == 0:
                remove_indices.append(i)
            else:
                remove_indices.append(j)
        
        # Create mask of points to keep
        keep_mask = np.ones(len(hex_pts), dtype=bool)
        # Map corner indices back to hex_pts indices?
        # Easier: just filter hex_pts by removing the specific coordinate values
        # But floating point equality is risky.
        # Let's just construct the list of 16 points by taking hex_pts and removing the 3 specific points.
        
        # Find indices in hex_pts corresponding to the removed corners
        remove_hex_idx = []
        for idx in remove_indices:
            target = corners[idx]
            # Find match in hex_pts
            for k, p in enumerate(hex_pts):
                if np.allclose(p, target, atol=1e-6):
                    remove_hex_idx.append(k)
                    break
        
        keep_mask[remove_hex_idx] = False
        seed_pts = hex_pts[keep_mask]
        
        if len(seed_pts) != N:
            continue
            
        x0 = seed_pts.flatten()
        
        # Run optimization
        # Check time
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'xtol': 1e-5, 'ftol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(N, D)
        except Exception as e:
            pass

    # Seed 2: Square Grid (just in case)
    if time.time() < deadline:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        seed_pts = np.array(grid, dtype=float)
        x0 = seed_pts.flatten()
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'xtol': 1e-5, 'ftol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(N, D)
        except Exception as e:
            pass

    # Seed 3: Random
    if time.time() < deadline and best_pts is None:
        seed_pts = np.random.rand(N, D) * 10
        x0 = seed_pts.flatten()
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'xtol': 1e-5, 'ftol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(N, D)
        except Exception as e:
            pass

    if best_pts is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)

    return best_pts

# EVOLVE-BLOCK-END
