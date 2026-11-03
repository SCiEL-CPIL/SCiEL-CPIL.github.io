import numpy as np
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3)

def hexagon_violation(x, y, S):
    R = S * np.sqrt(3) / 2
    thetas = np.deg2rad(np.arange(0, 360, 60))
    dots = x[:, np.newaxis] * np.cos(thetas)[np.newaxis, :] + y[:, np.newaxis] * np.sin(thetas)[np.newaxis, :]
    max_dots = np.max(dots, axis=1)
    return np.maximum(0, max_dots - R)**2

def packing_violation(positions, S):
    x = positions[:, 0]
    y = positions[:, 1]
    
    v_boundary = np.sum(hexagon_violation(x, y, S))
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    v_dist = np.sum(np.maximum(0, MIN_DIST - min_dists)**2)
    
    return v_boundary + v_dist

def try_pack(S, seed):
    np.random.seed(seed)
    # Initialize in a small circle to avoid huge initial violations
    r = S * 0.5
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.random.uniform(0, r, N)
    pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    def obj(p):
        return packing_violation(p.reshape(N, 2), S)
        
    res = minimize(obj, pos.flatten(), method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    return res.fun, res.x.reshape(N, 2)

def find_min_S():
    low = 2.5
    high = 5.0
    best_pos = None
    best_S = high
    
    # Binary search
    for _ in range(25):
        mid = (low + high) / 2
        success = False
        for seed in range(10):
            err, pos = try_pack(mid, seed)
            if err < 1e-6:
                success = True
                best_pos = pos
                best_S = mid
                break
        if success:
            high = mid
        else:
            low = mid
            
    return best_S, best_pos

def construct():
    S, positions = find_min_S()
    L = S + 1.0
    
    if positions is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for p in positions:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, (0.0, 0.0), float(L), 0.0
