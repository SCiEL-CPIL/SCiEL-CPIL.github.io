import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin

def _compute_score(pts):
    """Compute the valid packing radius."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check boundary
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0
    r_boundary = 1.0 - np.max(dists_origin)
    
    # Check pairwise distances
    if N < 2:
        return r_boundary
        
    # Vectorized pairwise distance calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, r_boundary)

def _soft_min_obj_and_grad(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    Minimizes -soft_min(constraints).
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Pairwise distances
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = pts[idx_i] - pts[idx_j]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=1)) / 2.0
    
    # Boundary distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    # Combine
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min
    x_min = np.min(all_dists)
    exp_vals = np.exp(-(all_dists - x_min) / beta)
    sum_exp = np.sum(exp_vals)
    soft_min_val = x_min - beta * np.log(sum_exp)
    
    # Gradient of soft_min w.r.t. pts
    grad = np.zeros_like(pts)
    weights = exp_vals / sum_exp
    
    # Pairwise gradient
    for k in range(len(pair_dists)):
        i, j = idx_i[k], idx_j[k]
        d = diff[k]
        norm = np.sqrt(np.sum(d ** 2))
        if norm < 1e-12:
            continue
        w = weights[k]
        factor = w * 0.5 / norm
        # d(soft_min)/d(pts[i]) = -w * 0.5 * d / norm
        # d(soft_min)/d(pts[j]) = -w * (-0.5 * d / norm)
        # We are minimizing -soft_min, so we negate the gradient of soft_min
        # grad_minimize = - grad_soft_min
        # grad_soft_min[i] = -w * 0.5 * d / norm
        # grad_minimize[i] = w * 0.5 * d / norm
        grad[i] += factor * d
        grad[j] -= factor * d
        
    # Boundary gradient
    for k in range(n):
        i = k
        r = radii[i]
        if r < 1e-12:
            continue
        w = weights[len(pair_dists) + k]
        # x_k = 1 - |p_i|
        # d(x_k)/d(p_i) = -p_i / |p_i|
        # d(soft_min)/d(p_i) = -w * (-p_i / r) = w * p_i / r
        # grad_minimize = - grad_soft_min = -w * p_i / r
        grad[i] -= w * pts[i] / r
        
    return -soft_min_val, grad.flatten()

def _init_rings():
    """Initialize with concentric rings: 1, 6, 12, 18, 13."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    
    # Ring 1: 6 points
    r1 = 0.25
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
        
    # Ring 2: 12 points
    r2 = 0.5
    for i in range(12):
        a = 2 * np.pi * i / 12 + np.pi/12
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
        
    # Ring 3: 18 points
    r3 = 0.75
    for i in range(18):
        a = 2 * np.pi * i / 18
        pts.append([r3 * np.cos(a), r3 * np.sin(a)])
        
    # Ring 4: 13 points (partial)
    r4 = 0.95
    # Spread 13 points evenly
    for i in range(13):
        a = 2 * np.pi * i / 13
        pts.append([r4 * np.cos(a), r4 * np.sin(a)])
        
    return np.array(pts[:N], dtype=np.float64)

def _init_hex():
    """Initialize with a hexagonal lattice."""
    pts = []
    spacing = 0.2
    # Generate hex lattice points
    for i in range(-10, 10):
        for j in range(-10, 10):
            x = i * spacing
            y = j * spacing * np.sqrt(3) + (i % 2) * spacing * np.sqrt(3) / 2
            if x*x + y*y <= 0.95**2:
                pts.append([x, y])
                
    pts = np.array(pts)
    # If too many, select closest to ideal radius? Or just take first N?
    # Better: sort by distance from center and take those that fit well?
    # For now, just take first N after shuffling to break symmetry
    np.random.shuffle(pts)
    return pts[:N]

def _init_random():
    """Random initialization."""
    pts = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y <= 0.9**2:
                pts.append([x, y])
                break
    return np.array(pts, dtype=np.float64)

def _optimize_from_start(pts):
    """Run optimization from a given start point."""
    x0 = pts.flatten()
    best_x = x0.copy()
    best_score = _compute_score(pts)
    
    # Adaptive beta
    betas = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for beta in betas:
        if time.time() > _DEADLINE - 5:
            break
            
        def obj(x):
            return _soft_min_obj_and_grad(x, beta)[0]
        
        def grad(x):
            return _soft_min_obj_and_grad(x, beta)[1]
            
        try:
            result = minimize(
                obj,
                best_x,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            new_x = result.x
            new_pts = new_x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_x = new_x.copy()
        except Exception:
            pass
            
    return best_x.reshape(-1, 2), best_score

def _polish(pts):
    """Local search polish."""
    pts = pts.copy()
    n = len(pts)
    improved = True
    max_iter = 100
    iter_count = 0
    
    while improved and iter_count < max_iter and time.time() < _DEADLINE - 2:
        improved = False
        iter_count += 1
        for i in range(n):
            # Try moving point i in small steps
            for dx in [-0.01, 0.01]:
                for dy in [-0.01, 0.01]:
                    new_pts = pts.copy()
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    score = _compute_score(new_pts)
                    if score > _compute_score(pts) + 1e-8:
                        pts = new_pts
                        improved = True
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_pts = None
    best_score = 0.0
    
    # Try different initializations
    inits = [
        _init_rings(),
        _init_hex(),
        _init_random()
    ]
    
    # Add more random restarts if time permits
    for i in range(5):
        if time.time() > deadline - 20:
            break
        inits.append(_init_random())
        
    for init_pts in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            optimized_pts, score = _optimize_from_start(init_pts)
            if score > best_score:
                best_score = score
                best_pts = optimized_pts
        except Exception:
            pass
            
    if best_pts is None:
        return _init_rings()
        
    # Final polish
    if time.time() < deadline - 5:
        best_pts = _polish(best_pts)
        best_score = _compute_score(best_pts)
        
    return best_pts
