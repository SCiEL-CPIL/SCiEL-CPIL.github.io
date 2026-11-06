"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(points):
    """Compute the minimum radius for a set of points."""
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    n = len(points)
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    return min(min_boundary, min_pair / 2.0)


def _generate_hexagonal_initial():
    """Generate a hexagonal lattice seed."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def _generate_ring_initial():
    """Generate a ring-based initial configuration."""
    points = np.zeros((N, 2))
    # Center point
    points[0] = [0, 0]
    remaining = N - 1
    ring = 1
    while remaining > 0:
        # Number of points in this ring
        n_in_ring = min(remaining, ring * 6)
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False) + np.pi / (ring * 6)
        radius = ring * 0.19
        for i in range(n_in_ring):
            idx = ring * 3 + i  # approximate index
            if idx < N:
                points[idx] = [radius * np.cos(angles[i]), radius * np.sin(angles[i])]
        remaining -= n_in_ring
        ring += 1
    return points


def _soft_min_objective(x, beta=50.0):
    """
    Smooth soft-min objective for maximizing the minimum radius.
    Returns negative of the soft-min radius so we minimize it.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary distances
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise distances / 2
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - points[j]) / 2.0
            if d < min_pair:
                min_pair = d
    
    # Soft min of boundary distances and pair distances
    all_vals = np.concatenate([dist_boundary, [min_pair]])
    
    # Use log-sum-exp for soft min
    max_val = np.max(all_vals)
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - max_val)))) / beta + max_val
    soft_min = max_val - log_sum / beta
    
    return -soft_min


def _optimize_physics(seed, max_iter=2000, lr=0.01, beta=50.0):
    """
    Physics-based relaxation using gradient descent on a smooth energy landscape.
    """
    x = seed.flatten().copy()
    n = N
    
    for iteration in range(max_iter):
        # Compute gradients numerically or analytically
        pts = x.reshape(-1, 2)
        
        # Compute objective
        obj = _soft_min_objective(x, beta)
        
        # Compute gradient numerically
        grad = np.zeros_like(x)
        eps = 1e-6
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (_soft_min_objective(x_plus, beta) - obj) / eps
        
        # Gradient descent
        x -= lr * grad
        
        # Adaptive learning rate
        if iteration % 100 == 0 and iteration > 0:
            lr *= 0.95
    
    return x.reshape(-1, 2)


def _optimize_lbfgsb(seed, max_iter=1000, beta=50.0):
    """Optimize using L-BFGS-B with smooth soft-min objective."""
    x0 = seed.flatten()
    
    def objective(x):
        return _soft_min_objective(x, beta)
    
    res = minimize(objective, x0, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12})
    return res.x.reshape(-1, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1.0
    
    # Try multiple seeds
    seeds = []
    
    # Hexagonal seed
    try:
        hex_seed = _generate_hexagonal_initial()
        seeds.append(hex_seed)
    except Exception:
        pass
    
    # Ring-based seed
    try:
        ring_seed = _generate_ring_initial()
        seeds.append(ring_seed)
    except Exception:
        pass
    
    # Random seeds
    np.random.seed(42)
    for _ in range(3):
        try:
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.9, N))
            rand_seed = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            seeds.append(rand_seed)
        except Exception:
            pass
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Try L-BFGS-B optimization
        try:
            optimized = _optimize_lbfgsb(seed, max_iter=500, beta=50.0)
            score = _score(optimized)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            pass
        
        # If time permits, try physics-based relaxation
        if time.time() < deadline - 30:
            try:
                optimized2 = _optimize_physics(seed, max_iter=1000, lr=0.005, beta=50.0)
                score2 = _score(optimized2)
                if score2 > best_score:
                    best_score = score2
                    best_points = optimized2
            except Exception:
                pass
    
    if best_points is None:
        # Fallback to hexagonal
        best_points = _generate_hexagonal_initial()
    
    return best_points
# EVOLVE-BLOCK-END
