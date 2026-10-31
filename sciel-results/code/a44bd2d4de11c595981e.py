import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2

def _min_area(points):
    # points: (N, 2)
    n = points.shape[0]
    # Pre-compute indices for all triples i < j < k
    # There are 165 triples for N=11
    i, j, k = np.indices((n, n, n))
    i = i.flatten()
    j = j.flatten()
    k = k.flatten()
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _penalty(points):
    # points: (N, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    v1 = np.maximum(0, -y)
    # y <= sqrt(3) * x
    v2 = np.maximum(0, y - np.sqrt(3) * x)
    # y <= sqrt(3) * (1 - x)
    v3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(flat_points):
    points = flat_points.reshape(-1, 2)
    area = _min_area(points)
    pen = _penalty(points)
    # We want to maximize area, minimize penalty
    # DE minimizes. So return -area + penalty
    return -area + 100.0 * pen

def _smooth_objective(flat_points, k=50.0):
    points = flat_points.reshape(-1, 2)
    n = points.shape[0]
    i, j, k_idx = np.indices((n, n, n))
    i = i.flatten()
    j = j.flatten()
    k_idx = k_idx.flatten()
    mask = (i < j) & (j < k_idx)
    i = i[mask]
    j = j[mask]
    k_idx = k_idx[mask]
    
    pi = points[i]
    pj = points[j]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: (1/k) * log(sum(exp(-k * areas)))
    # To avoid overflow/underflow, shift by max area?
    # areas are small. -k * areas will be large negative.
    # exp(large negative) -> 0.
    # sum(exp(-k * areas)) might be very small.
    # log(sum) will be negative large.
    # Let's use the standard trick:
    m = np.max(areas)
    # We want min. 
    # -min(A) approx - (1/k) log sum exp(-k A)
    # Let's just use -np.min(areas) for the hard version, 
    # but for L-BFGS-B we need gradient.
    # The soft min is differentiable.
    
    # Actually, L-BFGS-B on the hard min (with subgradients or just finite diff) might be tricky.
    # Let's stick to DE. It is derivative-free and handles non-smooth functions well.
    # I will skip the refinement step to keep it simple and robust, 
    # unless DE is very slow.
    
    # If we do refinement:
    # areas_shifted = areas - m
    # soft_min = m - (1/k) * np.log(np.sum(np.exp(-k * areas_shifted)))
    # return -soft_min + 100 * _penalty(points)
    
    area = np.min(areas)
    pen = _penalty(points)
    return -area + 100.0 * pen

def construct():
    start_time = time.time()
    timeout = 120.0
    
    bounds = [(0, 1), (0, H)] * N
    
    # Use differential evolution
    # popsize=15, maxiter=1000, tol=1e-6
    # We need to monitor time.
    
    def callback(xk, convergence):
        if time.time() - start_time > timeout - 10:
            raise Exception("Timeout")

    best_result = None
    try:
        # DE is stochastic. 
        # We can run it with a fixed seed for reproducibility if needed, but random is fine.
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-5, 
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False, # We will do our own or skip
            callback=callback
        )
        if res.success or res.fun < float('inf'):
            best_result = res.x
    except Exception as e:
        # If DE fails or times out, try a quick local search or just return random?
        # Better to catch timeout and return whatever we have if we stored it.
        # DE doesn't easily return intermediate bests on exception unless we hook it.
        # Let's just rely on the try/except.
        pass

    if best_result is None:
        # Fallback: simple random or grid
        # Let's try a quick random restart with local search?
        # Or just return a valid random configuration.
        # A 3x3 grid + 2 extra points?
        # Let's just return a random valid configuration.
        rng = np.random.default_rng(42)
        points = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = H * v
                    points.append([x, y])
                    break
        best_result = np.array(points).flatten()
        
    points = best_result.reshape(-1, 2)
    return points

# EVOLVE-BLOCK-END
