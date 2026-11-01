"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 200
DEADLINE = 120.0  # seconds, leave margin before 150s kill


def _score(pts):
    """Compute the packing radius for a set of points."""
    if len(pts) < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # Boundary distances
    bounds = 0.5 - np.abs(pts)
    min_bound = np.min(bounds)
    return 0.5 * min(min_pair, 2 * min_bound)


def _hex_init(n):
    """Create a hexagonal lattice initialization."""
    # Find best hex arrangement
    best_pts = None
    best_score = -1
    
    for rows in range(14, 20):
        for cols in range(14, 20):
            if rows * cols < n:
                continue
            dx = 1.0 / cols
            dy = 1.0 / rows
            # Hex offset
            offset = dx * 0.5
            pts = []
            for i in range(rows):
                for j in range(cols):
                    x = (j + (i % 2) * 0.5) * dx - 0.5 + dx/2
                    y = i * dy - 0.5 + dy/2
                    pts.append([x, y])
            pts = np.array(pts[:n])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:n]
    
    return best_pts


def _nlp_optimize(pts0, max_time=30.0):
    """
    Use SLSQP to maximize the minimum distance via NLP.
    We maximize t subject to:
      - min_pairwise_dist >= 2*t
      - min_boundary_dist >= t
    Equivalently, minimize -t.
    """
    start_time = time.time()
    n = len(pts0)
    x0 = np.concatenate([pts0.flatten(), [0.01]])  # last element is t
    t0 = x0[-1]
    
    def objective(x):
        return -x[-1]  # minimize -t
    
    def constraints(x):
        pts = x[:-1].reshape(-1, 2)
        t = x[-1]
        cons = []
        # Boundary constraints: each point must be at least t from boundary
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 0.5 - np.abs(x[2*i]) - x[-1]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 0.5 - np.abs(x[2*i+1]) - x[-1]})
        
        # Pairwise distance constraints: dist >= 2*t
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sum((x[2*i:2*i+2] - x[2*j:2*j+2])**2) - (2*x[-1])**2
                })
        return cons
    
    # This is too many constraints for SLSQP with N=200 (19900+ constraints)
    # Instead, use a simpler approach: local search with penalty
    
    # Use a simpler local optimization: random perturbation + acceptance
    best_pts = pts0.copy()
    best_score = _score(best_pts)
    step_size = 0.005
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < max_time:
        # Try random perturbation
        candidate = best_pts.copy()
        idx = rng.randint(n)
        candidate[idx] += rng.normal(0, step_size, 2)
        
        # Clamp to valid range
        candidate[idx] = np.clip(candidate[idx], -0.5 + 1e-6, 0.5 - 1e-6)
        
        s = _score(candidate)
        if s > best_score:
            best_pts = candidate
            best_score = s
            step_size = max(0.001, step_size * 0.95)
        else:
            step_size = min(0.01, step_size * 1.05)
    
    return best_pts, best_score


def _local_search(pts, max_time=60.0):
    """Local search with decreasing step size."""
    start_time = time.time()
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    step_size = 0.01
    rng = np.random.RandomState(123)
    stall_count = 0
    max_stall = 1000
    
    while time.time() - start_time < max_time:
        improved = False
        for _ in range(50):
            if time.time() - start_time >= max_time:
                break
            candidate = best_pts.copy()
            idx = rng.randint(n)
            candidate[idx] += rng.normal(0, step_size, 2)
            candidate[idx] = np.clip(candidate[idx], -0.5 + 1e-6, 0.5 - 1e-6)
            
            s = _score(candidate)
            if s > best_score + 1e-10:
                best_pts = candidate
                best_score = s
                improved = True
                stall_count = 0
                step_size = max(0.001, step_size * 0.98)
        
        if not improved:
            stall_count += 1
            step_size = max(0.0005, step_size * 0.9)
            if stall_count > max_stall:
                break
    
    return best_pts, best_score


def construct():
    global DEADLINE
    start_time = time.time()
    
    # Initialize with hexagonal lattice
    pts = _hex_init(N)
    
    # Local search with time budget
    remaining = DEADLINE - (time.time() - start_time) - 10  # 10s safety margin
    if remaining > 5:
        pts, score = _local_search(pts, max_time=remaining)
    
    # Ensure validity
    pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return pts
# EVOLVE-BLOCK-END
