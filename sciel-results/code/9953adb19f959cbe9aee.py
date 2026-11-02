import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def _pairwise_dists_sq(pts):
    """Compute all pairwise squared distances."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    # Get upper triangle
    iu = np.triu_indices(n, k=1)
    return d2[iu]

def _score(pts):
    """Compute (max/min)^2 ratio."""
    d2 = _pairwise_dists_sq(pts)
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _soft_objective(pts, beta=50.0):
    """Smooth surrogate using log-sum-exp for max and min."""
    d2 = _pairwise_dists_sq(pts)
    # Soft max of d2
    m = np.max(d2)
    soft_max = m + np.log(np.sum(np.exp(d2 - m))) / len(d2)
    # Soft min of d2: -soft_max of -d2
    mn = np.min(d2)
    soft_min = mn - np.log(np.sum(np.exp(mn - d2))) / len(d2)
    if soft_min < 1e-12:
        return 1e12
    ratio = soft_max / soft_min
    return ratio * ratio

def _fibonacci_sphere(n, radius=1.0):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden
        points[i] = radius * np.array([r * np.cos(theta), r * np.sin(theta), z])
    return points

def _structured_init_14():
    """14 points: cube vertices + octahedron vertices, scaled."""
    # Cube vertices at (±1, ±1, ±1)
    cube = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
    # Octahedron vertices at (±2, 0, 0), (0, ±2, 0), (0, 0, ±2)
    octa = np.array([
        [2, 0, 0], [-2, 0, 0],
        [0, 2, 0], [0, -2, 0],
        [0, 0, 2], [0, 0, -2]
    ], dtype=float)
    pts = np.vstack([cube, octa])
    return pts

def _random_init(scale=1.0):
    return np.random.randn(N, D) * scale

def _sa_optimize(pts0, max_iters=50000, deadline=None, rng=None):
    """Simulated annealing with adaptive step size."""
    if rng is None:
        rng = np.random.default_rng()
    
    pts = pts0.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    T = 0.1
    T_min = 1e-6
    step_size = 0.1
    n_accepted = 0
    n_total = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Propose move
        idx = rng.integers(N)
        delta = rng.normal(0, step_size, size=D)
        new_pts = pts.copy()
        new_pts[idx] += delta
        
        new_score = _score(new_pts)
        delta_score = new_score - current_score
        
        n_total += 1
        if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
            pts = new_pts
            current_score = new_score
            n_accepted += 1
            if current_score < best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        # Adaptive step size
        if n_total % 100 == 0:
            accept_rate = n_accepted / n_total
            if accept_rate > 0.5:
                step_size *= 1.1
            elif accept_rate < 0.2:
                step_size *= 0.9
            step_size = np.clip(step_size, 1e-4, 1.0)
            n_accepted = 0
            n_total = 0
        
        T = max(T_min, T * 0.9995)
    
    return best_pts, best_score

def _polish(pts, deadline=None):
    """Polish with scipy minimize using smooth surrogate."""
    def obj(x):
        if deadline is not None and time.time() > deadline:
            return 1e12
        return _soft_objective(x.reshape(N, D), beta=50.0)
    
    x0 = pts.flatten()
    try:
        res = minimize(obj, x0, method='L-BFGS-B', 
                       options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
        if res.success or res.fun < _score(pts):
            return res.x.reshape(N, D)
    except:
        pass
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Multiple initializations
    inits = []
    
    # 1. Structured: cube + octahedron
    inits.append(_structured_init_14())
    
    # 2. Fibonacci sphere
    inits.append(_fibonacci_sphere(N, radius=1.0))
    
    # 3. Random initializations at different scales
    for scale in [0.5, 1.0, 2.0]:
        inits.append(_random_init(scale))
    
    # 4. Two-layer hexagonal arrangement
    hex1 = []
    hex2 = []
    # 7 points in each layer
    for i in range(7):
        angle = 2 * np.pi * i / 6 if i < 6 else 0
        r = 1.0 if i < 6 else 0.3
        hex1.append([r * np.cos(angle), r * np.sin(angle), 0.5])
        hex2.append([r * np.cos(angle + np.pi/6), r * np.sin(angle + np.pi/6), -0.5])
    inits.append(np.array(hex1 + hex2))
    
    best_pts = None
    best_score = np.inf
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Run SA from this initialization
        pts, score = _sa_optimize(init, max_iters=30000, deadline=deadline, rng=rng)
        
        if score < best_score:
            best_score = score
            best_pts = pts
        
        # Polish the best
        if time.time() < deadline - 20:
            polished = _polish(pts, deadline=deadline)
            pol_score = _score(polished)
            if pol_score < best_score:
                best_score = pol_score
                best_pts = polished
    
    # Final polish on best
    if best_pts is not None and time.time() < deadline - 10:
        final = _polish(best_pts, deadline=deadline)
        final_score = _score(final)
        if final_score < best_score:
            best_pts = final
            best_score = final_score
    
    if best_pts is None:
        best_pts = inits[0]
    
    return best_pts
