import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin for 150s limit

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score(points)

def _make_regular_polygon(n, radius=1.0, offset=0.0):
    angles = np.linspace(offset, 2*np.pi + offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_ring_points(n_out, n_in, r, theta):
    """
    Generate points for a two-ring configuration.
    Outer ring: n_out points on unit circle.
    Inner ring: n_in points on circle of radius r, rotated by theta.
    """
    if n_out + n_in != N:
        return None
        
    pts_out = _make_regular_polygon(n_out, 1.0, 0.0)
    pts_in = _make_regular_polygon(n_in, r, theta)
    
    return np.vstack([pts_out, pts_in])

def _neg_ring_score(params, n_out, n_in):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    # Constraints: r > 0, theta in [0, 2*pi/gcd(n_out, n_in)] effectively, 
    # but DE handles bounds. We can just use [0, 2*pi] for theta and let symmetry handle the rest,
    # or tighter bounds for efficiency.
    # For n_out=7, n_in=7, symmetry is 2*pi/7.
    # Let's use [0, 1.0] for r and [0, 2*np.pi] for theta to be safe, 
    # but tighter is better.
    
    if r < 0.01 or r > 0.99:
        return 0.0
        
    points = _generate_ring_points(n_out, n_in, r, theta)
    if points is None:
        return 0.0
        
    s = _score(points)
    return -s

def _optimize_ring_split(n_out, n_in, time_budget):
    """
    Optimize a specific split (n_out, n_in) using DE + Local Polish.
    Returns (score, points)
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    g = np.gcd(n_out, n_in)
    # The configuration repeats every 2*pi/g. We can search in [0, 2*pi/g]
    # However, for simplicity and robustness, let's search [0, 2*pi] 
    # but DE might be slower. Let's use the symmetry to reduce search space.
    theta_max = 2 * np.pi / g
    
    bounds = [(0.05, 0.95), (0.0, theta_max)]
    
    best_score = -np.inf
    best_points = None
    
    # 1. Differential Evolution
    # Estimate time per eval. _score is ~1ms. 
    # DE with popsize 15, maxiter 40 -> 600 evals -> 0.6s. 
    # We have ~30-40s per split if we do 4 splits.
    # Let's be conservative.
    
    popsize = 15
    maxiter = 50
    
    try:
        res = differential_evolution(
            lambda p: _neg_ring_score(p, n_out, n_in),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            polish=True,
            # init='latinhypercube' can be good
        )
        
        if res.fun < 0: # Valid score
            best_score = -res.fun
            best_params = res.x
            best_points = _generate_ring_points(n_out, n_in, best_params[0], best_params[1])
        else:
            best_params = None
            
    except Exception:
        best_params = None
        best_points = None

    # If DE failed or didn't find good, try heuristic starts
    if best_points is None or best_score < 0.5:
        heuristic_starts = []
        for r in [0.3, 0.5, 0.7, 0.9]:
            for theta in [0, theta_max/2, theta_max/3]:
                heuristic_starts.append([r, theta])
        
        for start in heuristic_starts:
            if time.time() > deadline - 10:
                break
            s = -_neg_ring_score(start, n_out, n_in)
            if s > best_score:
                best_score = s
                best_points = _generate_ring_points(n_out, n_in, start[0], start[1])
                best_params = start

    # 2. Local Polish on Full Coordinates
    # Take the best ring configuration and optimize all 28 coordinates.
    # This allows points to leave the rings if beneficial.
    if best_points is not None and time.time() < deadline - 5:
        try:
            # Use Powell or Nelder-Mead. Powell is often more robust for high dim.
            # Nelder-Mead can struggle with 28 dims if not tuned.
            # Let's try Nelder-Mead with small steps first.
            res_opt = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            
            opt_points = res_opt.x.reshape((N, 2))
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
                
        except Exception:
            pass
            
    # 3. Second Polish with different method if time permits
    if best_points is not None and time.time() < deadline - 2:
        try:
            res_opt2 = minimize(
                _neg_score,
                best_points.flatten(),
                method='Powell',
                options={
                    'maxiter': 1000,
                    'maxfev': 3000
                }
            )
            opt_points2 = res_opt2.x.reshape((N, 2))
            opt_score2 = _score(opt_points2)
            
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_points2
        except Exception:
            pass

    return best_score, best_points

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0
    
    np.random.seed(42)
    
    # Define splits to explore
    # (n_out, n_in)
    splits = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
        (10, 4),
        (4, 10)
    ]
    
    # We have ~145 seconds.
    # Let's allocate time to splits.
    # Some splits are more promising (7,7) and (8,6).
    # Give more time to top candidates.
    
    # Initial quick scan to prioritize splits?
    # Or just run them sequentially with time checks.
    
    best_global_score = -np.inf
    best_global_points = None
    
    # Baseline: Regular polygon
    reg_poly = _make_regular_polygon(N)
    s_reg = _score(reg_poly)
    best_global_score = s_reg
    best_global_points = reg_poly.copy()
    
    # Sort splits by heuristic potential? 
    # 7/7 and 8/6 are usually best for this problem type.
    # Let's just iterate and check time.
    
    # We will try to spend ~15-20s per split for top 4 splits.
    # Total budget 145s.
    
    # Re-order splits to put likely best first
    priority_splits = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5)
    ]
    
    # Add remaining splits if time permits
    for split in priority_splits:
        if time.time() > DEADLINE - 30:
            break
            
        n_out, n_in = split
        # Allocate time: ~30s for first, 25s for second, etc?
        # Let's just use a fixed chunk and break if deadline is close.
        time_chunk = 35.0
        
        s, pts = _optimize_ring_split(n_out, n_in, time_chunk)
        
        if pts is not None and s > best_global_score:
            best_global_score = s
            best_global_points = pts
            
        # Check time
        if time.time() > DEADLINE - 20:
            break
            
    # If we have time left, try a few more splits or random restarts
    if time.time() < DEADLINE - 20:
        for split in [(5, 9), (9, 5), (10, 4)]:
            if time.time() > DEADLINE - 15:
                break
            n_out, n_in = split
            time_chunk = 15.0
            s, pts = _optimize_ring_split(n_out, n_in, time_chunk)
            if pts is not None and s > best_global_score:
                best_global_score = s
                best_global_points = pts
                
    # Final Polish on the absolute best
    if best_global_points is not None and time.time() < DEADLINE - 5:
        try:
            # One last high-precision polish
            res_final = minimize(
                _neg_score,
                best_global_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'maxfev': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14
                }
            )
            final_points = res_final.x.reshape((N, 2))
            final_score = _score(final_points)
            if final_score > best_global_score:
                best_global_score = final_score
                best_global_points = final_points
        except Exception:
            pass

    return best_global_points
