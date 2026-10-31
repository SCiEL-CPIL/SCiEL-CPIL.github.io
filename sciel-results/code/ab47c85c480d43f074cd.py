import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    # Prevent overflow in exp
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum

def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area

def _neg_soft_score(points, temperature):
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s

def _seed_regular():
    """Regular 13-gon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])

def _seed_two_ring(n_out, r_inner, angle_offset):
    """Two-ring configuration."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])

def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    """Three-ring configuration."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts

def _seed_quasi_regular():
    """Quasi-regular: regular polygon with small random perturbations."""
    rng = np.random.RandomState(42)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.03 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])

def _local_optimize(points, deadline, temp=0.01):
    """
    Perform local optimization using L-BFGS-B on the soft-min objective.
    Returns the optimized points and the score using the true objective.
    """
    if time.time() >= deadline:
        return points, _score(points)
    
    def obj(x):
        pts = x.reshape(N, 2)
        return _neg_soft_score(pts, temp)
    
    try:
        res = minimize(
            obj, points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        cand = res.x.reshape(N, 2)
        return cand, _score(cand)
    except Exception:
        return points, _score(points)

def _basinhopping_optimize(points, deadline, temp=0.05, n_iter=50):
    """
    Use basin hopping to escape local minima.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    rng = np.random.RandomState(42)
    
    def step(x):
        """Random perturbation step."""
        dx = rng.randn(2 * N) * 0.1
        return x + dx
    
    def objective(x):
        """Objective function for basin hopping (uses soft score for smoothness)."""
        pts = x.reshape(N, 2)
        return _neg_soft_score(pts, temp)
    
    def local_opt(x):
        """Local optimization step."""
        pts = x.reshape(N, 2)
        opt_pts, _ = _local_optimize(pts, deadline, temp)
        return opt_pts.flatten(), _neg_soft_score(opt_pts, temp)
    
    try:
        bh = basinhopping(
            objective,
            points.flatten(),
            niter=n_iter,
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}},
            stepsize=0.1,
            T=100.0,
            accept_test=lambda new_f, old_f: new_f < old_f or rng.rand() < np.exp(-(new_f - old_f) / 100.0),
            interval=10,
            disp=False
        )
        
        cand = bh.x.reshape(N, 2)
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand.copy()
    except Exception:
        pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    np.random.seed(42)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Generate diverse seeds
    seeds = []
    
    # Regular 13-gon
    seeds.append(_seed_regular())
    
    # Center + 12-gon
    a12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer12 = np.column_stack([np.cos(a12), np.sin(a12)])
    center = np.array([[0.0, 0.0]])
    seeds.append(np.vstack([outer12, center]))
    
    # Quasi-regular
    seeds.append(_seed_quasi_regular())
    
    # Two rings: 6 outer, 7 inner
    for r_in in [0.3, 0.5, 0.7]:
        for off in [0.0, np.pi/6, np.pi/3]:
            seeds.append(_seed_two_ring(6, r_in, off))
            
    # Two rings: 7 outer, 6 inner
    for r_in in [0.3, 0.5, 0.7]:
        for off in [0.0, np.pi/7, np.pi/14]:
            seeds.append(_seed_two_ring(7, r_in, off))
            
    # Two rings: 5 outer, 8 inner
    for r_in in [0.4, 0.6]:
        for off in [0.0, np.pi/5]:
            seeds.append(_seed_two_ring(5, r_in, off))
            
    # Three rings: 5+4+4
    seeds.append(_seed_three_ring(5, 4, 4, 0.6, 0.3, 0.0, 0.0))
    seeds.append(_seed_three_ring(5, 4, 4, 0.5, 0.25, np.pi/5, np.pi/4))
    
    # Three rings: 6+4+3
    seeds.append(_seed_three_ring(6, 4, 3, 0.6, 0.3, 0.0, 0.0))
    
    # Three rings: 7+3+3
    seeds.append(_seed_three_ring(7, 3, 3, 0.6, 0.3, 0.0, 0.0))
    
    # Three rings: 4+4+5
    seeds.append(_seed_three_ring(4, 4, 5, 0.5, 0.25, 0.0, np.pi/4))
    
    # Evaluate all seeds and keep track of top performers
    seed_scores = []
    for seed in seeds:
        s = _score(seed)
        seed_scores.append((s, seed))
    
    # Sort by score descending
    seed_scores.sort(key=lambda x: -x[0])
    
    # Update best
    if seed_scores:
        _update(seed_scores[0][1])
    
    # Optimize top seeds with Basin Hopping
    top_k = min(5, len(seed_scores))
    for i in range(top_k):
        if time.time() > deadline - 30:
            break
        
        s_seed, seed_pts = seed_scores[i]
        
        # Stage 1: Basin Hopping
        if time.time() < deadline - 20:
            try:
                pts, s = _basinhopping_optimize(seed_pts, deadline - 10, temp=0.05, n_iter=30)
                _update(pts)
            except Exception:
                pass
        
        # Stage 2: Local Optimization with finer temperature
        if time.time() < deadline - 10:
            try:
                pts, s = _local_optimize(best_points, deadline - 5, temp=0.01)
                _update(pts)
            except Exception:
                pass
        
        # Stage 3: Final Polish with Nelder-Mead on true objective
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    lambda x: _neg_score(x.reshape(N, 2)),
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                _update(res.x.reshape(N, 2))
            except Exception:
                pass
    
    # Final global polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
            
    if best_points is None:
        best_points = _seed_regular()
        
    return best_points
