import numpy as np
import time
from scipy.optimize import minimize
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
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
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
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular(jump=1.0):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = np.ones(N)
    r += 0.02 * np.random.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _local_search(points, deadline, n_iter=5000):
    """Direct coordinate perturbation local search."""
    best = points.copy()
    best_s = _score(best)
    current = points.copy()
    current_s = best_s
    sigma = 0.05
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Generate candidate by perturbing one or two coordinates
        n_pert = np.random.randint(1, 4)  # 1 to 3 coordinates
        idx = np.random.choice(N * 2, n_pert, replace=False)
        cand = current.copy()
        for i in idx:
            p = i // 2
            c = i % 2
            cand[p, c] += sigma * np.random.randn()
        
        s = _score(cand)
        if s > current_s:
            current = cand.copy()
            current_s = s
            if s > best_s:
                best = cand.copy()
                best_s = s
            sigma *= 1.05  # grow slightly
        else:
            sigma *= 0.95  # shrink
            sigma = max(sigma, 0.001)
    
    return best, best_s


def _annealed_local_search(points, deadline, max_temp=0.1, min_temp=1e-3, n_steps=10):
    """Anneal perturbation scale and do local search at each step."""
    best = points.copy()
    best_s = _score(best)
    
    # Geometric temperature schedule
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() > deadline - 2:
            break
        
        # Do a short local search with perturbation scale ~ temp
        n_iter = max(100, int(2000 * temp / max_temp))
        cand, s = _local_search(best, deadline - 1, n_iter=n_iter)
        if s > best_s:
            best = cand.copy()
            best_s = s
    
    return best, best_s


def _try_optimize_from_seed(seed, deadline):
    """Try multiple optimization strategies from a seed, return best."""
    best = seed.copy()
    best_s = _score(seed)
    
    # 1. Direct local search
    if time.time() < deadline - 5:
        cand, s = _local_search(seed, deadline - 5, n_iter=3000)
        if s > best_s:
            best = cand.copy()
            best_s = s
    
    # 2. Annealed local search
    if time.time() < deadline - 3:
        cand, s = _annealed_local_search(best, deadline - 3, max_temp=0.1, min_temp=1e-3, n_steps=8)
        if s > best_s:
            best = cand.copy()
            best_s = s
    
    # 3. L-BFGS-B polish on smooth objective
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), 0.01),
                best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best = cand.copy()
                best_s = s
        except Exception:
            pass
    
    # 4. Final local search polish
    if time.time() < deadline - 1:
        cand, s = _local_search(best, deadline - 1, n_iter=2000)
        if s > best_s:
            best = cand.copy()
            best_s = s
    
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill
    
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
    
    # ---------- Generate diverse seeds ----------
    seeds = []
    
    # Regular 13-gon
    seeds.append(_seed_regular())
    
    # Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.5, 0.7]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2 * n_out)]:
                seeds.append(_seed_two_ring(n_out, r_in, a_off))
    
    # Three-ring configurations
    three_ring_configs = [
        (5, 4, 4, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        (7, 4, 2, 0.6, 0.3, 0.0, 0.0),
        (6, 5, 2, 0.5, 0.25, np.pi / 6, 0.0),
    ]
    for (n1, n2, n3, r2, r3, o2, o3) in three_ring_configs:
        seeds.append(_seed_three_ring(n1, n2, n3, r2, r3, o2, o3))
    
    # Quasi-regular with random perturbations
    for _ in range(5):
        seeds.append(_seed_quasi_regular())
    
    # Random seeds
    for _ in range(10):
        pts = np.random.randn(N, 2) * 0.5
        seeds.append(pts)
    
    # ---------- Evaluate and optimize from each seed ----------
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        s = _score(seed)
        if s > best_score:
            best_score = s
            best_points = seed.copy()
        
        # Optimize from this seed
        time_per_seed = max(2.0, (deadline - 10 - time.time()) / max(1, len(seeds) - i))
        cand, s = _try_optimize_from_seed(seed, start_time + time_per_seed + time.time() - start_time)
        _update(cand)
    
    if best_points is None:
        best_points = _seed_regular()
        best_score = _score(best_points)
    
    # ---------- Final refinement on the best solution ----------
    # Multiple rounds of local search with decreasing perturbation
    for round_num in range(3):
        if time.time() > deadline - 5:
            break
        
        sigma_max = 0.1 / (round_num + 1)
        cand, s = _local_search(best_points, deadline - 5, n_iter=5000)
        _update(cand)
    
    # ---------- Final L-BFGS-B polish ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # ---------- Final ultra-fine local search ----------
    if best_points is not None and time.time() < deadline - 1:
        cand, s = _local_search(best_points, deadline - 1, n_iter=3000)
        _update(cand)
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
