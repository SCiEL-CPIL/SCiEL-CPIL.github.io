import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-10:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(x_flat):
    """Negative score for minimization."""
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(ratios, beta):
    """Compute soft minimum of ratios using log-sum-exp."""
    r_min = np.min(ratios)
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return r_min + log_sum / beta


def _soft_score(x_flat, beta):
    """Smooth score using soft-min of area ratios."""
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 0.0
    ratios = areas / hull_area
    return _soft_min_value(ratios, beta)


def _neg_soft_score(x_flat, beta):
    """Negative soft score for minimization."""
    s = _soft_score(x_flat, beta)
    if s <= 0:
        return 1e10
    return -s


def _seed_regular():
    """Regular 13-gon."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _seed_two_ring():
    """7 outer + 6 inner ring."""
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner_angles = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([0.55 * np.cos(inner_angles), 0.55 * np.sin(inner_angles)])
    return np.vstack([outer, inner])


def _seed_three_ring():
    """5 outer + 4 middle + 4 inner."""
    o_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    m_angles = np.linspace(np.pi/5, 2 * np.pi + np.pi/5, 4, endpoint=False)
    i_angles = np.linspace(np.pi/4, 2 * np.pi + np.pi/4, 4, endpoint=False)
    o = np.column_stack([np.cos(o_angles), np.sin(o_angles)])
    m = np.column_stack([0.65 * np.cos(m_angles), 0.65 * np.sin(m_angles)])
    i = np.column_stack([0.35 * np.cos(i_angles), 0.35 * np.sin(i_angles)])
    return np.vstack([o, m, i])


def _seed_quasi_regular():
    """Quasi-regular: 13 points with slight radial perturbation."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = 1.0 + 0.15 * np.sin(3 * angles)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _annealed_soft_minimize(x0, beta_start=10.0, beta_end=200.0, n_steps=30, maxiter_per_step=80):
    """Anneal beta from start to end, optimizing soft-min objective at each step."""
    x = x0.copy()
    betas = np.linspace(beta_start, beta_end, n_steps)
    for beta in betas:
        try:
            result = minimize(
                _neg_soft_score,
                x,
                args=(beta,),
                method='Nelder-Mead',
                options={'maxiter': maxiter_per_step, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            x = result.x
        except Exception:
            pass
    return x


def _local_search(points, n_iters=500, step_size=0.01):
    """Simple coordinate-wise local search."""
    x = points.flatten().copy()
    best_f = _neg_score(x)
    for it in range(n_iters):
        improved = False
        for d in range(2 * N):
            for delta in [step_size, -step_size]:
                x_try = x.copy()
                x_try[d] += delta
                f_try = _neg_score(x_try)
                if f_try < best_f:
                    x = x_try
                    best_f = f_try
                    improved = True
                    break
        if not improved:
            step_size *= 0.7
            if step_size < 1e-8:
                break
    return x.reshape(N, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin against 150s kill
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate diverse seeds
    seeds = [
        _seed_regular(),
        _seed_two_ring(),
        _seed_three_ring(),
        _seed_quasi_regular(),
    ]
    
    # Add some randomized seeds
    rng = np.random.default_rng(42)
    for _ in range(6):
        # Random ring-based configurations
        n_outer = rng.integers(5, 10)
        n_inner = N - n_outer
        if n_inner > 0:
            o_angles = rng.uniform(0, 2 * np.pi, n_outer)
            o_angles.sort()
            i_angles = rng.uniform(0, 2 * np.pi, n_inner)
            i_angles.sort()
            r_in = rng.uniform(0.3, 0.7)
            o = np.column_stack([np.cos(o_angles), np.sin(o_angles)])
            i = np.column_stack([r_in * np.cos(i_angles), r_in * np.sin(i_angles)])
            seeds.append(np.vstack([o, i]))
        else:
            angles = rng.uniform(0, 2 * np.pi, N)
            angles.sort()
            seeds.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Evaluate all seeds
    for seed in seeds:
        _update_best(seed)
    
    # Annealed optimization from each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 20:
            break
        try:
            x_opt = _annealed_soft_minimize(
                seed.flatten(),
                beta_start=10.0,
                beta_end=150.0,
                n_steps=20,
                maxiter_per_step=50
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass
    
    # Local search from best
    if best_points is not None and time.time() < deadline - 15:
        try:
            ls_points = _local_search(best_points, n_iters=800, step_size=0.02)
            _update_best(ls_points)
        except Exception:
            pass
    
    # Final hard-objective polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # More annealed restarts if time permits
    while time.time() < deadline - 8:
        seed_idx = rng.integers(0, len(seeds))
        base_seed = seeds[seed_idx].copy()
        # Add small perturbation
        noise = rng.normal(0, 0.05, (N, 2))
        perturbed = base_seed + noise
        try:
            x_opt = _annealed_soft_minimize(
                perturbed.flatten(),
                beta_start=20.0,
                beta_end=200.0,
                n_steps=15,
                maxiter_per_step=40
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
