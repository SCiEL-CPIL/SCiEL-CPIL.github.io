"""
Heilbronn triangles: maximize min triangle area for 11 points in unit equilateral triangle.
Strategy: aggressive SA with multi-point moves + annealed gradient refinement.
"""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0
_EPS = 1e-9

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]


def _areas_vec(pts):
    """All 165 triangle areas (vectorized)."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _min_area(pts):
    return float(np.min(_areas_vec(pts)))


def _clamp_all(pts):
    """Vectorized clamp to triangle."""
    pts = np.array(pts, dtype=float).copy()
    x = pts[:, 0]
    y = pts[:, 1]
    y = np.clip(y, _EPS, _H - _EPS)
    x_min = y / _SQRT3 + _EPS
    x_max = 1.0 - y / _SQRT3 - _EPS
    x = np.clip(x, x_min, x_max)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _clamp_one(x, y):
    """Clamp a single point to the triangle."""
    y = min(max(y, _EPS), _H - _EPS)
    x_min = y / _SQRT3 + _EPS
    x_max = 1.0 - y / _SQRT3 - _EPS
    x = min(max(x, x_min), x_max)
    return x, y


def _logsumexp_neg(pts, beta):
    """
    Smooth approximation of -min(areas) using log-sum-exp.
    As beta -> inf, this approaches -min(areas).
    """
    areas = _areas_vec(pts)
    neg_beta_areas = -beta * areas
    m = np.max(neg_beta_areas)
    if m > 500:
        neg_beta_areas = neg_beta_areas - m
        m = 0.0
    s = np.sum(np.exp(neg_beta_areas))
    if s < 1e-300:
        return 1e30
    return (m + np.log(s)) / beta


def _objective_lse(params, beta):
    """Smooth objective: log-sum-exp approximation of -min_area."""
    pts = _clamp_all(params.reshape(N, 2))
    return _logsumexp_neg(pts, beta)


def _objective_exact(params):
    """Exact non-smooth objective."""
    pts = _clamp_all(params.reshape(N, 2))
    return -_min_area(pts)


def _gradient_refine(pts, beta_start=50.0, beta_end=3000.0, n_steps=20, time_limit=10.0):
    """
    Anneal beta from low to high while doing gradient steps.
    """
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    start_time = time.time()
    best_x = x0.copy()
    best_score = _min_area(best_x.reshape(N, 2))
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        frac = step / max(1, n_steps - 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        try:
            res = minimize(
                _objective_lse,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000},
            )
            cand = _clamp_all(res.x.reshape(N, 2))
            s = _min_area(cand)
            if s > best_score:
                best_score = s
                best_x = cand.flatten()
                x0 = best_x.copy()
        except Exception:
            break
    
    return _clamp_all(best_x.reshape(N, 2)), best_score


def _simulated_annealing(start_points, time_limit=10.0, seed=None):
    """
    Simulated Annealing with adaptive step size for global exploration.
    Uses multi-point moves in early phase for faster exploration.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    points = _clamp_all(np.array(start_points, dtype=float))
    current_score = _min_area(points)
    
    best_points = points.copy()
    best_score = current_score
    
    T = 0.03
    T_min = 1e-8
    alpha = 0.9997
    step_size = 0.15
    
    start_time = time.time()
    total_time = time_limit
    n_iter = 0
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        n_iter += 1
        frac = n_iter / max(1, total_time * 200)  # approximate progress
        
        # In early phase, move 2-3 points at once for better exploration
        if frac < 0.3:
            n_move = int(rng.integers(2, 4))
            idxs = rng.choice(N, size=n_move, replace=False)
        elif frac < 0.7:
            n_move = int(rng.integers(1, 3))
            idxs = rng.choice(N, size=n_move, replace=False)
        else:
            n_move = 1
            idxs = np.array([rng.integers(N)])
        
        old_points = points[idxs].copy()
        
        for idx in idxs:
            noise = rng.normal(0, step_size, 2)
            new_point = points[idx] + noise
            nx, ny = _clamp_one(new_point[0], new_point[1])
            points[idx] = np.array([nx, ny])
        
        new_score = _min_area(points)
        
        delta = new_score - current_score
        if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[idxs] = old_points
            
        T *= alpha
        step_size *= 0.99985
        
    return best_points, best_score


def _nelder_mead_refine(pts, time_limit=5.0):
    """Nelder-Mead as a secondary refiner."""
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    start_time = time.time()
    best_pts = x0.reshape(N, 2).copy()
    best_score = _min_area(best_pts)
    try:
        res = minimize(
            _objective_exact,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 300, 'maxfev': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                     'adaptive': True},
        )
        if time.time() - start_time < time_limit:
            cand = _clamp_all(res.x.reshape(N, 2))
            s = _min_area(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
    except Exception:
        pass
    return best_pts, best_score


def _generate_seeds():
    """Generate diverse initial configurations."""
    h = _H
    seeds = []
    
    # Seed 1: Vertices + centroid + symmetric interior points
    s1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.167, h/12],
        [0.833, h/12],
        [0.333, h/4],
        [0.667, h/4],
    ])
    seeds.append(_clamp_all(s1))
    
    # Seed 2: Triangular grid pattern
    s2 = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, h/4],
        [0.5, h/4],
        [0.75, h/4],
        [0.5, h/2],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, 2*h/3],
        [0.25, 2*h/3],
    ])
    seeds.append(_clamp_all(s2))
    
    # Seed 3: 3-fold symmetric
    s3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.167, h/6],
        [0.833, h/6],
        [0.5, h/2],
        [0.333, 2*h/3],
        [0.667, 2*h/3],
        [0.25, h/4],
        [0.75, h/4],
    ])
    seeds.append(_clamp_all(s3))
    
    # Seed 4: Points near edges and center
    s4 = np.array([
        [0.1, 0.05],
        [0.9, 0.05],
        [0.5, h - 0.05],
        [0.5, h/3],
        [0.3, h/6],
        [0.7, h/6],
        [0.5, h/2],
        [0.2, h/4],
        [0.8, h/4],
        [0.4, 2*h/5],
        [0.6, 2*h/5],
    ])
    seeds.append(_clamp_all(s4))
    
    # Seed 5: More spread out, avoiding corners
    s5 = np.array([
        [0.15, 0.1],
        [0.85, 0.1],
        [0.5, h - 0.1],
        [0.5, h/3],
        [0.3, h/5],
        [0.7, h/5],
        [0.5, h/2],
        [0.25, 2*h/5],
        [0.75, 2*h/5],
        [0.4, h/2.5],
        [0.6, h/2.5],
    ])
    seeds.append(_clamp_all(s5))
    
    # Seed 6: Hexagonal-like arrangement
    s6 = np.array([
        [0.5, h/6],
        [0.25, h/6],
        [0.75, h/6],
        [0.375, h/3],
        [0.625, h/3],
        [0.5, h/2],
        [0.25, h/2],
        [0.75, h/2],
        [0.375, 2*h/3],
        [0.625, 2*h/3],
        [0.5, 5*h/6],
    ])
    seeds.append(_clamp_all(s6))
    
    # Random seeds with better distribution
    rng = np.random.default_rng(42)
    for i in range(10):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        # Add some structure: ensure at least one point near each vertex
        if i < 3:
            pts[0] = [0.05, 0.05]
            pts[1] = [0.95, 0.05]
            pts[2] = [0.5, h - 0.05]
        seeds.append(_clamp_all(pts))
    
    return seeds


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pts = None
    best_score = -1.0
    
    def _update(pts, score):
        nonlocal best_pts, best_score
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()
    
    seeds = _generate_seeds()
    n_seeds = len(seeds)
    
    # Phase 1: SA on each seed for global exploration - give more time per seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 40:
            break
        remaining = deadline - 40 - time.time()
        per = min(6.0, remaining / max(1, n_seeds - i))
        if per < 1.0:
            break
        try:
            opt_pts, score = _simulated_annealing(seed, time_limit=per, seed=i * 100)
            _update(opt_pts, score)
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = seeds[0]
        best_score = _min_area(best_pts)
    
    # Phase 2: Gradient refinement with annealed log-sum-exp on best
    if best_pts is not None and time.time() < deadline - 25:
        remaining = deadline - 25 - time.time()
        if remaining > 3:
            try:
                grad_pts, grad_score = _gradient_refine(
                    best_pts, beta_start=30, beta_end=5000,
                    n_steps=25, time_limit=min(15.0, remaining)
                )
                _update(grad_pts, grad_score)
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead refinement
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - 15 - time.time()
        if remaining > 2:
            try:
                nm_pts, nm_score = _nelder_mead_refine(
                    best_pts, time_limit=min(5.0, remaining)
                )
                _update(nm_pts, nm_score)
            except Exception:
                pass
    
    # Phase 4: Another SA round on the best with different seed (exploration)
    if best_pts is not None and time.time() < deadline - 12:
        remaining = deadline - 12 - time.time()
        if remaining > 3:
            try:
                sa_pts, sa_score = _simulated_annealing(
                    best_pts, time_limit=min(10.0, remaining), seed=9999
                )
                _update(sa_pts, sa_score)
            except Exception:
                pass
    
    # Phase 5: Final gradient polish with very high beta
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 2:
            try:
                final_pts, final_score = _gradient_refine(
                    best_pts, beta_start=500, beta_end=15000,
                    n_steps=15, time_limit=min(8.0, remaining)
                )
                _update(final_pts, final_score)
            except Exception:
                pass
    
    # Phase 6: One more Nelder-Mead if time allows
    if best_pts is not None and time.time() < deadline - 1:
        remaining = deadline - 1 - time.time()
        if remaining > 1:
            try:
                nm2_pts, nm2_score = _nelder_mead_refine(
                    best_pts, time_limit=min(3.0, remaining)
                )
                _update(nm2_pts, nm2_score)
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = seeds[0]
    
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
