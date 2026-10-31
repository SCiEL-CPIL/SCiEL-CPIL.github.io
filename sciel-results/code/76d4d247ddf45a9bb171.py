"""
SA + log-sum-exp gradient polishing hybrid.
SA provides robust global exploration; LSE gradient gives high-precision local refinement.
Multi-phase: SA global -> gradient refine -> SA re-explore -> final polish.
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


def _clamp_one(pt):
    """Clamp a single 2D point to triangle."""
    x, y = pt
    y = np.clip(y, _EPS, _H - _EPS)
    x_min = y / _SQRT3 + _EPS
    x_max = 1.0 - y / _SQRT3 - _EPS
    x = np.clip(x, x_min, x_max)
    return np.array([x, y])


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


def _logsumexp_neg(pts, beta):
    """
    Smooth approximation of -min(areas) using log-sum-exp.
    As beta -> inf, this approaches -min(areas).
    """
    areas = _areas_vec(pts)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    # Numerically stable log-sum-exp
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    return lse / beta


def _objective_lse(params, beta):
    """Smooth objective using log-sum-exp."""
    pts = _clamp_all(params.reshape(N, 2))
    return _logsumexp_neg(pts, beta)


def _objective_exact(params):
    """Exact non-smooth objective: negative min area."""
    pts = _clamp_all(params.reshape(N, 2))
    return -_min_area(pts)


def _gradient_refine(pts, time_limit=10.0, maxiter=200):
    """
    Gradient-based refinement using LSE smooth objective.
    Uses BFGS with annealed beta for high-precision local optimization.
    """
    start_time = time.time()
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    best_x = x0.copy()
    best_score = _min_area(best_x.reshape(N, 2))
    
    # Anneal beta from moderate to high
    for beta in [50.0, 100.0, 200.0, 400.0]:
        if time.time() > start_time + time_limit:
            break
        
        def obj(x):
            return _objective_lse(x, beta)
        
        def grad(x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                g[i] = (_objective_lse(xp, beta) - _objective_lse(xm, beta)) / (2 * eps)
            return g
        
        try:
            res = minimize(
                obj,
                best_x,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            cand = _clamp_all(res.x.reshape(N, 2))
            s = _min_area(cand)
            if s > best_score:
                best_score = s
                best_x = cand.flatten()
        except Exception:
            pass
    
    return _clamp_all(best_x.reshape(N, 2)), best_score


def _simulated_annealing(pts, time_limit=30.0, seed=42):
    """
    Simulated annealing with adaptive step size for global exploration.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    current = _clamp_all(np.array(pts, dtype=float)).copy()
    current_score = _min_area(current)
    best = current.copy()
    best_score = current_score
    
    T_init = 0.05
    T_min = 1e-6
    T_final = T_min
    steps = 5000
    step_size_init = 0.05
    
    for step in range(steps):
        if time.time() > start_time + time_limit:
            break
        
        # Adaptive temperature and step size
        progress = step / steps
        T = T_init * (T_final / T_init) ** progress
        step_size = step_size_init * (1.0 - 0.8 * progress)
        
        # Generate candidate: perturb one or two random points
        n_perturb = 1 if rng.random() < 0.6 else 2
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        candidate = current.copy()
        for i in idx:
            delta = rng.normal(0, step_size, size=2)
            candidate[i] += delta
        candidate = _clamp_all(candidate)
        
        cand_score = _min_area(candidate)
        delta_e = cand_score - current_score
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e > 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            current = candidate
            current_score = cand_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        # Occasionally try larger jumps to escape local minima
        if rng.random() < 0.05:
            idx = rng.choice(N, size=3, replace=False)
            for i in idx:
                delta = rng.normal(0, step_size * 2, size=2)
                candidate[i] += delta
            candidate = _clamp_all(candidate)
            cand_score = _min_area(candidate)
            if cand_score > current_score or rng.random() < np.exp(-(cand_score - current_score) / max(T, 1e-10)):
                current = candidate
                current_score = cand_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
    
    return _clamp_all(best), best_score


def _nelder_mead_refine(pts, time_limit=8.0, maxiter=300):
    """
    Nelder-Mead local search on the non-smooth min-area objective.
    """
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    start_time = time.time()
    
    best_x = x0.copy()
    best_score = _min_area(best_x.reshape(N, 2))
    
    try:
        res = minimize(
            _objective_exact,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
            },
        )
        cand = _clamp_all(res.x.reshape(N, 2))
        s = _min_area(cand)
        if s > best_score:
            best_score = s
            best_x = cand.flatten()
    except Exception:
        pass
    
    return _clamp_all(best_x.reshape(N, 2)), best_score


def _generate_seeds():
    """Generate diverse initial configurations."""
    h = _H
    seeds = []
    
    # Seed 1: Vertices + centroid + hexagonal pattern
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
    
    # Seed 2: Triangular grid
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
    
    # Seed 4: Random
    rng = np.random.default_rng(123)
    for _ in range(6):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        seeds.append(_clamp_all(pts))
        
    # Seed 5: Near edges
    s5 = np.array([
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
    seeds.append(_clamp_all(s5))
    
    return seeds


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_pts = None
    best_score = -1.0
    
    def _update(pts, score):
        nonlocal best_pts, best_score
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()
    
    seeds = _generate_seeds()
    
    # Phase 1: SA global exploration from multiple seeds
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 50:
            break
        remaining = deadline - 50 - time.time()
        if remaining < 3:
            break
        per = min(8.0, remaining / max(1, len(seeds) - i))
        if per < 2:
            break
        try:
            sa_pts, sa_score = _simulated_annealing(seed, time_limit=per, seed=42+i)
            _update(sa_pts, sa_score)
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = seeds[0]
        best_score = _min_area(best_pts)
    
    # Phase 2: Gradient refinement from best
    if best_pts is not None and time.time() < deadline - 30:
        remaining = deadline - 30 - time.time()
        if remaining > 5:
            try:
                grad_pts, grad_score = _gradient_refine(
                    best_pts, time_limit=min(15.0, remaining), maxiter=150
                )
                _update(grad_pts, grad_score)
            except Exception:
                pass
    
    # Phase 3: SA re-explore from gradient-refined best
    if best_pts is not None and time.time() < deadline - 20:
        remaining = deadline - 20 - time.time()
        if remaining > 5:
            try:
                sa2_pts, sa2_score = _simulated_annealing(
                    best_pts, time_limit=min(15.0, remaining), seed=999
                )
                _update(sa2_pts, sa2_score)
            except Exception:
                pass
    
    # Phase 4: Final gradient polish
    if best_pts is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 3:
            try:
                grad2_pts, grad2_score = _gradient_refine(
                    best_pts, time_limit=min(10.0, remaining), maxiter=100
                )
                _update(grad2_pts, grad2_score)
            except Exception:
                pass
    
    # Phase 5: Final NM polish
    if best_pts is not None and time.time() < deadline - 3:
        remaining = deadline - 3 - time.time()
        if remaining > 2:
            try:
                nm_pts, nm_score = _nelder_mead_refine(
                    best_pts, time_limit=min(5.0, remaining), maxiter=200
                )
                _update(nm_pts, nm_score)
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = seeds[0]
    
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
