"""
Hybrid strategy: Simulated Annealing for global exploration + Log-Sum-Exp gradient polishing.
This approach leverages the robustness of SA to escape local minima and the precision of 
gradient-based methods for final refinement.
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


def _gradient_refine(pts, beta_start=50.0, beta_end=2000.0, n_steps=30, time_limit=10.0):
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
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000},
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


def _simulated_annealing(pts, start_temp=0.1, end_temp=0.001, n_iter=20000, time_limit=30.0, seed=42):
    """
    Simulated Annealing with adaptive step size.
    """
    rng = np.random.default_rng(seed)
    current_pts = _clamp_all(np.array(pts, dtype=float))
    current_score = _min_area(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    start_time = time.time()
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Linear cooling schedule
        t = i / n_iter
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Adaptive step size: larger steps when temp is high
        step_size = 0.05 * (1 + 5 * temp)
        
        # Perturb
        new_pts = current_pts + rng.normal(0, step_size, current_pts.shape)
        new_pts = _clamp_all(new_pts)
        new_score = _min_area(new_pts)
        
        # Accept if better or with probability exp(-delta/temp)
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            delta = current_score - new_score
            if delta < 0:
                delta = 0
            prob = np.exp(-delta / temp)
            if rng.random() < prob:
                current_pts = new_pts
                current_score = new_score
    
    return best_pts, best_score


def _generate_seeds():
    """Generate diverse initial configurations."""
    h = _H
    seeds = []
    
    # Seed 1: Vertices + centroid + 7 points on a hexagonal-like pattern
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
    
    # Seed 2: 3-4-4 triangular grid
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
    
    # Seed 3: Symmetric with 3-fold rotation
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
    
    # Seed 4: Random with fixed seed
    rng = np.random.default_rng(123)
    for _ in range(10):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        seeds.append(_clamp_all(pts))
        
    # Seed 5: Points near edges and center
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
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pts = None
    best_score = -1.0
    
    def _update(pts, score):
        nonlocal best_pts, best_score
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()
            
    seeds = _generate_seeds()
    
    # Phase 1: Simulated Annealing from multiple seeds
    # SA is good for global exploration
    n_seeds = len(seeds)
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        remaining = deadline - 30 - time.time()
        per = min(5.0, remaining / max(1, n_seeds - i))
        if per < 1.0:
            break
        try:
            sa_pts, sa_score = _simulated_annealing(seed, time_limit=per, seed=100+i)
            _update(sa_pts, sa_score)
        except Exception:
            continue
            
    if best_pts is None:
        best_pts = seeds[0]
        best_score = _min_area(best_pts)
        
    # Phase 2: Gradient refinement with annealed log-sum-exp
    # This polishes the result from SA
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - 15 - time.time()
        if remaining > 3:
            try:
                grad_pts, grad_score = _gradient_refine(
                    best_pts, beta_start=30, beta_end=3000,
                    n_steps=25, time_limit=min(12.0, remaining)
                )
                _update(grad_pts, grad_score)
            except Exception:
                pass
                
    # Phase 3: Another round of SA from the best point found, with smaller steps
    # This can help escape local minima found by the gradient method
    if best_pts is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            try:
                # Use a smaller starting temperature for more local search
                sa2_pts, sa2_score = _simulated_annealing(
                    best_pts, start_temp=0.01, end_temp=0.0001,
                    time_limit=min(8.0, remaining), seed=999
                )
                _update(sa2_pts, sa2_score)
            except Exception:
                pass
                
    # Phase 4: Final gradient polish
    if best_pts is not None and time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 1.5:
            try:
                final_pts, final_score = _gradient_refine(
                    best_pts, beta_start=500, beta_end=10000,
                    n_steps=15, time_limit=min(5.0, remaining)
                )
                _update(final_pts, final_score)
            except Exception:
                pass
                
    if best_pts is None:
        best_pts = seeds[0]
        
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
