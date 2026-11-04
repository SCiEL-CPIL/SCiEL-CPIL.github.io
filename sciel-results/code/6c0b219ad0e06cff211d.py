"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute triple indices for efficiency
_triple_indices = np.array(list(combinations(range(N), 3)))
_TRI_I = _triple_indices[:, 0]
_TRI_J = _triple_indices[:, 1]
_TRI_K = _triple_indices[:, 2]

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    # Clip to avoid overflow (shifted should be <= 0)
    shifted = np.clip(shifted, -100, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function: minimize -smooth_min(ratios).
    params: 22 values representing p_3 ... p_13.
    p_1 is fixed at (0,0), p_2 is fixed at (1,0).
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _exact_objective(params):
    """
    Exact objective for Nelder-Mead polish: minimize -score.
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    score = _compute_min_ratio(pts)
    return -score

def _generate_initial_params(rng, method='polygon'):
    """
    Generate initial parameters for the 22 free coordinates.
    """
    if method == 'polygon':
        # Regular 13-gon with random perturbation
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles += rng.uniform(-0.15, 0.15, N)
        radii = 1.0 + rng.normal(0, 0.08, N)
        
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Transform: p[0] -> (0,0), p[1] -> (1,0)
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
    
    elif method == 'random':
        return rng.uniform(-1, 1, 22)
    
    elif method == 'halton':
        # Halton sequence for low-discrepancy
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        
        # Add small random perturbation
        pts += rng.normal(0, 0.05, (N, 2))
        
        # Transform
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
    
    elif method == 'ring':
        # Concentric ring initialization
        n_inner = 5
        n_outer = N - n_inner
        angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
        r_in = 0.4 + rng.uniform(-0.1, 0.1)
        angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
        r_out = 1.0 + rng.uniform(-0.1, 0.1)
        
        pts = np.zeros((N, 2))
        pts[:n_inner, 0] = r_in * np.cos(angles_in)
        pts[:n_inner, 1] = r_in * np.sin(angles_in)
        pts[n_inner:, 0] = r_out * np.cos(angles_out)
        pts[n_inner:, 1] = r_out * np.sin(angles_out)
        
        # Transform
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
    
    else:
        return rng.uniform(-1, 1, 22)

def _nelder_mead_polish(params, max_iter=300, tol=1e-12):
    """
    Polish using Nelder-Mead on the exact (non-smooth) objective.
    This can escape plateaus of the smooth-min surrogate.
    """
    try:
        result = minimize(
            _exact_objective,
            params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x
    except Exception:
        return params

def _differential_evolution_step(rng, population, scores, time_limit, elapsed_start):
    """
    Perform a DE mutation step.
    population: list of param arrays (22,)
    scores: list of scores
    Returns updated population and scores.
    """
    pop_size = len(population)
    if pop_size < 4:
        return population, scores
    
    # DE/rand/1 mutation
    new_member_idx = rng.integers(0, pop_size)
    a = rng.choice([i for i in range(pop_size) if i != new_member_idx])
    b = rng.choice([i for i in range(pop_size) if i != new_member_idx and i != a])
    c = rng.choice([i for i in range(pop_size) if i != new_member_idx and i != a and i != b])
    
    F = rng.uniform(0.5, 1.0)
    trial = population[a] + F * (population[b] - population[c])
    
    # Evaluate trial
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = trial.reshape(N - 2, 2)
    score = _compute_min_ratio(pts)
    
    if score > scores[new_member_idx]:
        population[new_member_idx] = trial
        scores[new_member_idx] = score
    
    return population, scores

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points
    num_starts = 16
    starts = []
    start_methods = ['polygon', 'halton', 'random', 'ring', 'polygon', 'halton', 'random', 'ring',
                     'polygon', 'halton', 'random', 'ring', 'polygon', 'halton', 'random', 'ring']
    
    for i in range(num_starts):
        if (time.time() - start_time) > time_limit - 25.0:
            break
        method = start_methods[i % len(start_methods)]
        p = _generate_initial_params(rng, method)
        starts.append(p)
    
    # Phase 2: Local optimization with L-BFGS-B using smooth min
    # Use multiple tau values for annealing
    tau_schedule = [15, 40, 100, 300, 800]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 15.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 8.0:
                break
            
            try:
                res = minimize(
                    _objective,
                    starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                # Evaluate true score of the result
                new_params = res.x
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = new_params.reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = new_params.copy()
                
                # Update the start for next tau iteration
                starts[i] = new_params.copy()
                
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish on the best solution (key improvement from inspiration 1)
    # This directly optimizes the non-smooth objective and can escape smooth-min plateaus
    if best_params is not None and (time.time() - start_time) < time_limit - 20.0:
        try:
            polished = _nelder_mead_polish(best_params, max_iter=500, tol=1e-12)
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = polished.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            if score > best_score:
                best_score = score
                best_params = polished.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish with high tau on the best solution found
    if best_params is not None and (time.time() - start_time) < time_limit - 15.0:
        try:
            # Polish with very high tau
            res = minimize(
                _objective,
                best_params,
                args=(2000,),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = res.x.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Second polish pass with NM if time allows
        if best_params is not None and (time.time() - start_time) < time_limit - 10.0:
            try:
                polished = _nelder_mead_polish(best_params, max_iter=400, tol=1e-13)
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = polished.reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_params = polished.copy()
            except Exception:
                pass
    
    # Phase 5: DE-inspired global search with local polish
    if (time.time() - start_time) < time_limit - 25.0:
        # Create population from best starts
        population = [starts[i].copy() for i in range(min(10, len(starts)))]
        scores = []
        for p in population:
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = p.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_params = p.copy()
        
        # Run DE for a few iterations
        max_de_iters = 40
        for it in range(max_de_iters):
            if (time.time() - start_time) > time_limit - 12.0:
                break
            
            population, scores = _differential_evolution_step(rng, population, scores, time_limit, start_time)
            
            # Track best
            for i, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_params = population[i].copy()
            
            # Occasionally do a local polish on the best DE result
            if it % 5 == 4 and (time.time() - start_time) < time_limit - 10.0:
                best_idx = int(np.argmax(scores))
                try:
                    res = minimize(
                        _objective,
                        population[best_idx],
                        args=(500,),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    pts = np.empty((N, 2))
                    pts[0] = [0.0, 0.0]
                    pts[1] = [1.0, 0.0]
                    pts[2:] = res.x.reshape(N - 2, 2)
                    score = _compute_min_ratio(pts)
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
                        population[best_idx] = res.x.copy()
                        scores[best_idx] = score
                except Exception:
                # Try NM polish instead
                    try:
                        polished = _nelder_mead_polish(population[best_idx], max_iter=150, tol=1e-11)
                        pts = np.empty((N, 2))
                        pts[0] = [0.0, 0.0]
                        pts[1] = [1.0, 0.0]
                        pts[2:] = polished.reshape(N - 2, 2)
                        score = _compute_min_ratio(pts)
                        if score > best_score:
                            best_score = score
                            best_params = polished.copy()
                            population[best_idx] = polished.copy()
                            scores[best_idx] = score
                    except Exception:
                        pass
    
    # Phase 6: Final NM polish on the overall best
    if best_params is not None and (time.time() - start_time) < time_limit - 5.0:
        try:
            polished = _nelder_mead_polish(best_params, max_iter=300, tol=1e-14)
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = polished.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            if score > best_score:
                best_score = score
                best_params = polished.copy()
        except Exception:
            pass
    
    # Reconstruct final points
    if best_params is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    final_pts = np.empty((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(N - 2, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
