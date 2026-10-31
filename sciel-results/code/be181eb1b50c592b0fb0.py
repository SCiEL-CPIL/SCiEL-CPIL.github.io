import numpy as np
from scipy.spatial import ConvexHull
import time
import warnings
from itertools import combinations
from scipy.optimize import minimize

N = 13

def _triangle_areas_vectorized(points):
    n = points.shape[0]
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            v1 = points[j] - points[i]
            for k in range(j+1, n):
                v2 = points[k] - points[i]
                areas[i, j, k] = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
    return areas

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            p3s = points[j+1:]
            v1 = p2 - p1
            v2 = p3s - p1
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            min_area = min(min_area, np.min(areas))
    return min_area / hull_area

def _neg_score(points):
    s = _score(points)
    return -s if s > 0 else 1e6

def _soft_min_value(areas, temperature):
    """Compute soft minimum using log-sum-exp for numerical stability."""
    areas = np.maximum(areas, 1e-15)
    min_val = np.min(areas)
    # Soft min: min_val - T * log(sum(exp(-(a - min_val)/T)))
    scaled = (areas - min_val) / temperature
    # Use log-sum-exp trick
    max_scaled = np.max(-scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(-scaled - max_scaled)))
    return min_val - temperature * log_sum

def _soft_score(points, temperature):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    n = points.shape[0]
    # Compute all triangle areas
    areas = np.array([])
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                areas = np.append(areas, _triangle_area_single(points[i], points[j], points[k]))
    soft_min = _soft_min_value(areas, temperature)
    return soft_min / hull_area

def _triangle_area_single(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _neg_soft_score(points, temperature):
    s = _soft_score(points, temperature)
    return -s if s > 0 else 1e6

def _seed_regular():
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _seed_two_ring():
    # 7 points on outer ring, 6 on inner ring
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    return np.vstack([outer, inner])

def _seed_three_ring():
    # 5 outer, 5 middle, 3 inner
    outer_angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    mid_angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    inner_angles = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    mid = np.column_stack([0.6*np.cos(mid_angles), 0.6*np.sin(mid_angles)])
    inner = np.column_stack([0.25*np.cos(inner_angles), 0.25*np.sin(inner_angles)])
    return np.vstack([outer, mid, inner])

def _seed_quasi_regular():
    # Slightly perturbed regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = 1.0 + 0.05*np.sin(3*angles)
    return np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])

def _annealed_soft_minimize(initial_points, time_budget):
    """Simulated annealing with soft-min objective."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    current = initial_points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    # Temperature schedule: start high, decay exponentially
    T0 = 0.1
    T_min = 1e-6
    n_steps = 5000
    cooling_rate = (T_min / T0) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        T = T0 * (cooling_rate ** step)
        
        # Perturb: random small move
        step_size = 0.05 * T / T0 + 0.01
        perturbed = current + np.random.randn(N, 2) * step_size
        
        # Compute scores
        curr_s = _score(current)
        pert_s = _score(perturbed)
        
        if pert_s >= curr_s:
            current = perturbed
            if pert_s > best_score:
                best_score = pert_s
                best = perturbed.copy()
        else:
            # Accept with probability exp(-delta/T)
            delta = curr_s - pert_s
            if delta < T:
                prob = np.exp(-delta / max(T, 1e-10))
                if np.random.rand() < prob:
                    current = perturbed
                    if pert_s > best_score:
                        best_score = pert_s
                        best = perturbed.copy()
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    seeds = [
        _seed_regular(),
        _seed_two_ring(),
        _seed_three_ring(),
        _seed_quasi_regular(),
    ]
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        
        # First: direct L-BFGS-B on Cartesian coordinates
        try:
            def objective_cart(coords):
                points = coords.reshape(N, 2)
                return _neg_score(points)
            
            x0 = seed.flatten()
            res = minimize(objective_cart, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
            cand_points = res.x.reshape(N, 2)
            cand_score = _score(cand_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_points.copy()
        except Exception:
            pass
        
        # Second: Simulated annealing with soft-min
        if time.time() > deadline - 25:
            break
        
        remaining = deadline - time.time()
        sa_budget = min(20.0, remaining * 0.4)
        
        if sa_budget > 5:
            try:
                sa_points, sa_score = _annealed_soft_minimize(seed, sa_budget)
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points.copy()
            except Exception:
                pass
        
        # Third: Refine with L-BFGS-B from SA result
        if best_points is not None and time.time() > deadline - 15:
            break
        
        if best_points is not None:
            try:
                def objective_refine(coords):
                    points = coords.reshape(N, 2)
                    return _neg_score(points)
                
                x0_ref = best_points.flatten()
                res_ref = minimize(objective_refine, x0_ref, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                cand_ref = res_ref.x.reshape(N, 2)
                cand_score_ref = _score(cand_ref)
                if cand_score_ref > best_score:
                    best_score = cand_score_ref
                    best_points = cand_ref.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
