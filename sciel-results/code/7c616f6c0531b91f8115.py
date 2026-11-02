"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14


# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """Compute the minimum area over all triples of points (vectorized)."""
    n = len(points)
    # Vectorized: compute all cross products
    # For each pair (i,j), cross with all k > j
    # Use broadcasting: for each i, compute cross of (p_j - p_i) x (p_k - p_i) for all j,k
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        others = np.delete(points, i, axis=0)
        v1 = others[:, 0] - pi[0]
        v2 = others[:, 1] - pi[1]
        cross = np.abs(np.outer(v1, v2) - np.outer(v2, v1))
        idx = np.triu_indices(n - 1, k=1)
        areas = 0.5 * cross[idx]
        if len(areas) > 0:
            min_in_i = np.min(areas)
            if min_in_i < min_area:
                min_area = min_in_i
    return min_area


def _all_triangle_areas(points):
    """Compute all triangle areas as a flat array."""
    n = len(points)
    areas = []
    for i in range(n):
        pi = points[i]
        others = np.delete(points, i, axis=0)
        v1 = others[:, 0] - pi[0]
        v2 = others[:, 1] - pi[1]
        cross = np.abs(np.outer(v1, v2) - np.outer(v2, v1))
        idx = np.triu_indices(n - 1, k=1)
        areas.append(0.5 * cross[idx])
    return np.concatenate(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _soft_min(areas, temperature):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    # soft_min = -log(sum(exp(-a_i / T))) * T
    # For stability: subtract max first
    neg_areas = -areas / temperature
    max_val = np.max(neg_areas)
    shifted = neg_areas - max_val
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val
    return -log_sum * temperature


def _objective_smooth(x, temperature=0.05):
    """Smooth objective: maximize soft-min triangle area / hull area."""
    points = x.reshape(-1, 2)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 1.0
    areas = _all_triangle_areas(points)
    sm = _soft_min(areas, temperature)
    return -(sm / hull_area)


def _objective(x):
    """Objective for optimization: minimize negative score (i.e., maximize score)."""
    points = x.reshape(-1, 2)
    return -_score(points)


def _sa_optimize(points, n_iters=5000, t_start=0.01, t_end=0.0001, 
                 step_size=0.05, deadline=None):
    """Vectorized simulated annealing to optimize min triangle area / hull area."""
    best_points = points.copy()
    best_score = _score(points)
    current_points = points.copy()
    current_score = best_score
    
    n = len(current_points)
    rng = np.random.default_rng(42)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate perturbation
        perturbation = rng.normal(0, step_size, size=(n, 2))
        new_points = current_points + perturbation
        
        new_score = _score(new_points)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        else:
            delta = current_score - new_score
            if delta < 1e-10:
                continue
            # Temperature annealing
            T = t_start * (t_end / t_start) ** (it / n_iters)
            if rng.random() < np.exp(-delta / (T * max(best_score, 1e-10))):
                current_points = new_points
                current_score = new_score
        
        # Reduce step size gradually
        step_size = max(0.001, 0.05 * (0.999 ** it))
    
    return best_points, best_score


def construct():
    """Construct 14 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # Strategy 1: Differential evolution with smooth objective
    bounds = [(-2, 2)] * (2 * N)
    
    try:
        result = differential_evolution(
            _objective_smooth,
            bounds=bounds,
            seed=42,
            maxiter=40,
            popsize=15,
            tol=1e-5,
            polish=False,
            init='sobol',
            args=(0.1,)
        )
        
        if time.time() < deadline - 20:
            candidate_points = result.x.reshape(-1, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
    except Exception:
        pass
    
    # Strategy 2: Simulated annealing (much more iterations)
    if time.time() < deadline - 30:
        try:
            sa_points, sa_score = _sa_optimize(
                best_points, 
                n_iters=20000,
                t_start=0.02,
                t_end=0.0001,
                step_size=0.03,
                deadline=deadline - 15
            )
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points
        except Exception:
            pass
    
    # Strategy 3: Polish with local optimization
    if time.time() < deadline - 15 and best_points is not None:
        try:
            # First try Nelder-Mead
            result2 = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            candidate_points = result2.x.reshape(-1, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
        except Exception:
            pass
    
    # Strategy 4: More SA restarts from different initial points
    if time.time() < deadline - 25:
        for seed in [123, 456]:
            if time.time() > deadline - 15:
                break
            try:
                # Perturb the best points
                rng = np.random.default_rng(seed)
                perturbed = best_points + rng.normal(0, 0.1, size=(N, 2))
                
                sa_points, sa_score = _sa_optimize(
                    perturbed,
                    n_iters=10000,
                    t_start=0.01,
                    t_end=0.0001,
                    step_size=0.03,
                    deadline=deadline - 15
                )
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points
                    
                # Quick polish
                if time.time() < deadline - 10:
                    result2 = minimize(
                        _objective,
                        best_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    candidate_points = result2.x.reshape(-1, 2)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_points
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 5 and best_points is not None:
        try:
            result3 = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate_points = result3.x.reshape(-1, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
