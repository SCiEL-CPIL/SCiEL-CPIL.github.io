import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
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
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
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

def _soft_min_score(params, beta=60.0):
    """Smooth surrogate using power mean of inverse areas."""
    points = params.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    inv_areas = 1.0 / areas
    inv_areas_p = inv_areas ** beta
    mean_inv_p = np.mean(inv_areas_p)
    soft_min = mean_inv_p ** (-1.0 / beta)
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _regular_13gon():
    """Return regular 13-gon on unit circle."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _coordinate_descent_polish(points, deadline, max_rounds=200):
    """Coordinate descent: perturb each point in x and y directions."""
    best_points = points.copy()
    best_score = _score(best_points)
    
    base_step = 0.01
    for round_num in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            for coord in [0, 1]:
                for sign in [1, -1]:
                    trial = best_points.copy()
                    trial[i, coord] += sign * base_step
                    s = _score(trial)
                    if s > best_score:
                        best_score = s
                        best_points = trial.copy()
                        improved = True
        if not improved:
            base_step *= 0.5
            if base_step < 1e-8:
                break
        else:
            base_step *= 0.8
    return best_points, best_score

def _simulated_annealing(points, deadline, t0=0.1, t1=1e-6, alpha=0.999):
    """Simulated annealing to escape local minima."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    t = t0
    
    while time.time() < deadline - 5:
        # Random perturbation
        trial = current + np.random.randn(N, 2) * t * 0.5
        trial_score = _score(trial)
        
        if trial_score > current_score:
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        else:
            delta = current_score - trial_score
            if np.random.rand() < np.exp(-delta / max(t, 1e-12)):
                current = trial
                current_score = trial_score
        
        t *= alpha
    
    return best, best_score

def _cma_es_polish(points, deadline):
    """Use CMA-ES via scipy's minimize with Powell for final polish."""
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Use Powell's method for final polish
    if time.time() < deadline - 3:
        try:
            def obj(x):
                s = _score(x.reshape(N, 2))
                return -s if s > 0 else 1e10
            
            res = minimize(
                obj,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    return best_points, best_score

def _local_polish_full(points, deadline):
    """Full local polish: coordinate descent + simulated annealing + Powell."""
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Coordinate descent
    if time.time() < deadline - 10:
        cd_points, cd_score = _coordinate_descent_polish(best_points, deadline)
        if cd_score > best_score:
            best_score = cd_score
            best_points = cd_points.copy()
    
    # Simulated annealing
    if time.time() < deadline - 5:
        sa_points, sa_score = _simulated_annealing(best_points, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # Powell polish
    if time.time() < deadline - 3:
        pw_points, pw_score = _cma_es_polish(best_points, deadline)
        if pw_score > best_score:
            best_score = pw_score
            best_points = pw_points.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Phase 1: Start from regular 13-gon and polish
    reg = _regular_13gon()
    _update_best(reg)
    print(f"Regular 13-gon: score={best_score:.6f}", flush=True)
    
    # Phase 2: Local polish from regular 13-gon
    if time.time() < deadline - 30:
        polished, score = _local_polish_full(reg, deadline)
        _update_best(polished)
        print(f"After polish: score={best_score:.6f}", flush=True)
    
    # Phase 3: Try various structured configurations
    rng = np.random.default_rng(42)
    
    structured_inits = [reg]
    
    # Two-ring configurations
    for n_out in [6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.5, 0.7]:
            angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
            angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + np.pi / n_out
            radii = np.concatenate([np.ones(n_out), np.full(n_in, r_in)])
            angles_all = np.concatenate([angles_out, angles_in])
            pts = np.column_stack([radii * np.cos(angles_all), radii * np.sin(angles_all)])
            structured_inits.append(pts)
    
    # Perturbed regular 13-gons
    for i in range(5):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.random(N) * 0.1
        radii = 1.0 + rng.random(N) * 0.2 - 0.1
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        structured_inits.append(pts)
    
    for i, init_pts in enumerate(structured_inits):
        if time.time() > deadline - 20:
            break
        _update_best(init_pts)
        if time.time() < deadline - 10:
            polished, score = _local_polish_full(init_pts, deadline)
            _update_best(polished)
        if i % 3 == 0:
            print(f"Start {i}: score={best_score:.6f}", flush=True)
    
    # Phase 4: DE for global search
    if time.time() < deadline - 40:
        try:
            bounds = [(-2, 2)] * (N * 2)
            
            def de_obj(x):
                s = _score(x.reshape(N, 2))
                return -s if s > 0 else 1e10
            
            de_params = {
                'maxiter': 30,
                'popsize': 12,
                'mutation': (0.5, 1.0),
                'recombination': 0.7,
                'tol': 1e-6,
                'seed': 42,
                'polish': False,
                'init': 'sobol',
                'updating': 'immediate',
            }
            
            result = differential_evolution(de_obj, bounds, **de_params)
            if result.fun < 1e9:
                de_points = result.x.reshape(N, 2)
                _update_best(de_points)
                print(f"DE: score={best_score:.6f}", flush=True)
                
                # Polish DE result
                if time.time() < deadline - 15:
                    polished, score = _local_polish_full(de_points, deadline)
                    _update_best(polished)
                    print(f"DE polished: score={best_score:.6f}", flush=True)
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if best_points is not None and time.time() < deadline - 5:
        polished, score = _local_polish_full(best_points, deadline)
        _update_best(polished)
        print(f"Final: score={best_score:.6f}", flush=True)
    
    if best_points is None:
        best_points = _regular_13gon()
    
    return best_points
