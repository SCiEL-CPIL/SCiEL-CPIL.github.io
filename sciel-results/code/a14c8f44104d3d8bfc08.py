"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11
TRI_AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1

# Precompute all C(11,3) = 165 triple indices
_TRIPLES = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
_T = np.array(_TRIPLES, dtype=np.int64)
_N_TRIPLES = len(_T)


def _min_area_raw(x):
    """x: (2N,) flat Cartesian coords. Returns min |det|/2 over all triples (actual min area)."""
    pts = x.reshape(N, 2)
    # Compute all triangle areas via cross product
    i, j, k = _T[:, 0], _T[:, 1], _T[:, 2]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    # det = (pj - pi) x (pk - pi) = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    dets = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = np.abs(dets) * 0.5
    return np.min(areas)


def _is_inside_triangle(pts):
    """Check if all points are inside the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x = pts[:, 0]
    y = pts[:, 1]
    # Half-plane 1: y >= 0
    if np.any(y < -1e-10):
        return False
    # Half-plane 2: y <= sqrt(3)*x  (below line from (0,0) to (0.5, sqrt(3)/2))
    if np.any(y > np.sqrt(3) * x + 1e-10):
        return False
    # Half-plane 3: y <= sqrt(3)*(1-x)  (below line from (1,0) to (0.5, sqrt(3)/2))
    if np.any(y > np.sqrt(3) * (1.0 - x) + 1e-10):
        return False
    return True


def _project_inside(x):
    """Project flat (2N,) Cartesian coords to be inside the triangle."""
    pts = x.reshape(N, 2).copy()
    # Clip y >= 0
    pts[:, 1] = np.maximum(pts[:, 1], 0.0)
    # Clip below line y = sqrt(3)*x
    max_y1 = np.sqrt(3) * pts[:, 0]
    pts[:, 1] = np.minimum(pts[:, 1], max_y1)
    # Clip below line y = sqrt(3)*(1-x)
    max_y2 = np.sqrt(3) * (1.0 - pts[:, 0])
    pts[:, 1] = np.minimum(pts[:, 1], max_y2)
    # Ensure x in [0, 1]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    # Re-clip y after x clipping
    pts[:, 1] = np.maximum(pts[:, 1], 0.0)
    pts[:, 1] = np.minimum(pts[:, 1], np.sqrt(3) * pts[:, 0])
    pts[:, 1] = np.minimum(pts[:, 1], np.sqrt(3) * (1.0 - pts[:, 0]))
    return pts.flatten()


def _objective_smooth(x, beta=50.0):
    """Smooth objective: -softmin of areas + penalty for being outside triangle."""
    pts = x.reshape(N, 2)
    i, j, k = _T[:, 0], _T[:, 1], _T[:, 2]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    dets = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = np.abs(dets) * 0.5
    
    # Log-sum-exp for smooth min
    a_min = np.min(areas)
    shifted = areas - a_min
    # Use stable logsumexp
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_min = a_min - (1.0 / beta) * log_sum_exp
    
    # Penalty for points outside triangle
    x_c = pts[:, 0]
    y_c = pts[:, 1]
    pen = 0.0
    # y < 0
    neg_y = np.maximum(0.0, -y_c)
    pen += np.sum(neg_y ** 2)
    # y > sqrt(3)*x
    viol1 = np.maximum(0.0, y_c - np.sqrt(3) * x_c)
    pen += np.sum(viol1 ** 2)
    # y > sqrt(3)*(1-x)
    viol2 = np.maximum(0.0, y_c - np.sqrt(3) * (1.0 - x_c))
    pen += np.sum(viol2 ** 2)
    # x outside [0,1]
    neg_x = np.maximum(0.0, -x_c)
    pen += np.sum(neg_x ** 2)
    pos_x = np.maximum(0.0, x_c - 1.0)
    pen += np.sum(pos_x ** 2)
    
    return -smooth_min + 1000.0 * pen


def _objective_de(x):
    """Objective for differential evolution: minimize negative min area (with penalty)."""
    pts = x.reshape(N, 2)
    if not _is_inside_triangle(pts):
        # Compute how far outside
        x_c = pts[:, 0]
        y_c = pts[:, 1]
        pen = 0.0
        pen += np.sum(np.maximum(0.0, -y_c) ** 2)
        pen += np.sum(np.maximum(0.0, y_c - np.sqrt(3) * x_c) ** 2)
        pen += np.sum(np.maximum(0.0, y_c - np.sqrt(3) * (1.0 - x_c)) ** 2)
        pen += np.sum(np.maximum(0.0, -x_c) ** 2)
        pen += np.sum(np.maximum(0.0, x_c - 1.0) ** 2)
        return 1.0 + 100.0 * pen
    
    return -_min_area_raw(x)


def _lbgfsb_polish(x0, beta=200.0, maxiter=300):
    """Polish with L-BFGS-B using smooth objective."""
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
    try:
        res = minimize(
            lambda x: _objective_smooth(x, beta=beta),
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        return res.x
    except Exception:
        return x0


def _nm_polish(x0, maxiter=500):
    """Polish with Nelder-Mead on the hard objective."""
    def obj(x):
        pts = x.reshape(N, 2)
        if not _is_inside_triangle(pts):
            x_c = pts[:, 0]
            y_c = pts[:, 1]
            pen = np.sum(np.maximum(0.0, -y_c)**2) + np.sum(np.maximum(0.0, y_c - np.sqrt(3)*x_c)**2)
            pen += np.sum(np.maximum(0.0, y_c - np.sqrt(3)*(1.0-x_c))**2)
            pen += np.sum(np.maximum(0.0, -x_c)**2) + np.sum(np.maximum(0.0, x_c-1.0)**2)
            return 1.0 + 100.0 * pen
        return -_min_area_raw(x)
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0


def _slsqp_polish(x0, maxiter=300):
    """Polish with SLSQP using explicit constraints."""
    pts0 = x0.reshape(N, 2)
    
    # Inequality constraints: g(x) >= 0 means point is inside
    # For each point i: y_i >= 0, y_i <= sqrt(3)*x_i, y_i <= sqrt(3)*(1-x_i), 0<=x_i<=1
    # We'll use bounds for x, and inequality constraints for y
    
    def objective(x):
        return _objective_de(x)
    
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
    
    # Inequality constraints: y - sqrt(3)*x <= 0 and y - sqrt(3)*(1-x) <= 0
    # SLSQP: constraint dict with 'type': 'ineq' means f(x) >= 0
    # So we need: sqrt(3)*x - y >= 0 and sqrt(3)*(1-x) - y >= 0
    def con1(x):
        pts = x.reshape(N, 2)
        return np.sqrt(3) * pts[:, 0] - pts[:, 1]
    
    def con2(x):
        pts = x.reshape(N, 2)
        return np.sqrt(3) * (1.0 - pts[:, 0]) - pts[:, 1]
    
    constraints = []
    for idx in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=idx: np.sqrt(3)*x[2*i] - x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=idx: np.sqrt(3)*(1.0-x[2*i]) - x[2*i+1]})
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        return res.x
    except Exception:
        return x0


def _good_init_1():
    """Symmetric hexagonal-like arrangement."""
    pts = np.array([
        [0.0, 0.0],           # vertex 0
        [1.0, 0.0],           # vertex 1
        [0.5, np.sqrt(3)/2],  # vertex 2
        [0.5, 0.0],           # midpoint bottom
        [0.25, np.sqrt(3)/4], # midpoint left edge
        [0.75, np.sqrt(3)/4], # midpoint right edge
        [0.5, np.sqrt(3)/6],  # center-ish
        [0.25, 0.0],          # quarter bottom left
        [0.75, 0.0],          # quarter bottom right
        [0.5, np.sqrt(3)/3],  # upper center
        [0.5, np.sqrt(3)/12], # lower center
    ])
    return pts.flatten()


def _good_init_2():
    """Another symmetric arrangement."""
    s3 = np.sqrt(3)
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, s3/2],
        [1/3, s3/6],
        [2/3, s3/6],
        [0.5, s3/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, s3/6],
        [0.333, s3/3],
        [0.667, s3/3],
    ])
    return pts.flatten()


def _good_init_3():
    """Grid-based."""
    s3 = np.sqrt(3)
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, s3/2],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.5, s3/6],
        [0.3, s3/4],
        [0.7, s3/4],
        [0.5, s3/3],
        [0.35, s3/6],
        [0.65, s3/6],
    ])
    return pts.flatten()


def _random_init(seed):
    """Random points inside triangle using rejection sampling."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.random()
            y = rng.random() * np.sqrt(3) / 2
            if y <= np.sqrt(3) * x and y <= np.sqrt(3) * (1.0 - x):
                pts[i] = [x, y]
                break
    return pts.flatten()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    best_x = None
    best_score = -float('inf')
    
    def update_best(x):
        nonlocal best_x, best_score
        pts = x.reshape(N, 2)
        if not _is_inside_triangle(pts):
            return
        score = _min_area_raw(x)
        if score > best_score:
            best_score = score
            best_x = x.copy()
    
    # Initial candidates
    inits = [
        _good_init_1(),
        _good_init_2(),
        _good_init_3(),
        _random_init(42),
        _random_init(123),
        _random_init(999),
    ]
    
    # Phase 1: L-BFGS-B polish from each initial
    for x0 in inits:
        if time_left() < 30.0:
            break
        x = _project_inside(x0)
        x = _lbgfsb_polish(x, beta=100.0, maxiter=200)
        x = _project_inside(x)
        update_best(x)
    
    # Phase 2: Differential Evolution (global search)
    if time_left() > 40.0:
        bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
        try:
            de_result = differential_evolution(
                _objective_de,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='immediate',
                callback=None,
            )
            if de_result.success or de_result.fun < 0:
                x_de = de_result.x
                x_de = _project_inside(x_de)
                update_best(x_de)
                # Polish the DE result
                if time_left() > 20.0:
                    x_p = _lbgfsb_polish(x_de, beta=200.0, maxiter=300)
                    x_p = _project_inside(x_p)
                    update_best(x_p)
                    if time_left() > 15.0:
                        x_p2 = _nm_polish(x_p, maxiter=500)
                        x_p2 = _project_inside(x_p2)
                        update_best(x_p2)
        except Exception:
            pass
    
    # Phase 3: SLSQP polish from best
    if best_x is not None and time_left() > 25.0:
        x_s = _slsqp_polish(best_x, maxiter=300)
        x_s = _project_inside(x_s)
        update_best(x_s)
    
    # Phase 4: More random restarts with quick polish
    if time_left() > 30.0:
        for seed in [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]:
            if time_left() < 10.0:
                break
            x0 = _random_init(seed)
            x0 = _project_inside(x0)
            x = _lbgfsb_polish(x0, beta=150.0, maxiter=150)
            x = _project_inside(x)
            update_best(x)
    
    # Phase 5: Final Nelder-Mead polish on best
    if best_x is not None and time_left() > 15.0:
        x_nm = _nm_polish(best_x, maxiter=1000)
        x_nm = _project_inside(x_nm)
        update_best(x_nm)
    
    # Phase 6: One more L-BFGS-B with high beta
    if best_x is not None and time_left() > 10.0:
        x_final = _lbgfsb_polish(best_x, beta=500.0, maxiter=500)
        x_final = _project_inside(x_final)
        update_best(x_final)
    
    # Fallback
    if best_x is None:
        best_x = _good_init_1()
    
    # Final validation and projection
    pts = _project_inside(best_x).reshape(N, 2)
    
    return pts
