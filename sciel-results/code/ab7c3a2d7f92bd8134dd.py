"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull


def _fourier_points(theta, a0, coeffs):
    """Convert Fourier params to points on boundary.
    coeffs: [a1, b1, a2, b2, a3, b3]
    r(theta) = a0 + a1*cos(theta) + b1*sin(theta) + a2*cos(2theta) + b2*sin(2theta) + a3*cos(3theta) + b3*sin(3theta)
    """
    k = np.array([1, 2, 3])
    a_part = np.array([coeffs[0], coeffs[2], coeffs[4]])
    b_part = np.array([coeffs[1], coeffs[3], coeffs[5]])
    
    kt = np.outer(theta, k)  # (n, 3)
    cos_terms = np.cos(kt)
    sin_terms = np.sin(kt)
    r = a0 + cos_terms @ a_part + sin_terms @ b_part
    r = np.maximum(r, 0.05)  # ensure positive radius
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    n = len(pts)
    min_a = np.inf
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            v1 = pts[j] - pi
            for k in range(j + 1, n):
                v2 = pts[k] - pi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a


def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10


def _score(params):
    """params: [a0, a1, b1, a2, b2, a3, b3] -> score (higher better)."""
    a0 = params[0]
    coeffs = params[1:]
    theta = 2.0 * np.pi * np.arange(N) / N
    pts = _fourier_points(theta, a0, coeffs)
    
    # Check distinctness
    min_d = np.inf
    for i in range(N - 1):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_d:
                min_d = d
    if min_d < 1e-8:
        return 0.0
    
    ta = _min_triangle_area(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    return ta / ha


def _score_neg(params):
    return -_score(params)


def construct():
    t_start = time.time()
    DEADLINE = 135.0  # safety margin
    
    bounds = [(0.3, 1.5),
              (-0.5, 0.5), (-0.5, 0.5),
              (-0.3, 0.3), (-0.3, 0.3),
              (-0.15, 0.15), (-0.15, 0.15)]
    
    best_params = None
    best_score = -np.inf
    rng = np.random.RandomState(42)
    
    # Initial evaluations
    init_guesses = [
        np.array([1.0, 0, 0, 0, 0, 0, 0]),  # circle
        np.array([1.0, 0.1, 0, 0.05, 0, 0, 0]),
        np.array([1.0, 0, 0.1, 0, 0.05, 0, 0]),
        np.array([0.8, 0.2, 0, 0, 0, 0, 0]),
        np.array([1.2, -0.1, 0.1, 0, 0, 0, 0]),
    ]
    for p in init_guesses:
        if time.time() > DEADLINE:
            break
        s = _score(p)
        if s > best_score:
            best_score = s
            best_params = p.copy()
    
    # Random starts
    for trial in range(30):
        if time.time() > DEADLINE:
            break
        p = np.array([rng.uniform(0.5, 1.3)] + [rng.uniform(-0.3, 0.3) for _ in range(6)])
        s = _score(p)
        if s > best_score:
            best_score = s
            best_params = p.copy()
    
    # Differential evolution
    if time.time() < DEADLINE - 25:
        try:
            result = differential_evolution(
                _score_neg, bounds,
                seed=42, maxiter=25, tol=1e-5,
                popsize=12, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, init='latinhypercube',
                callback=lambda x, convergence: (time.time() > DEADLINE - 28)
            )
            s = _score(result.x)
            if s > best_score:
                best_score = s
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead refinement
    if best_params is not None and time.time() < DEADLINE - 15:
        try:
            res = minimize(
                _score_neg, best_params,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            s = _score(res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Multi-start NM
        for _ in range(5):
            if time.time() > DEADLINE - 8:
                break
            pert = best_params + rng.normal(0, 0.08, size=7)
            pert[0] = np.clip(pert[0], 0.3, 1.5)
            for i in range(1, 7):
                pert[i] = np.clip(pert[i], bounds[i][0], bounds[i][1])
            try:
                res2 = minimize(
                    _score_neg, pert,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                s = _score(res2.x)
                if s > best_score:
                    best_score = s
                    best_params = res2.x.copy()
            except Exception:
                pass
    
    # Optimize point angles on best Fourier shape
    if best_params is not None and time.time() < DEADLINE - 5:
        a0 = best_params[0]
        coeffs = best_params[1:]
        
        def _score_angles(angle_params):
            theta = np.sort(angle_params)
            pts = _fourier_points(theta, a0, coeffs)
            min_d = np.inf
            for i in range(N - 1):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_d:
                        min_d = d
            if min_d < 1e-8:
                return 0.0
            ta = _min_triangle_area(pts)
            ha = _hull_area(pts)
            if ha < 1e-10:
                return 0.0
            return -ta / ha
        
        init_angles = 2.0 * np.pi * np.arange(N) / N
        try:
            res_ang = minimize(
                _score_angles, init_angles,
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            theta_opt = np.sort(res_ang.x)
            pts_opt = _fourier_points(theta_opt, a0, coeffs)
            ta = _min_triangle_area(pts_opt)
            ha = _hull_area(pts_opt)
            s = ta / ha if ha > 1e-10 else 0.0
            if s > best_score:
                return pts_opt
        except Exception:
            pass
    
    # Final: convert best Fourier params to 13 evenly-spaced points
    a0 = best_params[0]
    coeffs = best_params[1:]
    theta = 2.0 * np.pi * np.arange(N) / N
    pts = _fourier_points(theta, a0, coeffs)
    
    ta = _min_triangle_area(pts)
    ha = _hull_area(pts)
    if ha < 1e-10 or ta < 1e-10:
        theta_reg = 2.0 * np.pi * np.arange(N) / N
        pts = np.column_stack([np.cos(theta_reg), np.sin(theta_reg)])
    
    return pts
# EVOLVE-BLOCK-END
