"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle


def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    n = len(points)
    # Generate all index triples
    idx = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    # Point = u*V1 + v*V2 (since V0 is origin)
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    # Solve: x = u*1 + v*0.5, y = u*0 + v*sqrt(3)/2
    # v = 2*y/sqrt(3), u = x - 0.5*v
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])


def _objective_bary(x):
    """Objective: negative min triangle area, with points in barycentric coords."""
    bary = x.reshape(-1, 2)
    # Clamp to valid barycentric region
    bary = np.clip(bary, 0.0, None)
    # Ensure u + v <= 1
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)


def _make_initial_bary(seed, strategy='grid'):
    """Generate initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    if strategy == 'grid':
        # Use vertices, edge midpoints, and interior points
        bary_list = [
            [0, 0],      # V0
            [1, 0],      # V1
            [0, 1],      # V2
            [0.5, 0],    # midpoint V0-V1
            [0.5, 0.5],  # midpoint V1-V2
            [0, 0.5],    # midpoint V2-V0
        ]
        # Add interior points
        while len(bary_list) < N:
            u = rng.uniform(0.1, 0.9)
            v = rng.uniform(0.1, 0.9)
            if u + v < 0.8:
                bary_list.append([u, v])
        return np.array(bary_list[:N])
    else:
        # Random spread
        bary_list = []
        for _ in range(200):
            u = rng.uniform(0.05, 0.95)
            v = rng.uniform(0.05, 0.95)
            if u + v < 0.9:
                bary_list.append([u, v])
                if len(bary_list) == N:
                    break
        while len(bary_list) < N:
            u = rng.uniform(0.1, 0.8)
            v = rng.uniform(0.1, 0.8)
            if u + v < 0.85:
                bary_list.append([u, v])
        return np.array(bary_list[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    def time_left():
        return deadline - time.time()

    bounds = [(0.0, 1.0)] * (2 * N)

    # Use Differential Evolution for global search
    # DE is robust and handles the non-differentiability of the clamped objective well
    try:
        de_res = differential_evolution(
            _objective_bary,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False  # We will polish manually to control time
        )
        
        bary_de = de_res.x.reshape(-1, 2)
        bary_de = np.clip(bary_de, 0.0, None)
        s = bary_de[:, 0] + bary_de[:, 1]
        mask = s > 1.0
        if np.any(mask):
            bary_de[mask] = bary_de[mask] / s[mask, None]
        points_de = _bary_to_cart(bary_de)
        score_de = _min_triangle_area_vectorized(points_de)
        
        if score_de > best_score:
            best_score = score_de
            best_points = points_de.copy()
            
    except Exception:
        pass

    # Polish the best result found by DE (or fallback to grid if DE failed)
    if best_points is not None:
        bary_best = _cart_to_bary(best_points)
    else:
        bary_best = _make_initial_bary(0, 'grid')
        points_best = _bary_to_cart(bary_best)
        best_score = _min_triangle_area_vectorized(points_best)
        best_points = points_best.copy()

    # Local optimization with L-BFGS-B to refine the solution
    # Run multiple times with slight perturbations to escape local minima
    rng = np.random.RandomState(123)
    n_polish_restarts = 5
    for i in range(n_polish_restarts):
        if time_left() < 2.0:
            break
        
        if i == 0:
            x0 = bary_best.flatten()
        else:
            # Perturb the best solution
            perturbation_scale = 0.05 * (1 - i * 0.1)
            bary_pert = bary_best + rng.normal(0, perturbation_scale, size=bary_best.shape)
            bary_pert = np.clip(bary_pert, 0.0, None)
            s = bary_pert[:, 0] + bary_pert[:, 1]
            mask = s > 1.0
            if np.any(mask):
                bary_pert[mask] = bary_pert[mask] / s[mask, None]
            x0 = bary_pert.flatten()
            
        try:
            res = minimize(
                _objective_bary, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-10}
            )
            bary_opt = res.x.reshape(-1, 2)
            bary_opt = np.clip(bary_opt, 0.0, None)
            s = bary_opt[:, 0] + bary_opt[:, 1]
            mask = s > 1.0
            if np.any(mask):
                bary_opt[mask] = bary_opt[mask] / s[mask, None]
            points = _bary_to_cart(bary_opt)
            score = _min_triangle_area_vectorized(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                bary_best = bary_opt.copy()
                
        except Exception:
            pass

    # If we still have significant time left, try a few more DE runs with different seeds
    if time_left() > 20.0:
        for seed in [123, 456]:
            if time_left() < 10.0:
                break
            try:
                de_res = differential_evolution(
                    _objective_bary,
                    bounds=bounds,
                    seed=seed,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6,
                    polish=False
                )
                bary_de = de_res.x.reshape(-1, 2)
                bary_de = np.clip(bary_de, 0.0, None)
                s = bary_de[:, 0] + bary_de[:, 1]
                mask = s > 1.0
                if np.any(mask):
                    bary_de[mask] = bary_de[mask] / s[mask, None]
                points_de = _bary_to_cart(bary_de)
                score_de = _min_triangle_area_vectorized(points_de)
                
                if score_de > best_score:
                    best_score = score_de
                    best_points = points_de.copy()
                    
                # Polish this new candidate
                bary_polish = bary_de.copy()
                x0 = bary_polish.flatten()
                res = minimize(
                    _objective_bary, x0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-10}
                )
                bary_opt = res.x.reshape(-1, 2)
                bary_opt = np.clip(bary_opt, 0.0, None)
                s = bary_opt[:, 0] + bary_opt[:, 1]
                mask = s > 1.0
                if np.any(mask):
                    bary_opt[mask] = bary_opt[mask] / s[mask, None]
                points = _bary_to_cart(bary_opt)
                score = _min_triangle_area_vectorized(points)
                
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
            except Exception:
                pass

    # Final fallback
    if best_points is None:
        bary0 = _make_initial_bary(0, 'grid')
        best_points = _bary_to_cart(bary0)

    return np.array(best_points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
