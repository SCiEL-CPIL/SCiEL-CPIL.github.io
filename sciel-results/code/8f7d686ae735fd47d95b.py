"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area.
Combined approach: symmetric parameterization (11D) + multi-phase search (DE + Nelder-Mead)."""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _params_to_points_symmetric(params):
    """Map 11 parameters to 11 points enforcing reflection symmetry about x=0.5.
    params[0]: y of central point (x=0.5)
    params[1+2i]: x in [0, 0.5] for pair i
    params[2+2i]: y in [0, H] for pair i
    Pair i: (x, y) and (1-x, y)
    """
    points = np.zeros((N, 2))
    points[0, 0] = 0.5
    points[0, 1] = params[0]
    for i in range(5):
        x = params[1 + 2 * i]
        y = params[2 + 2 * i]
        points[1 + 2 * i, 0] = x
        points[1 + 2 * i, 1] = y
        points[2 + 2 * i, 0] = 1.0 - x
        points[2 + 2 * i, 1] = y
    return points


def _is_inside_symmetric(params):
    """Check if all 11 symmetric points are inside the triangle."""
    points = _params_to_points_symmetric(params)
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -1e-9):
        return False
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
    if np.any(y > np.sqrt(3) * (1.0 - x) + 1e-9):
        return False
    return True


def _objective_symmetric(params):
    """Minimize negative min-area for symmetric configuration. Returns large value if outside."""
    points = _params_to_points_symmetric(params)
    x = points[:, 0]
    y = points[:, 1]
    # Boundary penalty (smooth)
    viol = np.maximum(0.0, -y) + np.maximum(0.0, y - np.sqrt(3) * x) + np.maximum(0.0, y - np.sqrt(3) * (1.0 - x))
    if viol.sum() > 1e-6:
        return 1e6 + 1000.0 * viol.sum()
    return -_min_area(points)


def _objective_packing_symmetric(params):
    """Minimize negative min pairwise distance (packing proxy) for symmetric config."""
    points = _params_to_points_symmetric(params)
    x = points[:, 0]
    y = points[:, 1]
    viol = np.maximum(0.0, -y) + np.maximum(0.0, y - np.sqrt(3) * x) + np.maximum(0.0, y - np.sqrt(3) * (1.0 - x))
    if viol.sum() > 1e-6:
        return 1e6 + 1000.0 * viol.sum()
    # Compute all pairwise distances
    d2 = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
    # Mask diagonal
    np.fill_diagonal(d2, np.inf)
    min_d = np.sqrt(d2.min())
    return -min_d


def _symmetric_lattice_init(rng):
    """Generate a symmetric initial configuration on a lattice-like grid."""
    params = np.zeros(11)
    # Central point near centroid
    params[0] = rng.uniform(0.2 * H, 0.5 * H)
    # 5 symmetric pairs: place in rows
    # Bottom row: 2 pairs (4 points)
    # Middle row: 2 pairs (4 points)
    # Top row: 1 pair (2 points)
    row_y = [0.1 * H, 0.15 * H, 0.35 * H, 0.45 * H, 0.65 * H]
    row_x_spread = [0.25, 0.22, 0.18, 0.15, 0.10]
    for i in range(5):
        params[1 + 2 * i] = row_x_spread[i] * rng.uniform(0.7, 1.3)
        params[2 + 2 * i] = row_y[i] * rng.uniform(0.8, 1.2)
    return params


def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_params = None
    best_score = -np.inf

    def _update(params, score):
        nonlocal best_params, best_score
        if score > best_score:
            best_params = params.copy()
            best_score = score

    # Phase 1: Lattice-based symmetric initialization + packing polish
    try:
        rng0 = np.random.RandomState(42)
        for attempt in range(5):
            if time.time() > deadline - 60.0:
                break
            params = _symmetric_lattice_init(rng0)
            # Clamp to valid range
            params[0] = np.clip(params[0], 0.0, H)
            for i in range(5):
                params[1 + 2 * i] = np.clip(params[1 + 2 * i], 0.0, 0.5)
                params[2 + 2 * i] = np.clip(params[2 + 2 * i], 0.0, H)

            # Packing polish
            res_p = minimize(
                _objective_packing_symmetric,
                params,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            # Exact polish
            res_e = minimize(
                _objective_symmetric,
                res_p.x,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score = -res_e.fun
            if score > 0:  # valid
                _update(res_e.x, score)
    except Exception:
        pass

    # Phase 2: Differential evolution on symmetric space (11D)
    if time.time() < deadline - 45.0:
        try:
            bounds = [(0.0, H)]
            for _ in range(5):
                bounds.append((0.0, 0.5))
                bounds.append((0.0, H))

            result = differential_evolution(
                _objective_symmetric,
                bounds=bounds,
                maxiter=100,
                popsize=30,
                tol=1e-6,
                seed=42,
                polish=False,
                updating='immediate',
                init='sobol'
            )
            score_de = -_objective_symmetric(result.x)
            if score_de > 0:
                _update(result.x, score_de)
        except Exception:
            pass

    # Phase 3: Polish the best symmetric solution with exact objective
    if best_params is not None and time.time() < deadline - 25.0:
        try:
            res3 = minimize(
                _objective_symmetric,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            score3 = -res3.fun
            if score3 > 0:
                _update(res3.x, score3)
        except Exception:
            pass

    # Phase 4: Multiple restarts with perturbations around best
    if best_params is not None and time.time() < deadline - 15.0:
        rng = np.random.RandomState(777)
        for restart in range(15):
            if time.time() > deadline - 15.0:
                break
            try:
                # Perturb the best solution
                noise = rng.randn(11) * 0.05
                params = best_params + noise
                # Clamp
                params[0] = np.clip(params[0], 0.0, H)
                for i in range(5):
                    params[1 + 2 * i] = np.clip(params[1 + 2 * i], 0.0, 0.5)
                    params[2 + 2 * i] = np.clip(params[2 + 2 * i], 0.0, H)

                # Quick polish
                res_r = minimize(
                    _objective_symmetric,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                score_r = -res_r.fun
                if score_r > 0:
                    _update(res_r.x, score_r)
            except Exception:
                pass

    # Phase 5: Also try a few random symmetric starts with full polish
    if time.time() < deadline - 10.0:
        rng2 = np.random.RandomState(1234)
        for r in range(5):
            if time.time() > deadline - 10.0:
                break
            try:
                params = np.zeros(11)
                params[0] = rng2.uniform(0.1 * H, 0.6 * H)
                for i in range(5):
                    params[1 + 2 * i] = rng2.uniform(0.05, 0.45)
                    params[2 + 2 * i] = rng2.uniform(0.05 * H, 0.7 * H)

                res_r = minimize(
                    _objective_symmetric,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                score_r = -res_r.fun
                if score_r > 0:
                    _update(res_r.x, score_r)
            except Exception:
                pass

    # Phase 6: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _objective_symmetric,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_f = -res_final.fun
            if score_f > 0:
                _update(res_final.x, score_f)
        except Exception:
            pass

    # Convert to points
    if best_params is not None:
        points = _params_to_points_symmetric(best_params)
        # Final safety: clamp any points that might be slightly outside
        for i in range(N):
            x, y = points[i]
            if y < 0:
                points[i, 1] = 0.0
            if y > np.sqrt(3) * x:
                # Project to edge y = sqrt(3)*x
                points[i, 1] = np.sqrt(3) * x
            if y > np.sqrt(3) * (1.0 - x):
                points[i, 1] = np.sqrt(3) * (1.0 - x)
        return points
    else:
        # Fallback: random symmetric points
        rng = np.random.RandomState(0)
        points = np.zeros((N, 2))
        points[0] = [0.5, H / 3]
        for i in range(5):
            x = rng.uniform(0.1, 0.4)
            y = rng.uniform(0.05 * H, 0.6 * H)
            points[1 + 2 * i] = [x, y]
            points[2 + 2 * i] = [1.0 - x, y]
        return points
