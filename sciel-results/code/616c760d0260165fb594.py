"""Optimized 16-point placement: minimize (max_dist/min_dist)^2."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2
DEADLINE = 140.0  # safety margin before 150s kill


def _pairwise_dists(pts):
    """Compute all pairwise squared distances. pts: (16,2) -> (120,) array of d^2."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]


def _objective(params, mode='soft'):
    """
    params: (32,) flattened points
    Returns a smooth surrogate for (max/min)^2.
    Uses log-sum-exp based soft max and soft min on squared distances.
    """
    pts = params.reshape(N, D)
    d2 = _pairwise_dists(pts)
    # Guard against zero distances
    d2 = np.maximum(d2, 1e-12)
    
    if mode == 'soft':
        # Soft max: log-sum-exp with temperature
        temp = 0.01
        m = np.max(d2)
        soft_max = m + temp * np.log(np.sum(np.exp((d2 - m) / temp)))
        # Soft min: -log-sum-exp of negatives
        m2 = np.min(d2)
        soft_min = m2 - temp * np.log(np.sum(np.exp(-(d2 - m2) / temp)))
        ratio2 = soft_max / soft_min
        return ratio2
    else:
        # Exact objective
        return np.max(d2) / np.min(d2)


def _exact_score(pts):
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)


def construct():
    t0 = time.time()
    
    # ---- Phase 1: Global search with Differential Evolution on low-dim params ----
    # Use a parameterization: 4x4 grid with sinusoidal deformation
    # params: [a1, a2, b1, b2, c1, c2, phase1, phase2, scale]
    # x = base_x + a1*sin(b1*x + phase1)*scale + c1*x
    # y = base_y + a2*sin(b2*y + phase2)*scale + c2*y
    
    def _de_objective(p):
        a1, a2, b1, b2, c1, c2, ph1, ph2, scale = p
        # 4x4 base grid
        gx = np.array([0, 1, 2, 3], dtype=float)
        gy = np.array([0, 1, 2, 3], dtype=float)
        X, Y = np.meshgrid(gx, gy)
        X = X.flatten()
        Y = Y.flatten()
        # Deformation
        X = X + a1 * np.sin(b1 * X + ph1) * scale + c1 * X
        Y = Y + a2 * np.sin(b2 * Y + ph2) * scale + c2 * Y
        pts = np.column_stack([X, Y])
        return _objective(pts.flatten(), mode='soft')
    
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0.5, 2.0), (0.5, 2.0),
              (-0.3, 0.3), (-0.3, 0.3), (0.0, 2*np.pi), (0.0, 2*np.pi),
              (0.0, 0.5)]
    
    best_de = None
    try:
        result_de = differential_evolution(
            _de_objective, bounds, seed=42, maxiter=200, tol=1e-8,
            popsize=15, polish=True, init='sobol',
            time_limit=60
        )
        if time.time() - t0 < DEADLINE:
            best_de = result_de.x
    except Exception:
        pass
    
    # Also try direct 32D DE with smaller budget
    try:
        if time.time() - t0 < DEADLINE - 30:
            result_de2 = differential_evolution(
                lambda p: _objective(p, mode='soft'),
                [(-3, 3)] * 32, seed=123, maxiter=100, tol=1e-6,
                popsize=10, polish=False, init='sobol',
                time_limit=30
            )
            if time.time() - t0 < DEADLINE:
                if best_de is None or result_de2.fun < 100:  # sanity check
                    # Convert DE result to points
                    pts2 = result_de2.x.reshape(N, D)
                    score2 = _exact_score(pts2)
                    if best_de is not None:
                        # Compare with best_de converted
                        a1, a2, b1, b2, c1, c2, ph1, ph2, scale = best_de
                        gx = np.array([0, 1, 2, 3], dtype=float)
                        gy = np.array([0, 1, 2, 3], dtype=float)
                        X, Y = np.meshgrid(gx, gy)
                        X = X.flatten() + a1 * np.sin(b1 * X + ph1) * scale + c1 * X
                        Y = Y.flatten() + a2 * np.sin(b2 * Y + ph2) * scale + c2 * Y
                        pts_de = np.column_stack([X, Y])
                        score_de = _exact_score(pts_de)
                        if score2 < score_de:
                            best_de = None  # prefer direct
                    else:
                        best_de = None
                # We'll track best points directly
    except Exception:
        pass
    
    # Collect candidate points from all sources
    candidates = []
    
    # From DE low-dim
    if best_de is not None:
        a1, a2, b1, b2, c1, c2, ph1, ph2, scale = best_de
        gx = np.array([0, 1, 2, 3], dtype=float)
        gy = np.array([0, 1, 2, 3], dtype=float)
        X, Y = np.meshgrid(gx, gy)
        X = X.flatten() + a1 * np.sin(b1 * X + ph1) * scale + c1 * X
        Y = Y.flatten() + a2 * np.sin(b2 * Y + ph2) * scale + c2 * Y
        candidates.append(np.column_stack([X, Y]))
    
    # From direct DE
    if time.time() - t0 < DEADLINE - 20:
        try:
            result_de2 = differential_evolution(
                lambda p: _objective(p, mode='soft'),
                [(-3, 3)] * 32, seed=123, maxiter=100, tol=1e-6,
                popsize=10, polish=False, init='sobol',
                time_limit=25
            )
            candidates.append(result_de2.x.reshape(N, D))
        except Exception:
            pass
    
    # Random restarts
    for seed in range(5):
        if time.time() - t0 > DEADLINE - 20:
            break
        rng = np.random.RandomState(seed * 100 + 7)
        pts = rng.randn(N, D) * 1.5
        candidates.append(pts)
    
    # ---- Phase 2: Polish each candidate with L-BFGS-B on soft objective ----
    best_pts = None
    best_score = np.inf
    
    for i, init_pts in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        init_flat = init_pts.flatten()
        
        try:
            res = minimize(
                lambda p: _objective(p, mode='soft'),
                init_flat,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if time.time() - t0 > DEADLINE - 5:
                break
            pts = res.x.reshape(N, D)
            score = _exact_score(pts)
            if score < best_score:
                best_score = score
                best_pts = pts
        except Exception:
            continue
    
    # ---- Phase 3: Final high-precision polish on best point ----
    if best_pts is not None and time.time() - t0 < DEADLINE - 3:
        try:
            res_final = minimize(
                lambda p: _objective(p, mode='soft'),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            pts_final = res_final.x.reshape(N, D)
            score_final = _exact_score(pts_final)
            if score_final < best_score:
                best_pts = pts_final
                best_score = score_final
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
    
    return best_pts


if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
