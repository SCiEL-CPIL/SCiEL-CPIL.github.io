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


def _soft_objective(params, temp=0.05):
    """
    Smooth surrogate for (max/min)^2 using log-sum-exp.
    params: (32,) flattened points
    """
    pts = params.reshape(N, D)
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-12)
    
    # Soft max via log-sum-exp
    m = np.max(d2)
    soft_max = m + temp * np.log(np.sum(np.exp((d2 - m) / temp)))
    
    # Soft min via negative log-sum-exp
    m2 = np.min(d2)
    soft_min = m2 - temp * np.log(np.sum(np.exp(-(d2 - m2) / temp)))
    
    return soft_max / soft_min


def _exact_score(pts):
    """Compute exact (max_dist/min_dist)^2."""
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)


def _grid_init(scale=1.0, jitter=0.0, seed=42):
    """Create a 4x4 grid with optional jitter."""
    rng = np.random.RandomState(seed)
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    X = X.flatten() * scale
    Y = Y.flatten() * scale
    if jitter > 0:
        X += rng.randn(N) * jitter
        Y += rng.randn(N) * jitter
    return np.column_stack([X, Y])


def _hex_init(scale=1.0):
    """Create a hexagonal-like arrangement for 16 points."""
    # 4 rows of 4, offset every other row
    pts = []
    row_spacing = 0.866 * scale  # hexagonal spacing
    col_spacing = scale
    for row in range(4):
        offset = 0.5 * col_spacing if row % 2 == 1 else 0.0
        for col in range(4):
            x = col * col_spacing + offset
            y = row * row_spacing
            pts.append([x, y])
    return np.array(pts)


def _circular_init(scale=1.0):
    """Create points on a circle with some interior points."""
    pts = []
    # 12 points on outer circle
    for i in range(12):
        angle = 2 * np.pi * i / 12
        pts.append([scale * np.cos(angle), scale * np.sin(angle)])
    # 4 points in inner circle
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi/4
        pts.append([0.5 * scale * np.cos(angle), 0.5 * scale * np.sin(angle)])
    return np.array(pts)


def _local_polish(pts, temp=0.01, maxiter=10000, ftol=1e-14, gtol=1e-10):
    """Polish points using L-BFGS-B on soft objective."""
    init_flat = pts.flatten()
    
    res = minimize(
        lambda p: _soft_objective(p, temp=temp),
        init_flat,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return res.x.reshape(N, D)


def _coordinate_descent(pts, max_iter=200, tol=1e-8):
    """
    Exact coordinate descent: optimize each point's x and y coordinates
    to minimize the exact (max/min)^2 objective.
    """
    current_score = _exact_score(pts)
    pts = pts.copy()
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            for coord in range(D):
                # Try small perturbations in both directions
                best_val = pts[i, coord]
                best_score = current_score
                
                # Adaptive step size
                step = 0.1
                for _ in range(10):
                    for direction in [1, -1]:
                        test_pts = pts.copy()
                        test_pts[i, coord] += direction * step
                        test_score = _exact_score(test_pts)
                        if test_score < best_score - tol:
                            best_score = test_score
                            best_val = test_pts[i, coord]
                    
                    # If we improved, try smaller step to refine
                    if best_val != pts[i, coord]:
                        step *= 0.5
                    else:
                        step *= 1.5
                        if step > 1.0:
                            step = 1.0
                        break
                
                if abs(best_val - pts[i, coord]) > tol:
                    pts[i, coord] = best_val
                    improved = True
        
        new_score = _exact_score(pts)
        if abs(new_score - current_score) < tol:
            current_score = new_score
            break
        current_score = new_score
    
    return pts


def construct():
    t0 = time.time()
    
    # Generate diverse initial configurations
    inits = []
    
    # 4x4 grid with various scales and jitters
    for scale in [0.8, 1.0, 1.2]:
        for jitter in [0.0, 0.1, 0.2]:
            for seed in [42, 123, 456]:
                inits.append(_grid_init(scale=scale, jitter=jitter, seed=seed))
    
    # Hexagonal arrangements
    for scale in [0.8, 1.0, 1.2]:
        inits.append(_hex_init(scale=scale))
    
    # Circular arrangements
    for scale in [0.8, 1.0, 1.2]:
        inits.append(_circular_init(scale=scale))
    
    # Random inits
    for seed in range(5):
        rng = np.random.RandomState(seed * 7 + 1)
        inits.append(rng.randn(N, D) * 1.5)
    
    best_pts = None
    best_score = np.inf
    
    # Phase 1: Polish each init with L-BFGS-B on soft objective
    for i, init_pts in enumerate(inits):
        if time.time() - t0 > DEADLINE - 15:
            break
        
        try:
            polished = _local_polish(init_pts, temp=0.05, maxiter=3000, ftol=1e-12, gtol=1e-8)
            score = _exact_score(polished)
            if score < best_score:
                best_score = score
                best_pts = polished
        except Exception:
            continue
    
    # Phase 2: Use DE to find better starting points, then polish
    if time.time() - t0 < DEADLINE - 20:
        try:
            def _de_obj(p):
                return _soft_objective(p, temp=0.05)
            
            result_de = differential_evolution(
                _de_obj,
                [(-3, 3)] * 32,
                seed=42,
                maxiter=150,
                tol=1e-6,
                popsize=12,
                polish=False,
                init='sobol',
                time_limit=25
            )
            
            if time.time() - t0 < DEADLINE - 10:
                de_pts = result_de.x.reshape(N, D)
                polished_de = _local_polish(de_pts, temp=0.02, maxiter=5000, ftol=1e-13, gtol=1e-9)
                score_de = _exact_score(polished_de)
                if score_de < best_score:
                    best_score = score_de
                    best_pts = polished_de
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish on best point
    if best_pts is not None and time.time() - t0 < DEADLINE - 5:
        try:
            # Very tight soft objective polish
            final_polished = _local_polish(best_pts, temp=0.005, maxiter=20000, ftol=1e-15, gtol=1e-11)
            score_final = _exact_score(final_polished)
            if score_final < best_score:
                best_score = score_final
                best_pts = final_polished
        except Exception:
            pass
    
    # Phase 4: Coordinate descent for exact optimization
    if best_pts is not None and time.time() - t0 < DEADLINE - 3:
        try:
            cd_pts = _coordinate_descent(best_pts, max_iter=50, tol=1e-9)
            score_cd = _exact_score(cd_pts)
            if score_cd < best_score:
                best_score = score_cd
                best_pts = cd_pts
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: simple grid
        best_pts = _grid_init(scale=1.0, jitter=0.0, seed=42)
    
    return best_pts


if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
