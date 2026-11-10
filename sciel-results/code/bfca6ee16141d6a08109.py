import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 8

    def fast_log_score(x):
        """x: 16-vector (8 points, 2 coords each). Returns log(score) or -inf."""
        pts = x.reshape(8, 2)
        # Compute all pairwise distances efficiently
        # Using broadcasting: 8x8 distance matrix
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Upper triangle indices (i < j)
        iu = np.triu_indices(8, k=1)
        d = dists[iu]
        if np.any(d < 1e-12):
            return -np.inf
        D = np.max(d)
        if D < 1e-12:
            return -np.inf
        # Scale so diameter = 2
        # log(score) = 2 * sum_{i<j} log(d_ij * 2/D) - 8*log(8)
        #            = 2 * [sum log(d_ij) + 28*log(2) - 28*log(D)] - 8*log(8)
        sum_log_d = np.sum(np.log(d))
        log_score = 2.0 * (sum_log_d + 28.0 * np.log(2.0) - 28.0 * np.log(D)) - 8.0 * np.log(8.0)
        return log_score

    def objective(x):
        """For minimization: return -log_score."""
        if time.time() > deadline:
            return 1e10
        s = fast_log_score(x)
        if np.isneginf(s) or np.isnan(s):
            return 1e10
        return -s

    def powell_refine(x0, max_iter=300):
        """Run Powell from x0, return best (score, points)."""
        if time.time() > deadline - 5.0:
            return -np.inf, x0.reshape(8, 2)
        try:
            res = minimize(
                objective, x0, method='Powell',
                options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            s = -res.fun
            if np.isfinite(s) and s > -1e9:
                return s, res.x.reshape(8, 2)
        except Exception:
            pass
        s0 = fast_log_score(x0)
        return s0, x0.reshape(8, 2)

    best_score = -np.inf
    best_points = None

    # Strategy 1: Regular octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    s_oct = fast_log_score(pts_oct.flatten())
    if s_oct > best_score:
        best_score = s_oct
        best_points = pts_oct.copy()

    # Strategy 2: Two concentric squares with relative rotation
    # Parameters: R1, R2, theta (rotation of second square)
    # We'll search over these and then refine with full 16-param Powell
    def two_squares_to_16(R1, R2, theta):
        a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([R1*np.cos(a1), R1*np.sin(a1)])
        a2 = a1 + theta
        p2 = np.column_stack([R2*np.cos(a2), R2*np.sin(a2)])
        return np.vstack([p1, p2]).flatten()

    # Grid search over two-square parameters, then Powell refine top candidates
    R1_vals = np.linspace(0.5, 3.0, 12)
    R2_vals = np.linspace(0.5, 3.0, 12)
    theta_vals = np.linspace(0.0, np.pi/4, 8)

    candidates = []
    for R1 in R1_vals:
        for R2 in R2_vals:
            for th in theta_vals:
                x = two_squares_to_16(R1, R2, th)
                s = fast_log_score(x)
                if np.isfinite(s):
                    candidates.append((s, x))

    # Sort by score, take top candidates for Powell refinement
    candidates.sort(key=lambda t: -t[0])
    top_k = min(15, len(candidates))
    for s_init, x_init in candidates[:top_k]:
        if time.time() > deadline - 15.0:
            break
        s_ref, pts_ref = powell_refine(x_init)
        if s_ref > best_score:
            best_score = s_ref
            best_points = pts_ref.copy()

    # Strategy 3: Random multi-start Powell
    # Generate diverse random configurations and refine
    rng = np.random.default_rng(42)
    for trial in range(30):
        if time.time() > deadline - 10.0:
            break
        # Random points in unit disk, then scale
        pts = rng.uniform(-1, 1, (8, 2))
        # Add some structure: mix of symmetric and random
        if trial % 3 == 0:
            # Near-regular octagon with perturbation
            a = np.linspace(0, 2*np.pi, 8, endpoint=False) + rng.uniform(-0.1, 0.1, 8)
            r = 1.0 + rng.uniform(-0.2, 0.2, 8)
            pts = np.column_stack([r*np.cos(a), r*np.sin(a)])
        elif trial % 3 == 1:
            # Two squares with random params
            R1 = rng.uniform(0.5, 2.5)
            R2 = rng.uniform(0.5, 2.5)
            th = rng.uniform(0, np.pi/4)
            pts = two_squares_to_16(R1, R2, th).reshape(8, 2)
        else:
            # Fully random
            pts = rng.uniform(-2, 2, (8, 2))

        s_init = fast_log_score(pts.flatten())
        if np.isneginf(s_init):
            continue
        s_ref, pts_ref = powell_refine(pts.flatten(), max_iter=200)
        if s_ref > best_score:
            best_score = s_ref
            best_points = pts_ref.copy()

    # Strategy 4: Differential evolution on 16 params if time permits
    if time.time() < deadline - 40.0:
        try:
            # Use the best found as a starting hint via bounds
            if best_points is not None:
                # Set bounds around reasonable range
                bounds = [(-3.0, 3.0)] * 16
                # Add constraint: at least one pair has distance > 0.5 (handled by score)
                de_res = differential_evolution(
                    objective, bounds,
                    maxiter=100, popsize=15, tol=1e-6,
                    seed=123, init='sobol', workers=1
                )
                if de_res.fun < 1e9:
                    s_de = -de_res.fun
                    if s_de > best_score:
                        best_score = s_de
                        best_points = de_res.x.reshape(8, 2)
                    # Refine DE result
                    s_ref, pts_ref = powell_refine(de_res.x, max_iter=200)
                    if s_ref > best_score:
                        best_score = s_ref
                        best_points = pts_ref.copy()
        except Exception:
            pass

    # Final: one more Powell polish on the best
    if best_points is not None and time.time() < deadline - 5.0:
        s_final, pts_final = powell_refine(best_points.flatten(), max_iter=500)
        if s_final > best_score:
            best_score = s_final
            best_points = pts_final

    if best_points is None:
        best_points = pts_oct

    # Scale to diameter 2
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    D = np.max(dists)
    if D > 0:
        best_points = best_points * (2.0 / D)

    return [[float(p[0]), float(p[1])] for p in best_points]
