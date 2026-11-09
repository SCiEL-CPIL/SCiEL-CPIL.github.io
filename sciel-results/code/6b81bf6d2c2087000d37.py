import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0
    n = 20

    def normalize_polygon(pts):
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        dists = np.linalg.norm(centered, axis=1)
        mean_d = np.mean(dists)
        if mean_d <= 0.01:
            return None
        return centered / mean_d

    def is_strictly_convex(pts):
        n_pts = len(pts)
        crosses = []
        for i in range(n_pts):
            p0, p1, p2 = pts[i], pts[(i+1) % n_pts], pts[(i+2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            c = v1[0] * v2[1] - v1[1] * v2[0]
            crosses.append(c)
        crosses = np.array(crosses)
        if np.any(np.abs(crosses) < 1e-14):
            return False
        if not (np.all(crosses > 0) or np.all(crosses < 0)):
            return False
        # Check no repeated points
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                    return False
        return True

    def vertex_score(v, all_pts):
        d = np.linalg.norm(all_pts - v, axis=1)
        idx = np.argmin(d)
        d = np.delete(d, idx)
        d_sorted = np.sort(d)
        if d_sorted[0] < 0.05:
            return 0.0
        gaps = np.diff(d_sorted)
        k = int(np.argmin(gaps))
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            return 0.0
        if k == 0:
            sel = d_sorted[0:4]
        elif k == 17:
            sel = d_sorted[15:19]
        else:
            sel = d_sorted[k-1:k+3]
        ratios = np.maximum(sel / D, D / sel)
        raw = np.mean(ratios)
        if raw <= 0:
            return 0.0
        return min(1.0, 1.0 / raw)

    def min_vertex_score(pts):
        norm = normalize_polygon(pts)
        if norm is None:
            return 0.0
        if not is_strictly_convex(norm):
            return 0.0
        scores = [vertex_score(norm[i], norm) for i in range(n)]
        return min(scores)

    def make_regular():
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Start with regular polygon
    best_pts = make_regular()
    best_score = min_vertex_score(best_pts)

    # Parameterize as radial perturbation: r_i = 1 + eps * f(i)
    # Use Fourier series for smoothness
    def params_to_points(params):
        # params: [a1, b1, a2, b2, ..., aM, bM] for Fourier modes
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        r = np.ones(n)
        M = len(params) // 2
        for m in range(1, M+1):
            a = params[2*(m-1)]
            b = params[2*(m-1)+1]
            r += a * np.cos(m * angles) + b * np.sin(m * angles)
        # Ensure r > 0
        r = np.maximum(r, 0.3)
        return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)

    def objective(params):
        if time.time() - start_time > deadline:
            return 10.0
        try:
            pts = params_to_points(params)
            s = min_vertex_score(pts)
            return -s if s > 0 else 5.0
        except:
            return 5.0

    # Multi-start Nelder-Mead
    np.random.seed(42)
    n_params = 12  # 6 Fourier modes
    bounds = [(-0.15, 0.15)] * n_params

    # Generate diverse starting points
    starts = [np.zeros(n_params)]
    for scale in [0.01, 0.03, 0.05, 0.08, 0.12]:
        for _ in range(8):
            p = np.random.uniform(-scale, scale, n_params)
            starts.append(p)

    # Sort starts by initial score
    start_scores = []
    for s in starts:
        try:
            pts = params_to_points(s)
            sc = min_vertex_score(pts)
            start_scores.append((-sc, s))
        except:
            start_scores.append((5.0, s))
    start_scores.sort()

    # Try each start with Nelder-Mead
    for neg_sc, start_p in start_scores:
        if time.time() - start_time > deadline - 20:
            break
        try:
            res = minimize(
                objective, start_p,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if time.time() - start_time > deadline - 5:
                break
            sc = -res.fun
            if 0 < sc < 5 and sc > best_score:
                best_score = sc
                best_pts = params_to_points(res.x)
        except:
            pass

    # Final polish
    if time.time() - start_time < deadline - 10:
        try:
            # Find current best params
            best_params = None
            for s in starts:
                if np.array_equal(s, np.zeros(n_params)):
                    continue
            # Just do one more polish from best_pts by inverting
            # For simplicity, do a final NM from the zero point
            res = minimize(
                objective, np.zeros(n_params),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            sc = -res.fun
            if 0 < sc < 5 and sc > best_score:
                best_score = sc
                best_pts = params_to_points(res.x)
        except:
            pass

    # Final validation
    final_score = min_vertex_score(best_pts)
    if final_score > best_score:
        best_score = final_score

    # Ensure convexity
    if not is_strictly_convex(normalize_polygon(best_pts) if normalize_polygon(best_pts) is not None else best_pts):
        best_pts = make_regular()

    return best_pts
