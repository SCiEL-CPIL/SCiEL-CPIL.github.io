import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    n = len(points)
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def heilbronn_convex13() -> np.ndarray:
    n = 13
    deadline = time.time() + 600

    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)

    def _signed_areas(points):
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        return (
            p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
            p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
            p3[:, 0] * (p1[:, 1] - p2[:, 1])
        )

    def _subgrad_ascent(points, max_iters=3000, base_lr=0.03):
        pts = points.copy()
        best_pts = pts.copy()
        best_score = _score(pts)
        lr = base_lr

        for it in range(max_iters):
            if time.time() > deadline - 30:
                break

            sa = _signed_areas(pts)
            areas = 0.5 * np.abs(sa)
            min_val = np.min(areas)

            threshold = min_val * 1.01
            mask = areas <= threshold
            count = np.sum(mask)
            if count == 0:
                break

            grad = np.zeros((n, 2))
            for idx in np.where(mask)[0]:
                i, j, k = tri_indices[idx]
                S = sa[idx]
                s = 1.0 if S >= 0 else -1.0
                p1, p2, p3 = pts[i], pts[j], pts[k]
                grad[i] += s * np.array([p2[1] - p3[1], p3[0] - p2[0]])
                grad[j] += s * np.array([p3[1] - p1[1], p1[0] - p3[0]])
                grad[k] += s * np.array([p1[1] - p2[1], p2[0] - p1[0]])
            grad /= count

            gn = np.linalg.norm(grad)
            if gn < 1e-15:
                break
            grad /= gn

            new_pts = pts + lr * grad
            ns = _score(new_pts)
            if ns > best_score:
                best_score = ns
                best_pts = new_pts.copy()
                pts = new_pts
            else:
                ok = False
                for f in [0.5, 0.25, 0.1, 0.05]:
                    np_ = pts + lr * f * grad
                    ns2 = _score(np_)
                    if ns2 > best_score:
                        best_score = ns2
                        best_pts = np_.copy()
                        pts = np_
                        ok = True
                        break
                if not ok:
                    lr *= 0.9
                    if lr < 1e-7:
                        break

        return best_pts, best_score

    inits = []
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack((np.cos(angles), np.sin(angles))))

    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    poly12 = np.column_stack((np.cos(angles12), np.sin(angles12)))
    inits.append(np.vstack([poly12, [[0, 0]]]))

    rng = np.random.default_rng(42)
    for s in range(8):
        inits.append(rng.uniform(-1, 1, size=(n, 2)))

    best_points = inits[0]
    best_score = _score(best_points)

    for init in inits:
        if time.time() > deadline - 60:
            break
        rpts, rscore = _subgrad_ascent(init, max_iters=2000, base_lr=0.03)
        if rscore > best_score:
            best_score = rscore
            best_points = rpts

    if time.time() < deadline - 30:
        def _obj(x):
            pts = x.reshape(n, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            areas = _triangle_areas(pts)
            a_min = np.min(areas)
            if a_min < 1e-15:
                return 1e10
            try:
                hull = ConvexHull(pts)
                a_hull = hull.volume
                if a_hull < 1e-15:
                    return 1e10
                return -(a_min / a_hull)
            except Exception:
                return 1e10

        try:
            res = minimize(_obj, best_points.flatten(), method='nelder-mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            np_ = res.x.reshape(n, 2)
            ns = _score(np_)
            if ns > best_score:
                best_score = ns
                best_points = np_
        except Exception:
            pass

    if _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
