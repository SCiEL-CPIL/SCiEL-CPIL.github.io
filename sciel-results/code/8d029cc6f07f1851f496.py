import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
_T = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I0 = _T[:, 0]
_I1 = _T[:, 1]
_I2 = _T[:, 2]


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return None


def _score(pts):
    try:
        if pts is None or pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return 0.0
        ah = _hull_area(pts)
        if ah is None or ah <= 1e-12:
            return 0.0
        p1 = pts[_I0]
        p2 = pts[_I1]
        p3 = pts[_I2]
        cross = p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1])
        a = 0.5 * np.abs(cross)
        amin = np.min(a)
        if amin <= 1e-15:
            return 0.0
        return amin / ah
    except Exception:
        return 0.0


def _norm_unit(pts):
    ah = _hull_area(pts)
    if ah is None or ah <= 1e-12:
        return None
    c = pts.mean(axis=0)
    return (pts - c) / np.sqrt(ah)


def _smooth_obj(pts, beta):
    p1 = pts[_I0]
    p2 = pts[_I1]
    p3 = pts[_I2]
    cross = p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    a = 0.5 * np.abs(cross)
    m = np.min(a)
    if m < 1e-15:
        return 1e12, None
    z = a / m
    z = np.clip(z, 0.0, 50.0)
    sm = np.logaddexp.reduce(z) - np.log(z.size)
    return beta * sm, a


def _grad(pts, beta):
    val, a = _smooth_obj(pts, beta)
    if a is None:
        return val, None
    m = np.min(a)
    z = np.clip(a / m, 0.0, 50.0)
    w = np.exp(z - np.logaddexp.reduce(z))
    w = w / w.sum()
    p1 = pts[_I0]
    p2 = pts[_I1]
    p3 = pts[_I2]
    sign = np.sign(0.5 * (p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1])))
    sign = np.where(sign == 0.0, 1.0, sign)
    c = beta * w * sign
    g = np.zeros((N, 2))
    np.add.at(g, _I0, c[:, None] * (p2 - p3).T)
    np.add.at(g, _I1, c[:, None] * (p3 - p1).T)
    np.add.at(g, _I2, c[:, None] * (p1 - p2).T)
    return val, g


def _descend(pts, deadline):
    pts = _norm_unit(pts)
    if pts is None:
        return pts
    beta = 25.0
    lr = 0.05
    best = pts.copy()
    best_val = None
    val, g = _grad(pts, beta)
    if g is None:
        return pts
    for it in range(100000):
        if time.time() > deadline:
            break
        step = lr * g
        stepn = np.linalg.norm(step)
        if stepn > 0.5:
            step = step * (0.5 / stepn)
        cand = pts + step
        cand = _norm_unit(cand)
        if cand is None:
            lr *= 0.5
            if lr < 1e-5:
                break
            continue
        v, g2 = _grad(cand, beta)
        if g2 is None:
            lr *= 0.5
            if lr < 1e-5:
                break
            continue
        if best_val is None or v < best_val:
            best_val = v
            best = cand.copy()
        if v < val:
            val = v
            pts = cand
            g = g2
            if np.linalg.norm(g) < 1e-6:
                break
        else:
            lr *= 0.5
            if lr < 1e-5:
                break
        if it % 300 == 0:
            beta = min(beta * 1.25, 300.0)
    return best


def heilbronn_convex13() -> np.ndarray:
    deadline = time.time() + 580
    rng = np.random.default_rng(20240517)
    inits = []
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack((np.cos(ang), np.sin(ang))))
    a12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    inits.append(np.vstack([np.column_stack((np.cos(a12), np.sin(a12))), [[0.0, 0.0]]]))
    for r in (0.5, 0.7, 0.9):
        inits.append(r * np.column_stack((np.cos(ang), np.sin(ang))))
    for _ in range(6):
        inits.append(np.column_stack((np.cos(ang + rng.uniform(0, 2 * np.pi)), np.sin(ang + rng.uniform(0, 2 * np.pi)))))
    for _ in range(40):
        inits.append(rng.uniform(-1, 1, size=(N, 2)))

    best = None
    best_s = -1.0
    for i, init in enumerate(inits):
        if time.time() > deadline - 25:
            break
        per = min(1.8, max(0.3, (deadline - 25 - time.time()) / max(1, len(inits) - i)))
        cand = _descend(init, time.time() + per)
        if cand is None:
            continue
        s = _score(cand)
        if s > best_s:
            best_s = s
            best = cand

    def _obj(x):
        pts = x.reshape(N, 2)
        if not np.all(np.isfinite(pts)):
            return 1e12
        ah = _hull_area(pts)
        if ah is None or ah <= 1e-12:
            return 1e12
        p1 = pts[_I0]
        p2 = pts[_I1]
        p3 = pts[_I2]
        cross = p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1])
        amin = np.min(0.5 * np.abs(cross))
        if amin <= 1e-15:
            return 1e12
        return -(amin / ah)

    if best is not None and time.time() < deadline - 5:
        try:
            res = minimize(_obj, best.flatten(), method='nelder-mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            np_ = res.x.reshape(N, 2)
            if _score(np_) > best_s:
                best = np_
        except Exception:
            pass

    if best is None or _score(best) <= 0:
        best = np.column_stack((np.cos(ang), np.sin(ang)))
    return best
