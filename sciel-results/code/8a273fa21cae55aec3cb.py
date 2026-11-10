import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.deg2rad(angle_deg)
    th = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)

def _all_verts(centers):
    th = np.deg2rad(60 * np.arange(6))
    off = np.stack([np.cos(th), np.sin(th)], axis=1)
    return centers[:, None, :] + off[None, :, :]

def _required_s(centers, cx, cy, outer_angle):
    verts = _all_verts(centers)
    a = np.deg2rad(outer_angle)
    nk = a + np.deg2rad(30 + 60 * np.arange(6))
    normals = np.stack([np.cos(nk), np.sin(nk)], axis=1)
    d = verts - np.array([cx, cy])
    proj = np.einsum('nvi,ki->nvk', d, normals)
    m = proj.max()
    return max((2.0 / np.sqrt(3.0)) * m, 1e-6)

def _pair_sep(A, B):
    best = np.inf
    for poly in (A, B):
        for k in range(6):
            e = poly[(k + 1) % 6] - poly[k]
            n = np.array([-e[1], e[0]])
            pa = A @ n
            pb = B @ n
            sep = min(pa.max() - pb.min(), pb.max() - pa.min())
            if sep < best:
                best = sep
    return best

def _valid(inner, outer, s):
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(s)):
        return False
    if s < np.sqrt(12) - 1e-6:
        return False
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for k in range(6):
        for l in range(k + 1, 6):
            if np.linalg.norm(ov[k] - ov[l]) < 1e-6:
                return False
    verts = _all_verts(inner[:, :2])
    for i in range(12):
        for j in range(i + 1, 12):
            if _pair_sep(verts[i], verts[j]) >= 0:
                return False
    for i in range(12):
        for k in range(6):
            p1 = ov[k]
            p2 = ov[(k + 1) % 6]
            cross = (p2[0] - p1[0]) * (verts[i, k, 1] - p1[1]) - (p2[1] - p1[1]) * (verts[i, k, 0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def hexagon_packing_12():
    t0 = time.time()
    deadline = t0 + 595

    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
             (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]
    center = (0, 0)

    def to_xy(q, r):
        return (np.sqrt(3) * (q + r / 2.0), 1.5 * r)

    base = [center] + ring1

    # Stage 1: honeycomb subsets, optimize enclosing hexagon for each
    best = None
    for start in range(12):
        if time.time() > deadline:
            break
        chosen5 = [ring2[(start + i) % 12] for i in range(5)]
        centers = np.array([to_xy(q, r) for q, r in base + chosen5])

        def obj(p):
            return _required_s(centers, p[0], p[1], p[2])

        c0 = centers.mean(axis=0)
        res = minimize(obj, [c0[0], c0[1], 0.0], method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12})
        s = obj(res.x)
        if best is None or s < best[0]:
            best = (s, centers.copy(), res.x.copy())

    s_hc, centers_hc, outer_hc = best

    def joint_obj(p, w):
        centers = p[:24].reshape(12, 2)
        cx, cy, ang = p[24], p[25], p[26]
        s = _required_s(centers, cx, cy, ang)
        verts = _all_verts(centers)
        pen = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                sep = _pair_sep(verts[i], verts[j])
                if sep > 0:
                    pen += sep
        return s + w * pen

    def refine(x0):
        cur_s = None
        cur_inner = None
        cur_outer = None
        for w in (20.0, 10.0, 5.0, 2.0, 1.0):
            if time.time() > deadline:
                break
            res = minimize(lambda p: joint_obj(p, w), x0, method='Nelder-Mead',
                           options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-11})
            p = res.x
            centers = p[:24].reshape(12, 2)
            cx, cy, ang = p[24], p[25], p[26]
            s = _required_s(centers, cx, cy, ang)
            inner = np.zeros((12, 3))
            inner[:, :2] = centers
            inner[:, 2] = 0.0
            outer = np.array([cx, cy, ang])
            if _valid(inner, outer, s):
                if cur_s is None or s < cur_s:
                    cur_s, cur_inner, cur_outer = s, inner.copy(), outer.copy()
                    x0 = np.concatenate([inner[:, :2].ravel(), outer])
        return cur_s, cur_inner, cur_outer

    # Honeycomb fallback (always valid by construction)
    fb_inner = np.zeros((12, 3))
    fb_inner[:, :2] = centers_hc
    fb_inner[:, 2] = 0.0
    fb_outer = outer_hc.copy()
    best_s, best_inner, best_outer = s_hc, fb_inner, fb_outer

    # Joint refinement from honeycomb init
    x0_hc = np.concatenate([centers_hc.ravel(), outer_hc])
    rs, ri, ro = refine(x0_hc)
    if rs is not None and rs < best_s:
        best_s, best_inner, best_outer = rs, ri, ro

    # Joint refinement from a circular init
    if time.time() < deadline:
        ang_c = np.deg2rad(np.arange(12) * 30)
        circ = np.stack([1.8 * np.cos(ang_c), 1.8 * np.sin(ang_c)], axis=1)
        x0_c = np.concatenate([circ.ravel(), outer_hc])
        rs, ri, ro = refine(x0_c)
        if rs is not None and rs < best_s:
            best_s, best_inner, best_outer = rs, ri, ro

    if not _valid(best_inner, best_outer, best_s):
        best_s, best_inner, best_outer = s_hc, fb_inner, fb_outer

    return best_inner, best_outer, best_s
