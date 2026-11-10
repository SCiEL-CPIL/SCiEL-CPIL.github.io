import numpy as np
from scipy.optimize import minimize


def _support_unit_hex(n):
    ang = np.arctan2(n[1], n[0])
    k = np.arange(6) * (np.pi / 3.0)
    return np.max(np.cos(ang - k))


def _make_centers(shape, scale):
    e1 = np.array([np.sqrt(3.0), 0.0]) * scale
    e2 = np.array([np.sqrt(3.0) / 2.0, 1.5]) * scale
    return np.array([i * e1 + j * e2 for i, j in shape])


def _s_value(params, centers, base_normals):
    cx, cy, phi = params
    c = np.array([cx, cy])
    ca, sa = np.cos(phi), np.sin(phi)
    R = np.array([[ca, -sa], [sa, ca]])
    ns = base_normals @ R.T
    rel = centers - c
    proj = rel @ ns.T
    maxproj = np.max(proj, axis=0)
    sup = np.array([_support_unit_hex(ns[i]) for i in range(6)])
    M = maxproj + sup
    return (2.0 / np.sqrt(3.0)) * np.max(M)


def hexagon_packing_12():
    scale = 1.0 + 1e-4
    shapes = [
        [(i, j) for i in range(3) for j in range(4)],
        [(i, j) for i in range(4) for j in range(3)],
        [(i, j) for i in range(2) for j in range(6)],
        [(i, j) for i in range(6) for j in range(2)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1), (0, -2)],
    ]
    base_normals = np.array(
        [[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)]
    )

    best_s = np.inf
    best_result = None
    for shape in shapes:
        centers = _make_centers(shape, scale)
        centroid = np.mean(centers, axis=0)
        cc = centers - centroid
        best_local = None
        best_local_s = np.inf
        for phi0 in np.linspace(0, np.pi / 3, 30):
            for cx0 in np.linspace(-2.0, 2.0, 9):
                for cy0 in np.linspace(-2.0, 2.0, 9):
                    s0 = _s_value([cx0, cy0, phi0], cc, base_normals)
                    if s0 < best_local_s:
                        best_local_s = s0
                        best_local = [cx0, cy0, phi0]
        try:
            res = minimize(
                _s_value, best_local, args=(cc, base_normals), method="Nelder-Mead",
                options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 50000, "maxfev": 50000},
            )
            s_cand = res.fun
            params_cand = res.x
        except Exception:
            s_cand = best_local_s
            params_cand = best_local
        if s_cand < best_s:
            best_s = s_cand
            best_result = (centers, centroid, params_cand)

    centers, centroid, params = best_result
    cx, cy, phi = params
    s = best_s * (1.0 + 1e-9)
    inner = np.zeros((12, 3))
    for i, (x, y) in enumerate(centers):
        inner[i] = [x, y, 0.0]
    outer_center = centroid + np.array([cx, cy])
    outer = np.array([outer_center[0], outer_center[1], np.degrees(phi) % 360.0])
    return inner, outer, s
