import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11
AP = math.sqrt(3) / 2.0          # apothem of unit hexagon
SQ3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY = 8.0
EPS = 1e-9


def _time_left():
    return (TIME_LIMIT - SAFETY) - (time.time() - START_TIME)


def axial_to_cartesian(axial_list):
    pts = np.empty((len(axial_list), 2))
    for i, (q, r) in enumerate(axial_list):
        pts[i, 0] = SQ3 * (q + r / 2.0)
        pts[i, 1] = 1.5 * r
    return pts


def _rotmat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _build_candidates():
    """Return list of (name, pts) for diverse 11-hexagon layouts."""
    out = []

    def add(name, axial_list, theta=0.0):
        pts = axial_to_cartesian(axial_list)
        if theta != 0.0:
            pts = pts @ _rotmat(theta).T
        out.append((name, pts))

    # 1) Center + 6 ring + 4 spokes
    add("ring_spokes", [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2),
    ])
    # 2) Center + ring + 4 diagonal spokes
    add("ring_diag", [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, -2), (-1, -2),
    ])
    # 3) Center + ring + 4 diagonal spokes (other sign)
    add("ring_diag2", [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (-2, 1), (1, 2), (-1, -2),
    ])
    # 4) Center + ring + 4 axial spokes
    add("ring_axial", [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 1),
    ])
    # 5) 3x3 block
    add("block33", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
    ])
    # 6) 3x4 rectangle
    add("rect34", [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
    ])
    # 7) 4x3 rectangle
    add("rect43", [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
    ])
    # 8) Staggered 3-4-4
    add("stag344", [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
    ])
    # 9) Staggered 4-3-4
    add("stag434", [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
    ])
    # 10) Staggered 4-4-3
    add("stag443", [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
    ])
    # 11) 2-3-3-3
    add("r2333", [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    ])
    # 12) 3-2-3-3
    add("r3233", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    ])
    # 13) 3-3-2-3
    add("r3323", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
    ])
    # 14) 3-3-3-2
    add("r3332", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
    ])
    # 15) 3-3-3-3 minus one
    add("r3333", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
    ])
    # 16) 4-4-3
    add("r443", [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
    ])
    # 17) 3-4-4
    add("r344", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
    ])
    # 18) 5-6
    add("r56", [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
    ])
    # 19) 6-5
    add("r65", [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
    ])
    # 20) 2-2-2-2-3
    add("r22223", [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4), (2, 4),
    ])
    # 21) 2-3-2-2-2
    add("r23222", [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4),
    ])
    # 22) 3-2-2-2-2
    add("r32222", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4),
    ])
    # 23) 2-2-3-2-2
    add("r22322", [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4),
    ])
    # 24) 2-2-2-3-2
    add("r22232", [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (0, 4), (1, 4),
    ])
    # 25) 2-3-3-2-1
    add("r23321", [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4),
    ])
    # 26) 3-3-2-3 (variant)
    add("r3323b", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    ])
    # 27) 3-2-3-3 (variant)
    add("r3233b", [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    ])
    # 28) 2-3-3-3 (variant)
    add("r2333b", [
        (0, 0), (1, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    ])
    # 29) 3-3-3-2 (variant)
    add("r3332b", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3),
    ])
    # 30) 3-3-3-2 (variant 2)
    add("r3332c", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
    ])
    # 31) 2-3-3-3 (variant 2)
    add("r2333c", [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3),
    ])
    # 32) 3-2-3-3 (variant 2)
    add("r3233c", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3),
    ])
    # 33) 3-3-2-3 (variant 2)
    add("r3323c", [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (1, 3), (2, 3), (3, 3),
    ])

    # 34-37) Rotated variants of the most promising compact shapes
    for base in ["block33", "rect34", "stag344", "r3333"]:
        # find base
        for nm, pts in out:
            if nm == base:
                for k in range(1, 4):
                    ang = k * math.pi / 12.0
                    out.append((f"{base}_r{k}", pts @ _rotmat(ang).T))
                break

    return out


def enclosing_side(pts, theta):
    """Return (side_length, center) of the minimal enclosing regular hexagon
    (same orientation as inner hexagons) for a rigid set of centers `pts`
    rotated by `theta`. The center is the midpoint of the bounding box, so
    only the rotation matters."""
    R = _rotmat(theta)
    P = pts @ R.T
    # six outward normals of the enclosing hexagon
    angs = np.array([math.pi / 6.0 + k * math.pi / 3.0 for k in range(6)])
    Nrm = np.stack([np.cos(angs), np.sin(angs)], axis=1)   # (6,2)
    proj = P @ Nrm.T                                        # (n,6)
    mx = proj.max(axis=0)
    mn = proj.min(axis=0)
    D = float(np.max(mx - mn))
    S = (D + 2.0 * AP) / AP
    cx = float((mx + mn).max() / 2.0)   # not used; center is midpoint
    # center = midpoint of bounding box in original coords
    cx = float(P[:, 0].max() + P[:, 0].min()) / 2.0
    cy = float(P[:, 1].max() + P[:, 1].min()) / 2.0
    return S, (cx, cy)


def construct():
    if _time_left() <= 0:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = _build_candidates()

    best_S = float('inf')
    best_pts = None
    best_theta = 0.0
    best_center = (0.0, 0.0)

    for name, pts in candidates:
        if _time_left() <= 0:
            break
        # center the layout at origin for cleaner optimization
        p = pts - pts.mean(axis=0)
        # coarse grid over one symmetry period (0, pi/6)
        n_grid = 48
        best_local = float('inf')
        best_theta_local = 0.0
        for i in range(n_grid + 1):
            th = i * (math.pi / 6.0) / n_grid
            S, c = enclosing_side(p, th)
            if S < best_local:
                best_local = S
                best_theta_local = th
        # bounded scalar refinement
        try:
            res = minimize_scalar(lambda th: enclosing_side(p, th)[0],
                                  bounds=(0.0, math.pi / 6.0), method='bounded',
                                  options={'maxiter': 200, 'xatol': 1e-12})
            if res.fun < best_local:
                best_local = res.fun
                best_theta_local = float(res.x)
        except Exception:
            pass
        # high-precision polish with a few starts around the local best
        try:
            def obj(th):
                return enclosing_side(p, th)[0]
            starts = [best_theta_local,
                      best_theta_local + 0.01, best_theta_local - 0.01,
                      0.0, math.pi / 12.0]
            for st in starts:
                if _time_left() <= 0:
                    break
                st = ((st % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
                r2 = minimize_scalar(obj, bounds=(st - 0.05, st + 0.05),
                                     method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-14})
                if r2.fun < best_local:
                    best_local = r2.fun
                    best_theta_local = float(r2.x)
        except Exception:
            pass

        if best_local < best_S:
            best_S = best_local
            best_pts = p
            best_theta = best_theta_local
            _, best_center = enclosing_side(p, best_theta)

    if best_pts is None or not math.isfinite(best_S):
        # fallback: single hexagon
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Finalize: rotate best_pts by best_theta, recompute exact center & side
    R = _rotmat(best_theta)
    P = best_pts @ R.T
    angs = np.array([math.pi / 6.0 + k * math.pi / 3.0 for k in range(6)])
    Nrm = np.stack([np.cos(angs), np.sin(angs)], axis=1)
    proj = P @ Nrm.T
    mx = proj.max(axis=0)
    mn = proj.min(axis=0)
    D = float(np.max(mx - mn))
    S = (D + 2.0 * AP) / AP
    cx = float(P[:, 0].max() + P[:, 0].min()) / 2.0
    cy = float(P[:, 1].max() + P[:, 1].min()) / 2.0

    # tiny outward scale to guard against floating-point overlap
    SCALE = 1.0 + 1e-12
    P = cx + (P - cx) * SCALE
    proj = (P - np.array([cx, cy])) @ Nrm.T
    D = float(np.max(proj.max(axis=0) - proj.min(axis=0)))
    S = (D + 2.0 * AP) / AP

    angle_deg = math.degrees(best_theta)

    inner = [[float(P[i, 0]), float(P[i, 1]), float(angle_deg)] for i in range(P.shape[0])]
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(angle_deg)
    return inner, center, side, rotation
