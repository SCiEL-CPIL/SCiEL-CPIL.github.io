import numpy as np
import time

N = 11
_START = time.time()
_DEADLINE = 115.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _hex_vertices(cx, cy, s, ang):
    a = ang + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + s * np.cos(a), cy + s * np.sin(a)])

def _sat_pen(A, B):
    min_pen = np.inf
    for P in (A, B):
        for i in range(6):
            p0 = P[i]; p1 = P[(i+1)%6]
            edge = p1 - p0
            n = np.array([-edge[1], edge[0]])
            nl = np.linalg.norm(n)
            if nl < 1e-12: continue
            n /= nl
            a_min, a_max = np.min(P @ n), np.max(P @ n)
            b_min, b_max = np.min(B @ n), np.max(B @ n)
            ov = min(a_max, b_max) - max(a_min, b_min)
            if ov <= 0: return 0.0
            if ov < min_pen: min_pen = ov
    return min_pen

def _inside(verts, outer):
    center = np.mean(outer, axis=0)
    for i in range(6):
        p0 = outer[i]; p1 = outer[(i+1)%6]
        edge = p1 - p0
        ref = np.cross(edge, center - p0)
        for v in verts:
            c = np.cross(edge, v - p0)
            if np.sign(c) != np.sign(ref) and abs(c) > 1e-8:
                return False
    return True

def _no_overlap(positions, angles):
    verts = [_hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if _sat_pen(verts[i], verts[j]) > 1e-9:
                return False
    return True

def _energy(positions, angles, S, oc, oa):
    outer = _hex_vertices(oc[0], oc[1], S, oa)
    e = 0.0
    verts = []
    for i in range(N):
        v = _hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i])
        verts.append(v)
        if not _inside(v, outer):
            e += 20.0
    for i in range(N):
        for j in range(i+1, N):
            p = _sat_pen(verts[i], verts[j])
            if p > 0: e += 8.0 * p
    return e

def _min_side(positions, angles, oc, oa):
    lo, hi = 1.0, 40.0
    for _ in range(45):
        mid = 0.5*(lo+hi)
        ov = _hex_vertices(oc[0], oc[1], mid, oa)
        ok = True
        for i in range(N):
            v = _hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i])
            if not _inside(v, ov):
                ok = False; break
        if ok: hi = mid
        else: lo = mid
    return hi

def construct():
    rng = np.random.default_rng(20240517)
    def init():
        pts = [(0.0,0.0)]
        for k in range(6):
            a = k*np.pi/3.0
            pts.append((2.05*np.cos(a), 2.05*np.sin(a)))
        for k in range(5):
            a = (k+0.5)*np.pi/3.0
            pts.append((4.1*np.cos(a), 4.1*np.sin(a)))
        return np.array(pts[:N], dtype=float)
    positions = init()
    angles = np.zeros(N)
    S = 8.0; oc = np.array([0.0,0.0]); oa = 0.0
    cur = _energy(positions, angles, S, oc, oa)
    best = (positions.copy(), angles.copy(), cur)
    T = 3.0; step = 0.7
    while _remaining() > 3.0:
        if rng.random() < 0.65:
            i = rng.integers(N)
            np2 = positions.copy(); np2[i] += step*rng.standard_normal(2)
            na = angles.copy()
            e2 = _energy(np2, na, S, oc, oa)
        else:
            i = rng.integers(N)
            np2 = positions.copy()
            na = angles.copy(); na[i] += step*rng.standard_normal()
            e2 = _energy(np2, na, S, oc, oa)
        dE = e2 - cur
        if dE < 0 or rng.random() < np.exp(-dE/T):
            positions, angles, cur = np2, na, e2
            if cur < best[2]:
                best = (positions.copy(), angles.copy(), cur)
        T = max(0.05, T*0.996)
        if rng.random() < 0.02: step = max(0.02, step*0.93)

    bp, ba, _ = best
    # final de-overlap
    for _ in range(2000):
        if _remaining() < 1.5: break
        verts = [_hex_vertices(bp[i,0], bp[i,1], 1.0, ba[i]) for i in range(N)]
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                pen = _sat_pen(verts[i], verts[j])
                if pen > 1e-9:
                    d = bp[j] - bp[i]
                    dl = np.linalg.norm(d)
                    if dl < 1e-9: d = np.array([1.0,0.0]); dl = 1.0
                    d = d/dl
                    push = 0.5*(pen + 0.02)
                    bp[i] -= d*push; bp[j] += d*push
                    verts[i] = _hex_vertices(bp[i,0], bp[i,1], 1.0, ba[i])
                    verts[j] = _hex_vertices(bp[j,0], bp[j,1], 1.0, ba[j])
                    moved = True
        if not moved: break

    centroid = np.mean(bp, axis=0)
    best_side = np.inf; best_oc = centroid; best_oa = 0.0
    for oa in [0.0, np.pi/6, np.pi/3, np.pi/12, np.pi/4]:
        for off in [np.array([0,0]), np.array([0.2,0]), np.array([-0.2,0]), np.array([0,0.2]), np.array([0,-0.2])]:
            oc2 = centroid + off
            s = _min_side(bp, ba, oc2, oa)
            if s < best_side:
                best_side = s; best_oc = oc2; best_oa = oa
    inner = [[float(bp[i,0]), float(bp[i,1]), float(np.degrees(ba[i]))] for i in range(N)]
    return inner, (float(best_oc[0]), float(best_oc[1])), float(best_side), float(np.degrees(best_oa))
