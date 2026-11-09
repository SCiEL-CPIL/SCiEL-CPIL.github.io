# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 600.0
_t0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _t0)


def _vertex_score(v, P):
    """Per-vertex score per the task contract (0.0 if invalid)."""
    d = np.sort(np.linalg.norm(P - v, axis=1))
    if d[0] < 0.05:
        return 0.0
    gaps = d[1:] - d[:-1]
    k = int(np.argmin(gaps))
    D = 0.5 * (d[k] + d[k + 1])
    if D < 0.1:
        return 0.0
    if k == 0:
        four = d[0:4]
    elif k == 17:
        four = d[15:19]
    else:
        four = d[k - 1:k + 3]
    ratio = np.maximum(four / D, D / four)
    raw = np.mean(ratio)
    if not np.isfinite(raw) or raw <= 0.0:
        return 0.0
    return float(min(1.0, 1.0 / raw))


def _metric(P):
    """Minimum vertex score over all vertices."""
    s = np.empty(P.shape[0])
    for i in range(P.shape[0]):
        s[i] = _vertex_score(P[i], P)
    return float(np.min(s))


def _is_valid_convex(P):
    n = P.shape[0]
    if n < 3:
        return False
    if not np.all(np.isfinite(P)):
        return False
    # no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(P[i] - P[j]) < 1e-9:
                return False
    # cross products all same nonzero sign
    signs = []
    for i in range(n):
        a = P[i]
        b = P[(i + 1) % n]
        c = P[(i + 2) % n]
        cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cr) < 1e-12:
            return False
        signs.append(1 if cr > 0 else -1)
    if len(set(signs)) != 1:
        return False
    return True


def _normalize(P):
    c = P.mean(axis=0)
    Q = P - c
    dists = np.linalg.norm(Q, axis=1)
    m = dists.mean()
    if m <= 0.01:
        return Q  # will be scored 0
    return Q / m


def _build_poly(radii):
    n = radii.shape[0]
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radii * np.cos(ang)
    y = radii * np.sin(ang)
    P = np.stack([x, y], axis=1)
    return _normalize(P)


def _symmetric_radii(A, phi):
    n = 20
    i = np.arange(n)
    return 1.0 + A * np.cos(5.0 * i * (2.0 * np.pi / 20.0) + phi)


def _nm_refine(P0, max_iter=600, tol=1e-8):
    """Per-vertex radial Nelder-Mead maximizing the minimum vertex score."""
    n = P0.shape[0]
    # working coordinates: radii at fixed angles (before normalization)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # recover radii from P0 (un-normalized magnitudes are proportional)
    r = np.linalg.norm(P0, axis=1)
    best_r = r.copy()
    best_score = _metric(_build_poly(r))
    for _ in range(max_iter):
        if _remaining() < 30.0:
            break
        improved = False
        for vi in range(n):
            if _remaining() < 30.0:
                break
            base = best_r.copy()
            s0 = best_score

            def obj(rr):
                return -_metric(_build_poly(rr))

            x0 = base.copy()
            # simplex over radius of vertex vi
            step = 0.05
            simp = [x0.copy()]
            for _ in range(1):
                xi = x0.copy()
                xi[vi] += step
                simp.append(xi)
            # simple 2D Nelder-Mead on (radius_vi) is trivial; do scalar search
            # Use a small Nelder-Mead on 2 dims: (r_vi, r_{vi+1}) to allow local adjust
            a = vi
            b = (vi + 1) % n
            x02 = np.array([base[a], base[b]])

            def obj2(x):
                rr = base.copy()
                rr[a] = x[0]
                rr[b] = x[1]
                return -_metric(_build_poly(rr))

            # Nelder-Mead
            eps = 0.02
            S = [x02.copy(),
                 x02 + np.array([eps, 0.0]),
                 x02 + np.array([0.0, eps])]
            f = [obj2(s) for s in S]
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            for it in range(200):
                idx = np.argsort(f)
                S = [S[j] for j in idx]
                f = [f[j] for j in idx]
                if abs(f[0] - f[-1]) < tol:
                    break
                centroid = np.mean(S[:-1], axis=0)
                xr = centroid + alpha * (centroid - S[-1])
                fr = obj2(xr)
                if fr < f[0]:
                    xe = centroid + gamma * (centroid - S[-1])
                    fe = obj2(xe)
                    S[-1] = xe if fe < fr else xr
                    f[-1] = min(fe, fr)
                elif fr < f[-2]:
                    S[-1] = xr
                    f[-1] = fr
                else:
                    xc = centroid + rho * (S[-1] - centroid)
                    fc = obj2(xc)
                    if fc < f[-1]:
                        S[-1] = xc
                        f[-1] = fc
                    else:
                        for j in range(1, len(S)):
                            S[j] = S[0] + sigma * (S[j] - S[0])
                            f[j] = obj2(S[j])
            new_r = base.copy()
            new_r[a] = S[0][0]
            new_r[b] = S[0][1]
            ns = -f[0]
            if ns > s0 + 1e-12:
                best_r = new_r
                best_score = ns
                improved = True
        if not improved:
            break
    return _build_poly(best_r)


def equidistant_points_polygon20() -> np.ndarray:
    # 1) coarse symmetric search over (A, phi)
    best_A, best_phi = 0.0, 0.0
    best_score = -1.0
    for A in np.linspace(0.0, 0.35, 36):
        if _remaining() < 60.0:
            break
        for phi in np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False):
            if _remaining() < 60.0:
                break
            r = _symmetric_radii(A, phi)
            P = _build_poly(r)
            if not _is_valid_convex(P):
                continue
            sc = _metric(P)
            if sc > best_score:
                best_score = sc
                best_A, best_phi = A, phi

    # 2) local Nelder-Mead refinement over the symmetric basin
    r0 = _symmetric_radii(best_A, best_phi)
    P0 = _build_poly(r0)
    if _is_valid_convex(P0):
        P_ref = _nm_refine(P0, max_iter=8)
        if _is_valid_convex(P_ref) and _metric(P_ref) >= best_score - 1e-9:
            P0 = P_ref
            best_score = _metric(P0)

    # 3) fallback safety: regular polygon is always valid
    if not _is_valid_convex(P0):
        n = 20
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        P0 = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        P0 = _normalize(P0)

    return P0


# EVOLVE-BLOCK-END
