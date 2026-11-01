import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def _normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        return pts / norms

    def _hull_data(p):
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        F = len(facets)
        if F == 0:
            return None, None, None
        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]
        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n, axis=1)
        valid = nn > 1e-12
        cc = np.zeros((F, 3))
        radii = np.zeros(F)
        if np.any(valid):
            ccv = n[valid] / nn[valid, None]
            dots = ccv @ a[valid].T
            neg = dots < 0
            ccv[neg] *= -1
            cc[valid] = ccv
            cosr = np.clip(ccv @ a[valid].T, -1.0, 1.0)
            radii[valid] = np.arccos(cosr)
        return facets, cc, radii

    def _covering_radius(p):
        f, c, r = _hull_data(p)
        if f is None:
            return np.inf
        return float(np.max(r))

    def _soft_max_energy_and_grad(p, beta):
        f, cc, r = _hull_data(p)
        if f is None:
            return None, None, None, None
        r2 = r * r
        mx = r2.max()
        e = np.exp(beta * (r2 - mx))
        se = e.sum()
        w = r * e / se
        forces = np.zeros_like(p)
        for k in range(len(f)):
            wk = w[k]
            if wk < 1e-14:
                continue
            cck = cc[k]
            for idx in f[k]:
                forces[idx] += wk * (cck - p[idx])
        return forces, f, cc, r

    def _lloyd_steps(pts, n_steps, beta_start, beta_end, step_start, step_end,
                     rep_start, rep_end, time_budget):
        pts = pts.copy()
        best = pts.copy()
        best_r = _covering_radius(pts)
        for s in range(n_steps):
            if time.time() > time_budget:
                break
            t = s / max(n_steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * t
            eta = step_start * (step_end / step_start) ** t
            rep = rep_start * (rep_end / rep_start) ** t
            forces, f, cc, r = _soft_max_energy_and_grad(pts, beta)
            if forces is None:
                break
            # repulsion
            d = pts[:, None, :] - pts[None, :, :]
            dsq = np.einsum('ijk,ijk->ij', d, d)
            np.fill_diagonal(dsq, 1.0)
            inv = 1.0 / dsq
            np.fill_diagonal(inv, 0.0)
            repf = np.einsum('ij,ijk->ik', inv, d)
            forces += rep * repf
            pts = _normalize(pts + eta * forces)
            cur_r = _covering_radius(pts)
            if cur_r < best_r:
                best_r = cur_r
                best = pts.copy()
        return best, best_r

    def _lbfgs_polish(pts, time_budget, obj='sum_sq', maxiter=40):
        if time.time() > time_budget:
            return pts, _covering_radius(pts)
        br = _covering_radius(pts)
        if obj == 'sum_sq':
            def objf(x):
                X = _normalize(x.reshape(N, 3))
                f, c, r = _hull_data(X)
                if f is None:
                    return 1e10
                return float(np.sum(r * r))
        else:
            def objf(x):
                X = _normalize(x.reshape(N, 3))
                f, c, r = _hull_data(X)
                if f is None:
                    return 1e10
                r2 = r * r
                mx = r2.max()
                e = np.exp(150.0 * (r2 - mx))
                return float(mx / 150.0 + np.log(e.sum()) / 150.0)
        try:
            res = minimize(objf, pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-14,
                                    'gtol': 1e-12, 'maxls': 200})
            if res.fun < objf(pts.flatten()) and res.x is not None:
                np2 = _normalize(res.x.reshape(N, 3))
                cr = _covering_radius(np2)
                if cr < br:
                    br = cr
                    pts = np2
        except Exception:
            pass
        return pts, br

    def _worst_facet_push(pts, time_budget, n_push=20):
        """Nudge points toward the worst facet circumcenter, then re-polish."""
        for _ in range(n_push):
            if time.time() > time_budget:
                break
            f, cc, r = _hull_data(pts)
            if f is None:
                break
            wi = int(np.argmax(r))
            cck = cc[wi]
            for idx in f[wi]:
                pts[idx] += 0.01 * (cck - pts[idx])
            pts = _normalize(pts)
        pts, br = _lbfgs_polish(pts, time_budget, obj='sum_sq', maxiter=25)
        return pts, br

    def _optimize_from_start(init, time_budget):
        pts = init.copy()
        best = pts.copy()
        best_r = _covering_radius(pts)

        # Phase 1: coarse Lloyd with adaptive beta and step
        p1_end = time_budget - 18.0
        p1, r1 = _lloyd_steps(pts, 800, 15.0, 60.0, 0.02, 0.005, 0.12, 0.03, p1_end)
        if r1 < best_r:
            best_r = r1
            best = p1

        # Phase 2: L-BFGS polish
        p2_end = time_budget - 8.0
        p2, r2 = _lbfgs_polish(best, p2_end, obj='sum_sq', maxiter=50)
        if r2 < best_r:
            best_r = r2
            best = p2

        # Phase 3: finer Lloyd
        p3_end = time_budget - 5.0
        p3, r3 = _lloyd_steps(best, 400, 60.0, 120.0, 0.006, 0.002, 0.04, 0.01, p3_end)
        if r3 < best_r:
            best_r = r3
            best = p3

        # Phase 4: worst-facet push
        p4_end = time_budget - 3.0
        p4, r4 = _worst_facet_push(best, p4_end, n_push=15)
        if r4 < best_r:
            best_r = r4
            best = p4

        # Phase 5: final L-BFGS with soft-max
        p5_end = time_budget - 1.5
        p5, r5 = _lbfgs_polish(best, p5_end, obj='softmax', maxiter=30)
        if r5 < best_r:
            best_r = r5
            best = p5

        # Phase 6: tiny Lloyd + sum_sq polish
        p6_end = time_budget - 0.8
        p6, r6 = _lloyd_steps(best, 150, 120.0, 200.0, 0.003, 0.001, 0.01, 0.003, p6_end)
        if r6 < best_r:
            best_r = r6
            best = p6
        p7, r7 = _lbfgs_polish(best, time_budget - 0.3, obj='sum_sq', maxiter=20)
        if r7 < best_r:
            best_r = r7
            best = p7

        return best, best_r

    # --- Initial configurations ---
    def _fib(offset=0.5, phase=0.0):
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + phase)
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    # Icosahedron-based
    def _ico_based():
        g = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0, 1, g], [0, -1, g], [0, 1, -g], [0, -1, -g],
            [1, g, 0], [-1, g, 0], [1, -g, 0], [-1, -g, 0],
            [g, 0, 1], [-g, 0, 1], [g, 0, -1], [-g, 0, -1],
            [1, 0, -g], [-1, 0, -g], [1, 0, g], [-1, 0, g]
        ], dtype=float)
        ico = _normalize(ico)
        # Add rotated copies
        from scipy.spatial.transform import Rotation
        all_pts = [ico]
        for ang in [np.pi/5, 2*np.pi/5, 3*np.pi/5, np.pi/3, 2*np.pi/3]:
            rot = Rotation.from_rotvec([0, 0, ang])
            all_pts.append(rot.apply(ico))
            rot2 = Rotation.from_rotvec([0, 0, ang + np.pi/4])
            all_pts.append(rot2.apply(ico))
        # Random rotations too
        rng = np.random.default_rng(42)
        for _ in range(10):
            rot = Rotation.random(random_state=rng)
            all_pts.append(rot.apply(ico))
        all_pts = np.vstack(all_pts)
        # Greedy farthest-point selection
        selected = [all_pts[0]]
        for _ in range(N - 1):
            dists = np.array([min(np.linalg.norm(p - s) for s in selected) for p in all_pts])
            selected.append(all_pts[int(np.argmax(dists))])
        return _normalize(np.array(selected))

    # Random with minimum distance
    def _random_min_dist(seed=0):
        rng = np.random.default_rng(seed)
        pts = []
        min_d = 0.3
        attempts = 0
        while len(pts) < N and attempts < 20000:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            if all(np.linalg.norm(v - p) > min_d for p in pts):
                pts.append(v)
                min_d *= 0.998
            attempts += 1
        # Pad if needed
        fib = _fib(0.5, 0.0)
        for p in fib[len(pts):]:
            pts.append(p)
            if len(pts) >= N:
                break
        return _normalize(np.array(pts[:N]))

    inits = [
        _fib(0.5, 0.0),
        _fib(0.5, 0.37),
        _fib(0.5, 0.71),
        _fib(0.3, 0.15),
        _ico_based(),
        _random_min_dist(0),
        _random_min_dist(123),
        _random_min_dist(999),
    ]

    best_pts = inits[0].copy()
    best_radius = float('inf')

    # Allocate time: more to earlier configs
    n_cfg = len(inits)
    for idx, init in enumerate(inits):
        if time.time() > deadline - 3.0:
            break
        # First 2 configs get more time
        if idx < 2:
            tb = deadline - 12.0 - idx * 4.0
        elif idx < 4:
            tb = deadline - 8.0 - (idx - 2) * 3.0
        else:
            tb = deadline - 5.0 - (idx - 4) * 1.5
        if time.time() > tb - 1.0:
            continue
        try:
            pts_res, rad_res = _optimize_from_start(init, tb)
            if rad_res < best_radius:
                best_radius = rad_res
                best_pts = pts_res.copy()
        except Exception:
            pass

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = _fib(0.5, 0.0)

    # One last global polish if time remains
    if time.time() < deadline - 2.0:
        bp, br = _lbfgs_polish(best_pts, deadline - 1.0, obj='softmax', maxiter=25)
        if br < best_radius:
            best_radius = br
            best_pts = bp

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = _fib(0.5, 0.0)

    return best_pts
