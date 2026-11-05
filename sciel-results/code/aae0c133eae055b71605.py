import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 8
    # Ordered pairs: all (i,j) with i != j, so n*(n-1) = 56 distances
    iu, ju = np.where(~np.eye(n, dtype=bool))  # 56 ordered pairs
    num_ordered = len(iu)  # 56
    nlogn = n * np.log(n)

    def exact_log_score(P):
        """Exact log(Delta/n^n) for 8 points using ORDERED pairs (diameter-scaled). Returns -inf if degenerate."""
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dsq = np.sum(diff * diff, axis=2)
        d = np.sqrt(dsq[iu, ju])  # 56 ordered pair distances
        if np.any(d < 1e-12):
            return -np.inf
        diam = np.max(d)
        if diam < 1e-12:
            return -np.inf
        s = d * (2.0 / diam)
        s = np.maximum(s, 1e-15)
        return np.sum(np.log(s)) - nlogn

    # ---- Two-concentric-squares parameterization (4-dim structural manifold) ----
    def two_sq_to_pts(r1, r2, phi, rot):
        """r1, r2: radii of the two squares; phi: relative rotation (radians); rot: global rotation."""
        a1 = np.pi / 4.0 + rot + np.array([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0])
        a2 = np.pi / 4.0 + rot + phi + np.array([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0])
        p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        return np.vstack([p1, p2])

    def neg_log_score_4d(params):
        """Negative of exact log-score for two-squares configuration. Returns large if degenerate."""
        r1, r2, phi, rot = params
        if r1 < 1e-10 or r2 < 1e-10:
            return 1e10
        P = two_sq_to_pts(r1, r2, phi, rot)
        d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-10:
            return 1e10
        s = exact_log_score(P)
        if np.isneginf(s):
            return 1e10
        return -s

    def smooth_obj(x, beta):
        """Smooth surrogate: -log(score) with soft-max diameter. Minimize this. Uses ordered pairs."""
        P = x.reshape(n, 2)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dsq = np.sum(diff * diff, axis=2)
        d = np.sqrt(dsq[iu, ju])  # 56 ordered pair distances
        if np.any(d < 1e-12):
            return 1e10
        # soft-max for the diameter
        scaled = d / beta
        m = np.max(scaled)
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        diam = beta * lse
        if diam < 1e-12:
            return 1e10
        s = d * (2.0 / diam)
        s = np.maximum(s, 1e-15)
        return nlogn - np.sum(np.log(s))

    # ---------- Baseline: regular octagon ----------
    ang = 2.0 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(ang), np.sin(ang)])
    best_score = exact_log_score(reg_oct)   # Should be 0 for regular octagon with ordered pairs
    best_points = reg_oct.copy()

    def try_update(P):
        nonlocal best_score, best_points
        if P is None or np.any(np.isnan(P)) or np.any(np.isinf(P)):
            return
        d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-8:
            return
        s = exact_log_score(P)
        if s > best_score:
            best_score = s
            best_points = P.copy()

    # ---------- Build diverse initializations ----------
    starts = [reg_oct.flatten().copy()]

    # Two-squares with varied relative rotation (strong structural prior)
    for rot in np.linspace(0.0, 0.49, 25):
        a1 = 2.0 * np.pi * np.arange(4) / 4
        a2 = 2.0 * np.pi * (np.arange(4) / 4 + rot)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())

    # ---- Structural search on the 4-dim two-squares manifold ----
    r1_grid = np.linspace(0.3, 1.8, 25)
    r2_grid = np.linspace(0.3, 1.8, 25)
    phi_grid = np.linspace(0.0, np.pi/2.0, 35)

    best_4d_score = -np.inf
    best_4d_params = None
    grid_starts = []

    for r1 in r1_grid:
        for r2 in r2_grid:
            for phi in phi_grid:
                P = two_sq_to_pts(r1, r2, phi, 0.0)
                d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
                np.fill_diagonal(d, np.inf)
                if np.min(d) < 1e-8:
                    continue
                s = exact_log_score(P)
                if np.isneginf(s):
                    continue
                if s > best_4d_score:
                    best_4d_score = s
                    best_4d_params = (r1, r2, phi, 0.0)
                if s > best_4d_score - 0.05:
                    grid_starts.append((r1, r2, phi, 0.0))

    # Add the best grid point and neighbors as starts
    if best_4d_params is not None:
        P_best = two_sq_to_pts(*best_4d_params)
        starts.append(P_best.flatten())
        for dr1 in [-0.1, 0.1]:
            for dr2 in [-0.1, 0.1]:
                for dphi in [-0.05, 0.05]:
                    rp = (best_4d_params[0] + dr1, best_4d_params[1] + dr2,
                          best_4d_params[2] + dphi, 0.0)
                    if rp[0] > 0.1 and rp[1] > 0.1:
                        P_near = two_sq_to_pts(*rp)
                        d_check = np.sqrt(np.sum((P_near[:, None, :] - P_near[None, :, :]) ** 2, axis=2))
                        np.fill_diagonal(d_check, np.inf)
                        if np.min(d_check) > 1e-8:
                            starts.append(P_near.flatten())

    # Antipodal-pair initializations
    rng = np.random.RandomState(7)
    for _ in range(25):
        base = np.sort(rng.uniform(0.0, np.pi, 4))
        allang = np.concatenate([base, base + np.pi])
        pts = np.column_stack([np.cos(allang), np.sin(allang)])
        starts.append(pts.flatten())

    # Perturbed circle
    for _ in range(25):
        ba = 2.0 * np.pi * np.arange(n) / n + rng.uniform(-0.25, 0.25, n)
        rd = 1.0 + rng.uniform(-0.25, 0.25, n)
        pts = np.column_stack([np.cos(ba) * rd, np.sin(ba) * rd])
        starts.append(pts.flatten())

    # Random points
    for _ in range(20):
        starts.append(rng.uniform(-1.5, 1.5, (n, 2)).flatten())

    # ---------- Phase 0: local refinement on the 4-dim structural manifold ----------
    if best_4d_params is not None and time.time() < deadline - 50:
        refine_list = [best_4d_params]
        seen = set()
        for gp in grid_starts[:20]:
            key = (round(gp[0], 2), round(gp[1], 2), round(gp[2], 2))
            if key not in seen:
                seen.add(key)
                refine_list.append(gp)

        for gp in refine_list[:10]:
            if time.time() > deadline - 45:
                break
            try:
                res4d = minimize(neg_log_score_4d, gp,
                                  method='L-BFGS-B',
                                  bounds=[(0.05, 3.0), (0.05, 3.0), (0.0, np.pi/2.0), (0.0, np.pi/2.0)],
                                  options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                if res4d.fun < 1e9:
                    P_ref = two_sq_to_pts(*res4d.x)
                    try_update(P_ref)
            except Exception:
                pass

    # ---------- Phase 1: smooth L-BFGS-B from every start ----------
    betas_coarse = [8.0, 40.0]
    for s in starts:
        if time.time() > deadline - 45:
            break
        for beta in betas_coarse:
            if time.time() > deadline - 45:
                break
            try:
                res = minimize(lambda x, b=beta: smooth_obj(x, b), s,
                               method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e9:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # ---------- Phase 2: refine current best with increasing beta ----------
    if time.time() < deadline - 25:
        for beta in [100.0, 500.0, 2000.0, 10000.0, 1e6]:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(lambda x: smooth_obj(x, beta), best_points.flatten(),
                               method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < 1e9:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # ---------- Phase 3: global differential evolution ----------
    if time.time() < deadline - 35:
        bounds = [(-4.0, 4.0)] * (2 * n)
        try:
            de_res = differential_evolution(lambda x: smooth_obj(x, 60.0), bounds,
                                            seed=42, maxiter=120, popsize=20,
                                            tol=1e-8, polish=True, updateinterval=10)
            if de_res.fun < 1e9:
                try_update(de_res.x.reshape(n, 2))
        except Exception:
            pass

    # ---------- Phase 3b: dual_annealing on the 4-dim structural manifold ----------
    if time.time() < deadline - 25:
        bounds_4d = [(0.05, 3.0), (0.05, 3.0), (0.0, np.pi/2.0), (0.0, np.pi/2.0)]
        try:
            da_res = dual_annealing(neg_log_score_4d, bounds_4d,
                                     seed=42, maxiter=100, initial_temp=2.0,
                                     restart_temp_ratio=2.0)
            if da_res.fun < 1e9:
                P_da = two_sq_to_pts(*da_res.x)
                try_update(P_da)
        except Exception:
            pass

    # ---------- Phase 4: local DE around the best solution ----------
    if time.time() < deadline - 20:
        try:
            bnd = [(best_points[i, j] - 0.6, best_points[i, j] + 0.6)
                   for i in range(n) for j in range(2)]
            de2 = differential_evolution(lambda x: smooth_obj(x, 300.0), bnd,
                                         seed=11, maxiter=100, popsize=15,
                                         tol=1e-10, polish=True)
            if de2.fun < 1e9:
                try_update(de2.x.reshape(n, 2))
        except Exception:
            pass

    # ---------- Phase 5: final exact polish ----------
    if time.time() < deadline - 5:
        for beta in [1e5, 1e7]:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(lambda x: smooth_obj(x, beta), best_points.flatten(),
                               method='L-BFGS-B',
                               options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.fun < 1e9:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # ---------- Phase 5b: final polish on 4-dim manifold ----------
    if time.time() < deadline - 2:
        if best_4d_params is not None:
            try:
                res_final_4d = minimize(neg_log_score_4d, best_4d_params,
                                         method='L-BFGS-B',
                                         bounds=[(0.05, 3.0), (0.05, 3.0), (0.0, np.pi/2.0), (0.0, np.pi/2.0)],
                                         options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
                if res_final_4d.fun < 1e9:
                    P_final = two_sq_to_pts(*res_final_4d.x)
                    try_update(P_final)
            except Exception:
                pass

    # ---------- Final validation ----------
    d = np.sqrt(np.sum((best_points[:, None, :] - best_points[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-10:
        best_points = reg_oct
    if np.max(np.abs(best_points)) > 1e6:
        best_points = reg_oct

    return [[float(best_points[i, 0]), float(best_points[i, 1])] for i in range(n)]
