# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start = time.time()
    deadline = 580.0
    radius = 1.0

    def _normalize(pts):
        c = pts.mean(axis=0)
        d = pts - c
        mean_r = np.mean(np.linalg.norm(d, axis=1))
        if mean_r <= 0.01:
            return None
        return d / mean_r

    def _vertex_score(dists):
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        D = 0.5 * (dists[k] + dists[k + 1])
        if D < 0.1:
            return 0.0
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
        sel = dists[idx]
        raw = np.mean(np.maximum(sel / D, D / sel))
        if raw <= 0.0:
            return 0.0
        return min(1.0, 1.0 / raw)

    def _vertex_score_smooth(dists, beta=30.0):
        """
        Smooth surrogate: blend over nearby k values to avoid hard argmin.
        """
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        
        # Soft weights over k values using softmax on negative diffs
        neg_diffs = -diffs / beta
        neg_diffs -= neg_diffs.max()
        exp_vals = np.exp(neg_diffs)
        weights = exp_vals / exp_vals.sum()
        
        # Compute score for each k, weighted
        total = 0.0
        for k in range(19):
            if weights[k] < 1e-15:
                continue
            D = 0.5 * (dists[k] + dists[k + 1])
            if D < 0.1:
                continue
            if k == 0:
                idx = [0, 1, 2, 3]
            elif k == 17:
                idx = [15, 16, 17, 18]
            else:
                idx = [k - 1, k, k + 1, k + 2]
            sel = dists[idx]
            raw = np.mean(np.maximum(sel / D, D / sel))
            if raw <= 0.0:
                continue
            # Smooth min(1, 1/raw)
            a = 1.0
            b = 1.0 / raw
            soft_min_val = -beta * np.log((np.exp(-a/beta) + np.exp(-b/beta)) / 2.0)
            total += weights[k] * max(0.0, soft_min_val)
        
        return total

    def _polygon_score(pts, smooth=False, beta=30.0):
        p = _normalize(pts)
        if p is None:
            return 0.0
        m = p.shape[0]
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        best = 1.0
        for i in range(m):
            other = np.delete(dist[i], i)
            other.sort()
            if smooth:
                s = _vertex_score_smooth(other, beta=beta)
            else:
                s = _vertex_score(other)
            if s < best:
                best = s
                if best <= 0.0:
                    break
        return best

    def _build_polygon(offsets):
        """
        Build 20-gon with 5-fold rotational symmetry.
        offsets: 4 values in [0, 2*pi/5) representing angular positions within one sector.
        """
        offs = np.sort(np.asarray(offsets, dtype=float))
        if len(offs) < 4:
            return None
        if np.any(np.diff(offs) < 1e-10):
            return None
        wrap_gap = (offs[0] + 2 * np.pi / 5) - offs[3]
        if wrap_gap < 1e-10:
            return None

        angles = np.array([])
        for j in range(5):
            sector_start = j * 2 * np.pi / 5
            angles = np.append(angles, offs + sector_start)
        angles = np.sort(angles)

        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pts = np.stack([x, y], axis=1)
        return pts

    def _obj(offsets, smooth=False, beta=30.0):
        pts = _build_polygon(offsets)
        if pts is None:
            return 1.0
        return -_polygon_score(pts, smooth=smooth, beta=beta)

    # Regular 20-gon as baseline
    base_offsets = np.array([0.0, np.pi/10, 2*np.pi/10, 3*np.pi/10])
    best_pts = _build_polygon(base_offsets)
    if best_pts is None:
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)

    best_score = _polygon_score(best_pts)
    best_offsets = base_offsets.copy()

    # Phase 1: Differential Evolution with smooth objective for better exploration
    bounds = [(0.0, 2.0 * np.pi / 5) for _ in range(4)]
    try:
        res = differential_evolution(
            lambda x: _obj(x, smooth=True, beta=50.0),
            bounds,
            seed=20240611,
            maxiter=2000,
            popsize=35,
            tol=1e-10,
            mutation=(0.4, 1.2),
            recombination=0.85,
            polish=False,
            init="sobol",
            callback=lambda xk, convergence: (
                True if (time.time() - start) < deadline - 120.0 else False
            ),
        )
        if res is not None and res.x is not None:
            pts = _build_polygon(res.x)
            if pts is not None:
                sc = _polygon_score(pts)
                if sc > best_score:
                    best_pts, best_score = pts, sc
                    best_offsets = np.sort(res.x).copy()
    except Exception:
        pass

    # Phase 2: Adaptive smooth Nelder-Mead with decreasing beta
    if (time.time() - start) < deadline - 80.0:
        betas = [40.0, 25.0, 15.0]
        for beta in betas:
            if (time.time() - start) >= deadline - 40.0:
                break
            try:
                x0 = best_offsets.copy()
                r2 = minimize(
                    lambda x: _obj(x, smooth=True, beta=beta),
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 25000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
                        "adaptive": True
                    }
                )
                if r2 is not None and r2.x is not None:
                    pts = _build_polygon(r2.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
                            best_offsets = np.sort(r2.x).copy()
            except Exception:
                pass

    # Phase 3: Multi-start with exact objective for escaping local minima
    if (time.time() - start) < deadline - 30.0:
        rng = np.random.default_rng(42)
        n_starts = 12
        for si in range(n_starts):
            if (time.time() - start) >= deadline - 15.0:
                break
            if si == 0:
                x0 = best_offsets.copy()
            else:
                # Increasingly diverse perturbations
                noise_scale = 0.015 * (1.0 + si * 0.4)
                x0 = best_offsets + rng.uniform(-noise_scale, noise_scale, size=4)
                x0 = np.clip(x0, 0.0, 2.0 * np.pi / 5)
                x0 = np.sort(x0)
                if np.any(np.diff(x0) < 1e-8):
                    continue

            try:
                # Use smooth objective first, then exact
                r3 = minimize(
                    lambda x: _obj(x, smooth=True, beta=20.0),
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 12000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
                        "adaptive": True
                    }
                )
                if r3 is not None and r3.x is not None:
                    pts = _build_polygon(r3.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
                            best_offsets = np.sort(r3.x).copy()
            except Exception:
                pass

    # Phase 4: Final ultra-precise polish with exact objective
    if (time.time() - start) < deadline - 5.0:
        try:
            x0 = best_offsets.copy()
            r4 = minimize(
                lambda x: _obj(x, smooth=False),
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": 50000,
                    "xatol": 1e-16,
                    "fatol": 1e-18,
                    "adaptive": True
                }
            )
            if r4 is not None and r4.x is not None:
                pts = _build_polygon(r4.x)
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts, best_score = pts, sc
        except Exception:
            pass

    # Final safety check
    if _polygon_score(best_pts) <= 0.0:
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)

    return best_pts

# EVOLVE-BLOCK-END
