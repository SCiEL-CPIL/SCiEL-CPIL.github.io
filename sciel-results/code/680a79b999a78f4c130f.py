# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = 620.0  # Leave 10s margin

    def _normalize(pts):
        """Normalize points: subtract centroid, divide by mean distance to centroid."""
        c = pts.mean(axis=0)
        d = pts - c
        mean_r = np.mean(np.linalg.norm(d, axis=1))
        if mean_r <= 0.01:
            return None
        return d / mean_r

    def _vertex_score(dists):
        """Calculate score for a single vertex given sorted distances to other vertices."""
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
        # Avoid division by zero or tiny numbers
        ratio1 = sel / D
        ratio2 = D / sel
        raw = np.mean(np.maximum(ratio1, ratio2))
        if raw <= 0.0:
            return 0.0
        return min(1.0, 1.0 / raw)

    def _polygon_score(pts):
        """Calculate the minimum vertex score for the polygon."""
        p = _normalize(pts)
        if p is None:
            return 0.0
        
        # Calculate pairwise distances
        # p is (n, 2)
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        
        best = 1.0
        for i in range(n):
            other_dists = np.delete(dist[i], i)
            other_dists.sort()
            s = _vertex_score(other_dists)
            if s < best:
                best = s
                if best <= 0.0:
                    return 0.0
        return best

    def _build_polygon_from_gaps(gaps):
        """
        Build polygon from 19 gaps (angular differences between consecutive vertices).
        The 20th gap is implied to close the circle.
        Returns points in cyclic order on the unit circle (before normalization).
        """
        if len(gaps) != n - 1:
            return None
        if np.any(gaps <= 1e-6):
            return None
        
        total_gap = np.sum(gaps)
        if total_gap >= 2 * np.pi:
            return None
        
        g_last = 2 * np.pi - total_gap
        if g_last <= 1e-6:
            return None
            
        all_gaps = np.append(gaps, g_last)
        angles = np.cumsum(all_gaps)
        angles = angles - angles[0] # Start at 0 for symmetry invariance? No, just relative angles matter.
        
        x = np.cos(angles)
        y = np.sin(angles)
        pts = np.stack([x, y], axis=1)
        return pts

    def _obj_gaps(gaps, penalty=True):
        """Objective function for gap optimization. Returns -score."""
        pts = _build_polygon_from_gaps(gaps)
        if pts is None:
            return 1.0 if penalty else 10.0
        sc = _polygon_score(pts)
        return -sc

    # --- Strategy 1: Global Search on Gaps (Unit Circle) ---
    # Bounds for gaps: very small to large
    # Since sum must be < 2pi, individual gaps can be up to ~2pi.
    # But for convexity, we just need > 0.
    # To avoid boundary issues in DE, we use a transformation or just strict bounds.
    # Let's use bounds [1e-3, 2*np.pi - (n-1)*1e-3] which is roughly [0.001, 6.28]
    
    lower_bound = 1e-3
    upper_bound = 2.0 * np.pi - (n - 1) * 1e-3
    bounds = [(lower_bound, upper_bound)] * (n - 1)
    
    best_pts = None
    best_score = -1.0
    
    # Initialize with regular polygon gaps
    reg_gap = 2.0 * np.pi / n
    reg_gaps = np.full(n - 1, reg_gap)
    init_pts = _build_polygon_from_gaps(reg_gaps)
    if init_pts is not None:
        best_pts = init_pts
        best_score = _polygon_score(best_pts)
    
    # Run Differential Evolution
    if time.time() - start_time < deadline - 100.0:
        try:
            # Use a wrapper that handles the sum constraint more gracefully if DE pushes it
            # Actually, the bounds [lower, upper] don't guarantee sum < 2pi.
            # We rely on _build_polygon_from_gaps returning None if sum >= 2pi.
            # This might cause DE to get stuck if the feasible region is a small corner.
            # Better approach: Optimize log-gaps? Or just rely on the penalty.
            
            def _obj_de(x):
                pts = _build_polygon_from_gaps(x)
                if pts is None:
                    return 1.0
                sc = _polygon_score(pts)
                return -sc

            res = differential_evolution(
                _obj_de,
                bounds,
                seed=42,
                maxiter=1000,
                popsize=30,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                callback=lambda xk, convergence: (
                    True if (time.time() - start_time) < deadline - 50.0 else False
                )
            )
            if res is not None and res.x is not None:
                pts = _build_polygon_from_gaps(res.x)
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts
        except Exception:
            pass

    # --- Strategy 2: Local Polish on Gaps ---
    # Use Nelder-Mead on the best found gaps
    if best_pts is not None and time.time() - start_time < deadline - 20.0:
        # Extract gaps from best_pts
        # best_pts are on unit circle? No, they are normalized.
        # We need to recover angles from best_pts to get gaps.
        c = best_pts.mean(axis=0)
        d_pts = best_pts - c
        mean_r = np.mean(np.linalg.norm(d_pts, axis=1))
        if mean_r > 1e-6:
            unit_pts = d_pts / mean_r
            angles = np.arctan2(unit_pts[:, 1], unit_pts[:, 0])
            # Sort angles to ensure cyclic order
            order = np.argsort(angles)
            angles_sorted = angles[order]
            
            # Calculate gaps
            gaps_calc = np.diff(angles_sorted)
            # Handle wrap-around
            gap_last = 2 * np.pi - (angles_sorted[-1] - angles_sorted[0])
            all_gaps = np.append(gaps_calc, gap_last)
            
            # We optimize 19 gaps, the 20th is determined
            # We need to map the sorted points back to the optimization variable order?
            # The optimization variable `gaps` corresponds to consecutive vertices in the order they were built.
            # _build_polygon_from_gaps builds them in the order of `gaps`.
            # So if we have `best_pts` from a specific `gaps` vector, we can just use that `gaps` vector.
            # But if `best_pts` came from the bipartite search or was modified, we need to re-derive.
            # For now, let's assume best_pts corresponds to some angular sequence.
            # We will use the derived `all_gaps[:19]` as the starting point for local polish.
            
            x0 = all_gaps[:19].copy()
            
            # Ensure x0 is feasible
            if np.any(x0 < 1e-6) or np.sum(x0) >= 2*np.pi - 1e-6:
                 x0 = reg_gaps.copy()
            
            try:
                res_local = minimize(
                    _obj_gaps,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if res_local.x is not None:
                    pts = _build_polygon_from_gaps(res_local.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = pts
            except Exception:
                pass

    # --- Strategy 3: Bipartite Radial Search (Strong Baseline) ---
    # This often yields high scores. We run it if we have time.
    def _build_bipartite(R1, R2, phi):
        R1 = max(0.3, min(2.0, R1))
        R2 = max(0.3, min(2.0, R2))
        phi = np.mod(phi, 2.0 * np.pi)
        
        angles1 = np.linspace(0, 2*np.pi, 10, endpoint=False)
        angles2 = angles1 + phi + np.pi/10
        
        x1 = R1 * np.cos(angles1)
        y1 = R1 * np.sin(angles1)
        x2 = R2 * np.cos(angles2)
        y2 = R2 * np.sin(angles2)
        
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        all_angles = np.concatenate([angles1, angles2])
        
        order = np.argsort(all_angles)
        x = all_x[order]
        y = all_y[order]
        
        pts = np.stack([x, y], axis=1)
        return pts

    def _obj_bipartite(params):
        R1, R2, phi = params
        pts = _build_bipartite(R1, R2, phi)
        if pts is None:
            return 1.0
        sc = _polygon_score(pts)
        return -sc

    if time.time() - start_time < deadline - 30.0:
        # Try a few initial points for bipartite
        bip_starts = [
            (1.0, 1.0, 0.0),
            (0.9, 1.1, np.pi/20),
            (1.1, 0.9, np.pi/15),
            (1.0, 1.2, 0.0),
            (0.8, 1.2, np.pi/10),
        ]
        
        best_bip_score = -1.0
        best_bip_pts = None
        
        for R1_0, R2_0, phi_0 in bip_starts:
            if time.time() - start_time > deadline - 10.0:
                break
            try:
                res_bip = minimize(
                    _obj_bipartite,
                    np.array([R1_0, R2_0, phi_0]),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 10000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if res_bip.x is not None:
                    pts = _build_bipartite(res_bip.x[0], res_bip.x[1], res_bip.x[2])
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_bip_score:
                            best_bip_score = sc
                            best_bip_pts = pts
            except Exception:
                pass
        
        if best_bip_score > best_score:
            best_score = best_bip_score
            best_pts = best_bip_pts

    # --- Final Safety Check ---
    if best_pts is None or _polygon_score(best_pts) <= 0.0:
        # Fallback to regular 20-gon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return best_pts

# EVOLVE-BLOCK-END
