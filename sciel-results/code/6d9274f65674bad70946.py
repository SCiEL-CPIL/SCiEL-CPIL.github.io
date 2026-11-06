"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score_from_points(pts):
    """Compute the score Delta/n^n after scaling to diameter 2."""
    pts = np.array(pts, dtype=float)
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Diameter
    d = np.max(dists)
    if d < 1e-12:
        return 0.0
    # Scale to diameter 2
    scale = 2.0 / d
    dists_scaled = dists * scale
    # Product over ordered pairs (i != j)
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    prod_log = np.sum(np.log(dists_scaled[mask]))
    log_score = prod_log - n * n * np.log(n)
    return np.exp(log_score)

def _log_score_from_points(pts):
    """Compute log of the score Delta/n^n after scaling to diameter 2."""
    pts = np.array(pts, dtype=float)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    d = np.max(dists)
    if d < 1e-12:
        return -np.inf
    scale = 2.0 / d
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    # Avoid log(0) by clipping
    dists_scaled = np.maximum(dists_scaled, 1e-15)
    prod_log = np.sum(np.log(dists_scaled[mask]))
    return prod_log - n * n * np.log(n)

def construct():
    """
    Find 8 points maximizing the product of pairwise distances at fixed diameter.
    Uses angular parameterization on a circle with variable gaps, optimized
    via scipy's SLSQP with diameter constraint.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n = 8
    
    # Start from regular octagon
    angles0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    def angles_to_points(angles):
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def neg_log_score(angles):
        pts = angles_to_points(angles)
        return -_log_score_from_points(pts)
    
    # Constraint: diameter <= 2 (after scaling, we want diameter exactly 2,
    # but since we normalize, we just need the raw diameter to be consistent)
    # Actually, the score is scale-invariant (we scale to diameter 2), so we can
    # just optimize the log-product of distances minus n^2*log(n) with the
    # scaling built in. The angles define shape only.
    
    # But wait - the points are on the unit circle in this parameterization.
    # The diameter of the unit circle is 2, so all pairwise distances are <= 2.
    # The actual diameter depends on the angular gaps.
    # Since we scale to diameter 2, the score only depends on the shape (angles).
    
    # Let's just optimize the angles directly.
    # We need to handle the circular nature: sum of gaps = 2*pi.
    # Use 7 gaps (the 8th is determined).
    
    def gaps_to_angles(gaps):
        gaps = np.concatenate([gaps, [2*np.pi - np.sum(gaps)]])
        angles = np.cumsum(gaps) - gaps[0]
        return angles % (2*np.pi)
    
    def neg_log_score_from_gaps(gaps):
        angles = gaps_to_angles(gaps)
        pts = angles_to_points(angles)
        return -_log_score_from_points(pts)
    
    # Initial: regular octagon gaps
    gaps0 = np.full(n-1, 2*np.pi/n)
    
    # Bounds: each gap in (epsilon, 2*pi - epsilon)
    eps = 1e-6
    bounds = [(eps, 2*np.pi - eps)] * (n-1)
    
    # Constraint: sum of gaps < 2*pi (so the 8th gap is positive)
    def sum_constraint(gaps):
        return 2*np.pi - np.sum(gaps)
    
    constraints = [{'type': 'ineq', 'fun': sum_constraint}]
    
    best_score = _score_from_points(angles_to_points(angles0))
    best_gaps = gaps0.copy()
    
    # Try multiple starting points
    rng = np.random.RandomState(42)
    starts = [gaps0.copy()]
    
    # Add some perturbed starts
    for i in range(5):
        pert = rng.uniform(-0.3, 0.3, size=n-1)
        g = gaps0 + pert
        # Ensure positive and sum < 2*pi
        g = np.maximum(g, eps)
        s = np.sum(g)
        if s < 2*np.pi - eps:
            g = g / s * (2*np.pi - eps)
            starts.append(g)
    
    # Also try some asymmetric starts
    for i in range(3):
        g = rng.uniform(0.3, 1.2, size=n-1)
        s = np.sum(g)
        g = g / s * (2*np.pi - 0.1)
        starts.append(g)
    
    result_history = []
    
    for idx, g0 in enumerate(starts):
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(
                neg_log_score_from_gaps,
                g0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if time.time() - start_time > deadline:
                break
            if res.success or res.fun < np.inf:
                sc = np.exp(-res.fun)
                if sc > best_score:
                    best_score = sc
                    best_gaps = res.x.copy()
                result_history.append((sc, res.x.copy()))
        except Exception:
            pass
    
    # Try differential evolution if we have time
    if time.time() - start_time < deadline - 30:
        from scipy.optimize import differential_evolution
        try:
            de_res = differential_evolution(
                neg_log_score_from_gaps,
                bounds=bounds,
                constraints=constraints,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=True
            )
            if time.time() - start_time > deadline:
                pass
            else:
                if de_res.fun < np.inf:
                    sc = np.exp(-de_res.fun)
                    if sc > best_score:
                        best_score = sc
                        best_gaps = de_res.x.copy()
        except Exception:
            pass
    
    # Final local refinement from best
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                neg_log_score_from_gaps,
                best_gaps,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if time.time() - start_time > deadline:
                pass
            else:
                if res.fun < np.inf:
                    sc = np.exp(-res.fun)
                    if sc > best_score:
                        best_score = sc
                        best_gaps = res.x.copy()
        except Exception:
            pass
    
    # Convert best gaps to points
    angles = gaps_to_angles(best_gaps)
    pts = angles_to_points(angles)
    
    # Return as list of [x, y]
    return [[float(pts[i][0]), float(pts[i][1])] for i in range(n)]

# EVOLVE-BLOCK-END
