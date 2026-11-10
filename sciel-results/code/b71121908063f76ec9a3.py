# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute sofa points
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_N_SOFa = _SOFA_X.size
_SOFA_X_1D = _SOFA_X.ravel()
_SOFA_Y_1D = _SOFA_Y.ravel()


def _generate_placements(waypoints):
    """Generate all placements including start, waypoints, end, with linear interpolation."""
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.asarray(p, dtype=np.float64) for p in waypoints] + [PN]
    placements = [full[0].copy()]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append(p1 + t * (p2 - p1))
    return placements


def _compute_area(waypoints):
    """Compute the area of the sofa intersection. Returns count * 0.0001."""
    placements = _generate_placements(waypoints)
    inside_all = np.ones(_N_SOFa, dtype=bool)
    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        dx = _SOFA_X_1D - x
        dy = _SOFA_Y_1D - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        in_h = (u >= -1e-12) & (u <= 1 + 1e-12) & (v >= -3 - 1e-12) & (v <= 1 + 1e-12)
        in_v = (u >= -3 - 1e-12) & (u <= 1 + 1e-12) & (v >= -1e-12) & (v <= 1 + 1e-12)
        in_base = in_h | in_v
        inside_all &= in_base
        if not inside_all.any():
            break
    return inside_all.sum() * 0.0001


def _smooth_inside_value(waypoints, beta=50.0):
    """
    Smooth surrogate: for each placement, compute a smooth 'inside' value in [0,1]
    using soft-min of distances to the boundary. Then take the min over placements
    (soft-min) and sum over sofa points.
    
    Actually, let's use a different approach: for each point, compute the minimum
    distance to the boundary across all placements, and use a smooth step.
    """
    placements = _generate_placements(waypoints)
    # For each placement, compute signed distance to the L-shape boundary
    # This is complex. Instead, use a simpler smooth approximation:
    # For each point and placement, compute a smooth indicator.
    
    # Simpler approach: use the hard indicator but with a soft-min over placements
    # Actually, let's just use the hard computation but with a smoothing parameter
    # for the min over placements.
    
    # For now, use the hard computation as the objective, but add a small
    # smoothing to help the optimizer.
    
    # Let's use a different strategy: compute the area directly, but for
    # optimization, use a smooth approximation of the minimum over placements.
    
    # For each sofa point, compute the minimum "margin" across all placements.
    # Margin = min over placements of (distance to boundary)
    # If margin > 0, point is inside.
    
    # This is expensive. Let's just use the hard computation.
    return _compute_area(waypoints)


def _neg_area(params):
    n_wp = params.shape[0] // 3
    waypoints = params.reshape(n_wp, 3)
    return -_compute_area(waypoints)


def _neg_smooth_area(params, beta=50.0):
    """Smooth objective for optimization."""
    n_wp = params.shape[0] // 3
    waypoints = params.reshape(n_wp, 3)
    return -_compute_area(waypoints)


def _generate_initial_candidates(n_wp):
    """Generate diverse initial candidate paths."""
    ts = np.linspace(0, 1, n_wp)
    candidates = []
    
    # S-curve variations
    for amp in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for x_scale in [1.0, 1.2, 1.5]:
            candidates.append(np.column_stack([
                -x_scale * ts,
                amp * np.sin(np.pi * ts),
                90.0 * ts
            ]))
    
    # Early turn
    for p in [1.2, 1.5, 2.0]:
        candidates.append(np.column_stack([
            -1.2 * ts**p,
            0.3 * np.sin(np.pi * ts),
            90.0 * ts
        ]))
    
    # Late turn
    for p in [0.5, 0.7, 0.8]:
        candidates.append(np.column_stack([
            -1.2 * ts**p,
            0.3 * np.sin(np.pi * ts),
            90.0 * ts
        ]))
    
    # Arc-like paths
    for r in [1.0, 1.5, 2.0, 2.5]:
        angles = np.linspace(0, np.pi/2, n_wp)
        x = r * (1 - np.cos(angles)) - 1.0
        y = r * np.sin(angles)
        a = 90.0 * angles / (np.pi/2)
        candidates.append(np.column_stack([x, y, a]))
    
    return candidates


def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    N_WAYPOINTS = 12
    
    # Generate initial candidates
    candidates = _generate_initial_candidates(N_WAYPOINTS)
    
    # Evaluate all candidates
    scored = []
    for c in candidates:
        if time.time() > deadline - 300:
            break
        area = _compute_area(c)
        scored.append((area, c))
    
    if not scored:
        # Fallback
        ts = np.linspace(0, 1, N_WAYPOINTS)
        waypoints = np.column_stack([
            -1.2 * ts,
            0.3 * np.sin(np.pi * ts),
            90.0 * ts
        ])
        return waypoints
    
    scored.sort(key=lambda x: -x[0])
    
    best_params = scored[0][1].ravel().copy()
    best_area = scored[0][0]
    
    # Optimize top candidates with Nelder-Mead
    for i, (sp_area, sp) in enumerate(scored[:6]):
        if time.time() > deadline - 150:
            break
        try:
            result = minimize(
                _neg_area,
                sp.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 400,
                    'maxfev': 400,
                    'xatol': 1e-4,
                    'fatol': 1e-5,
                    'adaptive': True,
                }
            )
            candidate = result.x.reshape(N_WAYPOINTS, 3)
            cand_area = _compute_area(candidate)
            if cand_area > best_area:
                best_area = cand_area
                best_params = candidate.ravel().copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 50:
        try:
            result = minimize(
                _neg_area,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 800,
                    'maxfev': 800,
                    'xatol': 1e-5,
                    'fatol': 1e-6,
                    'adaptive': True,
                }
            )
            candidate = result.x.reshape(N_WAYPOINTS, 3)
            cand_area = _compute_area(candidate)
            if cand_area > best_area:
                best_area = cand_area
                best_params = candidate.ravel().copy()
        except Exception:
            pass
    
    # Second round of optimization from slightly perturbed best
    if time.time() < deadline - 30:
        try:
            rng = np.random.RandomState(42)
            for trial in range(3):
                if time.time() > deadline - 10:
                    break
                perturbed = best_params + rng.randn(*best_params.shape) * 0.01
                result = minimize(
                    _neg_area,
                    perturbed,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'maxfev': 300,
                        'xatol': 1e-5,
                        'fatol': 1e-6,
                        'adaptive': True,
                    }
                )
                candidate = result.x.reshape(N_WAYPOINTS, 3)
                cand_area = _compute_area(candidate)
                if cand_area > best_area:
                    best_area = cand_area
                    best_params = candidate.ravel().copy()
        except Exception:
            pass
    
    waypoints = best_params.reshape(N_WAYPOINTS, 3)
    
    if not np.all(np.isfinite(waypoints)):
        ts = np.linspace(0, 1, N_WAYPOINTS)
        waypoints = np.column_stack([
            -1.2 * ts,
            0.3 * np.sin(np.pi * ts),
            90.0 * ts
        ])
    
    return waypoints

# EVOLVE-BLOCK-END
