"""50 equal circles in a unit circle - ring-structured optimization."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 50
# Ring structure: 1 centre + rings of 6, 12, 18, 13 (matches known best packing)
RING_SIZES = [1, 6, 12, 18, 13]
assert sum(RING_SIZES) == N

# Precompute angular positions for each circle within its ring
# Ring i (0-indexed, i>=1) has RING_SIZES[i] circles evenly spaced
# We store (ring_index, angle_offset) for each circle
_circle_info = []
for ri, rs in enumerate(RING_SIZES):
    if ri == 0:
        _circle_info.append((0, 0.0))
    else:
        for k in range(rs):
            # Evenly spaced around the ring
            _circle_info.append((ri, 2.0 * np.pi * k / rs))
N_CIRCLES = len(_circle_info)
assert N_CIRCLES == N

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _build_centres(params):
    """Build centre coordinates from ring parameters.
    params: array of length 4*2 = 8 (radius and angle_offset per non-centre ring)
    """
    # params[2*i] = radius of ring i (i=0..3), params[2*i+1] = angle offset
    radii = np.array([0.0] + [params[2*i] for i in range(4)])
    ang_offs = np.array([0.0] + [params[2*i+1] for i in range(4)])

    xs = np.zeros(N_CIRCLES)
    ys = np.zeros(N_CIRCLES)
    for ci, (ri, ang) in enumerate(_circle_info):
        if ri == 0:
            xs[ci] = 0.0
            ys[ci] = 0.0
        else:
            r = radii[ri]
            a = ang + ang_offs[ri]
            xs[ci] = r * np.cos(a)
            ys[ci] = r * np.sin(a)
    return np.column_stack([xs, ys])


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    # Distance from each centre to origin
    dist_origin = np.hypot(centres[:, 0], centres[:, 1])
    # Max radius from containment: 1 - max distance to origin
    r_contain = 1.0 - np.max(dist_origin)
    # Min pairwise distance / 2
    # Use vectorized approach
    min_pair = np.inf
    for i in range(N_CIRCLES):
        d = np.hypot(centres[i+1:, 0] - centres[i, 0], centres[i+1:, 1] - centres[i, 1])
        if d.size > 0:
            m = np.min(d)
            if m < min_pair:
                min_pair = m
    r_pair = min_pair / 2.0
    return min(r_contain, r_pair)


def _soft_violation(params, target_r):
    """Soft cost: how much the packing violates the target radius.
    Returns 0 if valid, >0 if violated. Lower is better.
    """
    centres = _build_centres(params)
    # Containment violation
    dist_origin = np.hypot(centres[:, 0], centres[:, 1])
    contain_viol = np.max(dist_origin + target_r - 1.0)  # >0 if any circle sticks out

    # Pairwise violation
    pair_viol = 0.0
    for i in range(N_CIRCLES):
        d = np.hypot(centres[i+1:, 0] - centres[i, 0], centres[i+1:, 1] - centres[i, 1])
        if d.size > 0:
            v = 2.0 * target_r - np.min(d)
            if v > pair_viol:
                pair_viol = v

    return max(contain_viol, pair_viol)


def _binary_search_radius(params, lo, hi, tol=1e-10, max_iter=60):
    """Find max valid radius for given ring parameters via binary search."""
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if _soft_violation(params, mid) <= 1e-12:
            lo = mid
        else:
            hi = mid
    return lo


def _de_objective(params):
    """Objective for DE: maximize radius (negate for minimization)."""
    r = _binary_search_radius(params, 0.05, 0.15, tol=1e-8, max_iter=40)
    return -r


def construct():
    global _DEADLINE
    t_start = time.time()

    # Initial guess: evenly spaced rings
    # Ring radii roughly proportional to sqrt of cumulative circles
    cum = np.cumsum(RING_SIZES)
    # Approximate: ring i radius ~ sqrt(cum[i]/N) * 0.9
    init_radii = np.sqrt(cum[1:] / N) * 0.85
    init_params = np.zeros(8)
    for i in range(4):
        init_params[2*i] = init_radii[i]
        init_params[2*i+1] = 0.0  # no angle offset initially

    # First, get a baseline
    best_params = init_params.copy()
    best_r = _binary_search_radius(best_params, 0.05, 0.15, tol=1e-10, max_iter=50)

    # Differential evolution over the 8-parameter ring space
    bounds = [
        (0.0, 0.95),  # ring 1 radius
        (-0.5, 0.5),  # ring 1 angle offset
        (0.0, 0.95),  # ring 2 radius
        (-0.5, 0.5),  # ring 2 angle offset
        (0.0, 0.95),  # ring 3 radius
        (-0.5, 0.5),  # ring 3 angle offset
        (0.0, 0.95),  # ring 4 radius
        (-0.5, 0.5),  # ring 4 angle offset
    ]

    # Run DE with time check
    elapsed = time.time() - t_start
    if elapsed < 80:  # enough time for DE
        try:
            de_result = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 0.8),
                recombination=0.9,
                polish=False,
                init='sobol',
                callback=None,
            )
            if de_result.success or de_result.fun < -best_r:
                candidate = de_result.x
                r_candidate = _binary_search_radius(candidate, 0.05, 0.15, tol=1e-10, max_iter=50)
                if r_candidate > best_r:
                    best_params = candidate
                    best_r = r_candidate
        except Exception:
            pass

    # Polish with Nelder-Mead on the continuous parameters
    elapsed = time.time() - t_start
    if elapsed < 110:
        try:
            def polish_obj(p):
                return -_binary_search_radius(p, 0.05, 0.15, tol=1e-10, max_iter=50)

            nm_result = minimize(
                polish_obj,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if nm_result.fun < -best_r:
                r_nm = -nm_result.fun
                if r_nm > best_r:
                    best_params = nm_result.x
                    best_r = r_nm
        except Exception:
            pass

    # Final high-precision polish: coordinate descent on individual ring params
    elapsed = time.time() - t_start
    if elapsed < 130:
        try:
            # Fine-tune each parameter with a small grid search
            for dim in range(8):
                p0 = best_params[dim]
                # Try small perturbations
                for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                    test_p = best_params.copy()
                    test_p[dim] = p0 + delta
                    # Clamp to bounds
                    test_p[dim] = max(bounds[dim][0], min(bounds[dim][1], test_p[dim]))
                    r_test = _binary_search_radius(test_p, 0.05, 0.15, tol=1e-10, max_iter=50)
                    if r_test > best_r:
                        best_params = test_p
                        best_r = r_test
                        p0 = best_params[dim]
        except Exception:
            pass

    # Build final centres
    centres = _build_centres(best_params)

    # Verify validity
    dist_origin = np.hypot(centres[:, 0], centres[:, 1])
    if np.max(dist_origin) >= 1.0:
        # Scale down slightly
        scale = (1.0 - 1e-9) / np.max(dist_origin)
        centres = centres * scale

    # Final safety: ensure all circles are strictly inside
    dist_origin = np.hypot(centres[:, 0], centres[:, 1])
    r_final = _compute_radius(centres)
    # Shrink if needed to ensure validity
    if r_final <= 0:
        # Fallback to simple hexagonal packing
        return _fallback_hex()

    return centres


def _fallback_hex():
    """Simple hexagonal packing as fallback."""
    spacing = 0.24
    points = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = i * spacing + 0.5 * spacing * (j % 2)
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0 - spacing / 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Sort by distance from origin, take closest N
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        return points[idx]
    return np.zeros((N, 2))
