# EVOLVE-BLOCK-START
import time
import numpy as np


def _tri_areas(pts, idx):
    """Vectorized triangle areas for all triples in idx (shape (m,3))."""
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    a = 0.5 * np.abs(
        p0[:, 0] * (p1[:, 1] - p2[:, 1])
        + p1[:, 0] * (p2[:, 1] - p0[:, 1])
        + p2[:, 0] * (p0[:, 1] - p1[:, 1])
    )
    return a


def _hull_area(pts):
    """Area of convex polygon given in CCW order (shoelace)."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _objective(radii, angles, idx, unit_hull):
    """Return A_min / A_hull for a radial configuration (all on convex hull)."""
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    amin = _tri_areas(pts, idx).min()
    ahull = unit_hull * float(np.prod(radii))
    if not (np.isfinite(amin) and np.isfinite(ahull)) or ahull <= 0:
        return -1.0
    return float(amin / ahull)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in convex position maximizing
    (smallest triangle area) / (convex hull area).

    Parameterization: 13 points at regular 13-gon angles with variable radii,
    guaranteeing convex position. Optimized via multi-start simulated annealing
    plus Nelder-Mead polish, under a wall-clock deadline.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates.
    """
    n = 13
    t0 = time.time()
    DEADLINE = 560.0  # safety margin before the 630s kill

    # Fixed regular 13-gon angles
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # Precompute all C(13,3)=286 triangle index triples
    idx = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    )

    # Unit-hull area (radii all 1)
    unit_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    unit_hull = _hull_area(unit_pts)

    def time_left():
        return DEADLINE - (time.time() - t0)

    def make_points(radii):
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    # ---- Build a diverse pool of starting radii ----
    rng = np.random.default_rng(20240611)
    starts = []
    starts.append(np.ones(n))  # regular 13-gon
    for _ in range(24):
        r = np.ones(n) * (0.85 + 0.30 * rng.random())
        r += 0.12 * rng.standard_normal(n)
        r = np.clip(r, 0.35, 1.6)
        starts.append(r)

    best_score = -1.0
    best_radii = starts[0].copy()

    for r0 in starts:
        if time_left() < 15.0:
            break

        # ---- Simulated annealing on radii ----
        r = r0.copy()
        s0 = _objective(r, angles, idx, unit_hull)
        if s0 > best_score:
            best_score = s0
            best_radii = r.copy()

        cur = s0
        T = 0.004
        cool = 0.985
        iters = 3000
        step = 0.045

        for it in range(iters):
            if time_left() < 12.0:
                break
            # perturb one radius
            i = rng.integers(0, n)
            r_new = r.copy()
            r_new[i] += step * rng.standard_normal()
            r_new[i] = np.clip(r_new[i], 0.35, 1.6)
            s_new = _objective(r_new, angles, idx, unit_hull)
            if s_new > cur or rng.random() < np.exp((s_new - cur) / max(T, 1e-9)):
                r = r_new
                cur = s_new
                if cur > best_score:
                    best_score = cur
                    best_radii = r.copy()
            T *= cool
            step *= 0.9995

        # ---- Nelder-Mead polish on the best found so far ----
        if time_left() > 25.0:
            try:
                from scipy.optimize import minimize

                r_start = best_radii.copy()
                res = minimize(
                    lambda rr: -_objective(rr, angles, idx, unit_hull),
                    r_start,
                    method="Nelder-Mead",
                    bounds=[(0.35, 1.6)] * n,
                    options={"maxiter": 2500, "xatol": 1e-6, "fatol": 1e-8, "adaptive": True},
                )
                s_pol = _objective(res.x, angles, idx, unit_hull)
                if s_pol > best_score:
                    best_score = s_pol
                    best_radii = np.clip(res.x, 0.35, 1.6)
            except Exception:
                pass

    points = make_points(best_radii)
    return points


# EVOLVE-BLOCK-END
