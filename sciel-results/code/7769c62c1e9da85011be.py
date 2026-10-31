"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 150s limit

    # Precompute all triples
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    n_triples = len(triples)

    def _compute_areas(x):
        """Vectorized triangle area computation."""
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def _hull_area(x):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(x)
            return hull.volume
        except Exception:
            return 0.0

    def score(x):
        """Returns the minimum triangle area divided by convex hull area. x is (N,2)."""
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        min_area = np.min(areas)
        return min_area / hull_area

    def score_flat(x_flat):
        """Score from flat array."""
        return score(x_flat.reshape(N, 2))

    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])

    def coordinate_descent_polish(x, max_rounds=50, tol=1e-12):
        """
        Coordinate descent: for each point, optimize its (x,y) position
        given all other points fixed, to maximize the minimum triangle area
        involving that point (normalized by hull area).
        """
        x = x.copy().astype(float)
        for rnd in range(max_rounds):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                # Triangles involving point i
                tri_mask = (idx_i == i) | (idx_j == i) | (idx_k == i)
                tri_areas = _compute_areas(x)
                min_area_current = np.min(tri_areas)
                hull_area = _hull_area(x)
                if hull_area < 1e-10:
                    continue
                score_current = min_area_current / hull_area

                # For each triangle involving i, compute area as function of x_i
                # Area of triangle (a,b,c) = 0.5 * |cross(b-a, c-a)|
                # If we vary point i, the area is linear in x_i coordinates
                # We want to maximize the minimum of these areas
                # Use a simple 1D/2D search: try moving in small steps

                # Get the critical triangles (smallest areas involving i)
                crit_indices = np.where(tri_mask)[0]
                if len(crit_indices) == 0:
                    continue
                crit_areas = tri_areas[crit_indices]
                min_crit = np.min(crit_areas)

                # Optimize position of point i using Nelder-Mead on 2D
                def obj_i(x_i_new):
                    x_trial = x.copy()
                    x_trial[i] = x_i_new
                    hull_a = _hull_area(x_trial)
                    if hull_a < 1e-10:
                        return 0.0
                    areas_t = _compute_areas(x_trial)
                    return -np.min(areas_t) / hull_a

                try:
                    res = optimize.minimize(
                        obj_i, x[i],
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if -res.fun > score_current * (1 + 1e-10):
                        x[i] = res.x
                        improved = True
                except Exception:
                    pass

            if not improved:
                break

        return x

    def polar_coordinate_descent(x, max_rounds=30):
        """
        Optimize in polar coordinates (r, theta) for each point.
        This is a different parameterization that can escape local minima.
        """
        x = x.copy().astype(float)
        for rnd in range(max_rounds):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                r_i = np.linalg.norm(x[i])
                if r_i < 1e-10:
                    theta_i = 0.0
                else:
                    theta_i = np.arctan2(x[i, 1], x[i, 0])

                def obj_polar(p):
                    r_new = max(p[0], 0.01)
                    theta_new = p[1]
                    x_trial = x.copy()
                    x_trial[i] = [r_new * np.cos(theta_new), r_new * np.sin(theta_new)]
                    hull_a = _hull_area(x_trial)
                    if hull_a < 1e-10:
                        return 0.0
                    areas_t = _compute_areas(x_trial)
                    return -np.min(areas_t) / hull_a

                try:
                    res = optimize.minimize(
                        obj_polar, [r_i, theta_i],
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-7}
                    )
                    r_new = max(res.x[0], 0.01)
                    theta_new = res.x[1]
                    new_pos = np.array([r_new * np.cos(theta_new), r_new * np.sin(theta_new)])
                    x_trial = x.copy()
                    x_trial[i] = new_pos
                    s_new = score(x_trial)
                    s_old = score(x)
                    if s_new > s_old * (1 + 1e-10):
                        x = x_trial
                        improved = True
                except Exception:
                    pass

            if not improved:
                break

        return x

    def de_global_search(bounds, max_iter=200, popsize=15, seed=42):
        """Differential evolution for global search."""
        rng = np.random.RandomState(seed)
        try:
            res = optimize.differential_evolution(
                lambda x: -score_flat(x),
                bounds,
                maxiter=max_iter,
                popsize=popsize,
                seed=seed,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                updating='deferred'
            )
            return res.x
        except Exception:
            return None

    best_x = None
    best_score = -np.inf

    def try_update(x_candidate):
        nonlocal best_x, best_score
        x_2d = x_candidate.reshape(N, 2)
        s = score(x_2d)
        if s > best_score:
            best_score = s
            best_x = x_2d.copy()
        return s

    # Phase 0: Regular polygon baseline
    x_reg = regular_polygon(N)
    try_update(x_reg)

    # Phase 1: Differential evolution global search
    # Bounds: points in unit disk (roughly)
    bounds_de = [(-1.5, 1.5)] * (2 * N)
    if time.time() < deadline - 40:
        x_de = de_global_search(bounds_de, max_iter=80, popsize=12, seed=12345)
        if x_de is not None:
            try_update(x_de)

    # Phase 2: Coordinate descent polish on best
    if best_x is not None and time.time() < deadline - 30:
        x_cd = coordinate_descent_polish(best_x, max_rounds=30)
        try_update(x_cd)

    # Phase 3: Polar coordinate descent from best
    if best_x is not None and time.time() < deadline - 20:
        x_polar = polar_coordinate_descent(best_x, max_rounds=20)
        try_update(x_polar)

    # Phase 4: More DE restarts with different seeds
    for seed in [999, 777, 555, 333]:
        if time.time() > deadline - 25:
            break
        x_de = de_global_search(bounds_de, max_iter=50, popsize=10, seed=seed)
        if x_de is not None:
            s_de = try_update(x_de)
            if s_de > best_score * 0.95:  # Promising
                if time.time() < deadline - 10:
                    x_cd = coordinate_descent_polish(x_de.reshape(N, 2), max_rounds=20)
                    try_update(x_cd)

    # Phase 5: Multi-start coordinate descent from diverse initializations
    starts = []
    # Regular polygon variants
    for phase in [0.0, np.pi/N, 2*np.pi/N]:
        starts.append(regular_polygon(N, phase=phase))

    # Two-ring structures
    for n_inner in [3, 4, 5]:
        n_outer = N - n_inner
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + np.pi/n_outer
        x_inner = np.column_stack([0.4*np.cos(angles_inner), 0.4*np.sin(angles_inner)])
        x_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        starts.append(np.vstack([x_inner, x_outer]))

    # Quasi-regular: regular polygon with radial modulation
    for k in [2, 3]:
        base = regular_polygon(N)
        theta = np.arctan2(base[:, 1], base[:, 0])
        r = 1.0 + 0.08 * np.sin(k * theta)
        starts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))

    # Random in disk
    for _ in range(4):
        angles = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii = np.sqrt(np.random.uniform(0.6, 1.0, N))
        starts.append(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]))

    for i, x_start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        x_cd = coordinate_descent_polish(x_start, max_rounds=25)
        try_update(x_cd)
        if time.time() < deadline - 8:
            x_polar = polar_coordinate_descent(x_cd, max_rounds=10)
            try_update(x_polar)

    # Phase 6: Final intensive coordinate descent
    if best_x is not None and time.time() < deadline - 5:
        x_final = coordinate_descent_polish(best_x, max_rounds=40, tol=1e-13)
        try_update(x_final)

    # Phase 7: Final polar polish
    if best_x is not None and time.time() < deadline - 3:
        x_final2 = polar_coordinate_descent(best_x, max_rounds=15)
        try_update(x_final2)

    if best_x is None:
        best_x = regular_polygon(N)

    return best_x

# EVOLVE-BLOCK-END
