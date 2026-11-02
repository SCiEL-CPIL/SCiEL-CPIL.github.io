"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

Structural change: replace the previous symmetry-constrained grid search + generic local
optimizers with a Fermat-spiral / golden-angle initialization family plus a
worst-triangle-targeted simulated annealing that uses a smooth soft-min surrogate for
acceptance but tracks the exact objective for bookkeeping.  This changes the representation
of the search (angular spiral parameterization and a different annealing schedule) and the
optimization method (SA with worst-triangle focus) rather than merely re-tuning constants.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 145.0  # safety margin before the 150 s wall-clock kill

# Pre-compute all C(13,3)=286 triple indices for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)
_num_triples = len(_triple_i)


def _triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _soft_min_score(pts, temperature=0.02):
    """Smooth surrogate of min triangle area / hull area using soft-min."""
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    # soft min: -T * log(mean(exp(-areas/T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -temperature * (log_sum - np.log(len(areas)))
    return soft_min / hull_area


def _make_initial_guesses():
    """Generate a diverse set of initial point configurations.

    The primary family is the Fermat (golden-angle) spiral, which is known to
    distribute points with good minimal-distance and area properties.  We also
    include regular polygons and two-layer configurations for diversity.
    """
    inits = []

    # Fermat spiral with various radii and rotations
    for r_scale in [0.8, 1.0, 1.2]:
        for rot in [0.0, np.pi / 13, np.pi / 6]:
            t = np.arange(N)
            r = r_scale * np.sqrt(t / (N - 1))
            theta = t * 2 * np.pi / 13 + rot  # golden angle approx 2*pi/13 for 13 pts
            # Use golden angle ~ 2.39996 rad for better distribution
            theta = t * 2.399963229728653 + rot
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            inits.append(pts.copy())

    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_reg.copy())

    # Two-layer: 9 outer + 4 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6]:
        outer_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 9
        pts = np.vstack([
            np.column_stack([np.cos(outer_angles), np.sin(outer_angles)]),
            np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
        ])
        inits.append(pts)

    # Two-layer: 8 outer + 5 inner
    for r_inner in [0.35, 0.45, 0.55]:
        outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 8
        pts = np.vstack([
            np.column_stack([np.cos(outer_angles), np.sin(outer_angles)]),
            np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
        ])
        inits.append(pts)

    # Three-layer: 6 + 5 + 2
    for r2, r3 in [(0.5, 0.2), (0.6, 0.25), (0.55, 0.15)]:
        a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 6
        a3 = np.linspace(0, 2 * np.pi, 2, endpoint=False) + np.pi / 5
        pts = np.vstack([
            np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
            np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
        ])
        inits.append(pts)

    return inits


def _simulated_annealing(init_pts, t_start, rng, max_time=25.0):
    """Worst-triangle-targeted simulated annealing.

    At each step, pick the worst triangle, move one of its vertices with a
    Gaussian perturbation, and accept/reject based on a soft-min surrogate
    while tracking the exact objective.
    """
    t0 = t_start
    best_pts = init_pts.copy()
    best_score = _score_points(best_pts)
    cur_pts = init_pts.copy()
    cur_score = best_score

    T_start = 0.15
    T_end = 0.001
    n_steps = 20000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))

    for step in range(n_steps):
        if (time.time() - t0) > max_time:
            break
        T = T_start * (cooling ** step)

        # Compute areas and find worst triangle
        areas = _triangle_areas(cur_pts)
        min_area = np.min(areas)
        worst_idx = np.argmin(areas)

        # Pick one of the three vertices of the worst triangle to move
        i = _triple_i[worst_idx]
        j = _triple_j[worst_idx]
        k = _triple_k[worst_idx]
        move_idx = rng.choice([i, j, k])

        # Gaussian perturbation scaled by temperature
        sigma = T * 0.5
        delta = rng.normal(0, sigma, size=2)
        new_pts = cur_pts.copy()
        new_pts[move_idx] += delta

        # Score the new point
        new_score = _score_points(new_pts)
        if new_score <= 0:
            continue

        # Acceptance: use soft-min surrogate for a smoother landscape
        cur_soft = _soft_min_score(cur_pts, temperature=T)
        new_soft = _soft_min_score(new_pts, temperature=T)

        if new_soft >= cur_soft:
            accept = True
        else:
            delta_energy = cur_soft - new_soft
            if delta_energy < 0:
                accept = True
            else:
                accept = rng.rand() < np.exp(-delta_energy / max(T, 1e-12))

        if accept:
            cur_pts = new_pts
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()

    return best_pts, best_score


def _local_refine(pts, t_start, max_time=5.0):
    """Quick local refinement using Nelder-Mead on the exact objective."""
    from scipy.optimize import minimize
    t0 = t_start
    params = pts.flatten().copy()

    def neg_score(p):
        return -_score_points(p.reshape(N, 2))

    try:
        res = minimize(
            neg_score,
            params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        cand = res.x.reshape(N, 2)
        score = _score_points(cand)
        if score > 0:
            return cand, score
    except Exception:
        pass
    return pts, _score_points(pts)


def construct():
    t_start = time.time()
    rng = np.random.RandomState(42)

    best_pts = None
    best_score = 0.0

    # Generate diverse initializations
    inits = _make_initial_guesses()

    # Evaluate all initializations
    scored_inits = []
    for init in inits:
        score = _score_points(init)
        if score > 0:
            scored_inits.append((init, score))

    if not scored_inits:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return best_pts

    # Sort by score descending
    scored_inits.sort(key=lambda x: -x[1])

    # Run simulated annealing from the top initializations
    top_k = min(len(scored_inits), 8)
    for idx in range(top_k):
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
        init_pts, init_score = scored_inits[idx]
        sa_pts, sa_score = _simulated_annealing(init_pts, t_start, rng, max_time=12.0)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts

    # Local refinement on the best solution
    if best_pts is not None and (time.time() - t_start) < (DEADLINE - 5.0):
        ref_pts, ref_score = _local_refine(best_pts, t_start, max_time=5.0)
        if ref_score > best_score:
            best_score = ref_score
            best_pts = ref_pts

    # Safety fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_pts.reshape(N, 2)
