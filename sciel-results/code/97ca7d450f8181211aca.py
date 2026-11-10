# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin

    # Precompute triangle indices
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    tri_a = tri_indices[:, 0]
    tri_b = tri_indices[:, 1]
    tri_c = tri_indices[:, 2]

    def compute_areas(points):
        """Compute all 286 triangle areas vectorized."""
        pa = points[tri_a]
        pb = points[tri_b]
        pc = points[tri_c]
        return 0.5 * np.abs(pa[:, 0] * (pb[:, 1] - pc[:, 1]) +
                            pb[:, 0] * (pc[:, 1] - pa[:, 1]) +
                            pc[:, 0] * (pa[:, 1] - pb[:, 1]))

    def compute_score(points):
        """Compute A_min / A_hull ratio."""
        if not np.all(np.isfinite(points)):
            return 0.0
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
        except Exception:
            return 0.0
        if a_hull <= 1e-12:
            return 0.0
        areas = compute_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull

    def coordinate_descent(points, max_iter=500, step_init=0.1):
        """
        Direct minimax coordinate descent: for each point, try small moves
        in each coordinate direction and keep the move that maximizes the
        minimum triangle area (or the ratio).
        """
        points = points.copy()
        best_score = compute_score(points)
        step = step_init
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(n):
                for d in range(2):
                    best_local = best_score
                    best_move = 0.0
                    # Try several step sizes in both directions
                    for s in [step, step*0.5, step*0.25, step*2.0]:
                        for sign in [1.0, -1.0]:
                            p_new = points.copy()
                            p_new[i, d] += sign * s
                            sc = compute_score(p_new)
                            if sc > best_local:
                                best_local = sc
                                best_move = sign * s
                    if best_move != 0.0:
                        points[i, d] += best_move
                        best_score = best_local
                        improved = True
            if not improved:
                step *= 0.7
                if step < 1e-6:
                    break
        return points, best_score

    def simulated_annealing(points, t0=0.5, t_end=1e-4, alpha=0.995, max_iter=3000):
        """
        Simulated annealing on the point set, accepting moves that worsen
        the score with a probability based on temperature.
        """
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        t = t0
        rng = np.random.default_rng(seed=42)
        for it in range(max_iter):
            # Pick a random point and a random perturbation
            i = rng.integers(0, n)
            d = rng.integers(0, 2)
            delta = rng.normal(0, 0.05)
            p_new = points.copy()
            p_new[i, d] += delta
            new_score = compute_score(p_new)
            if new_score > current_score or (t > t_end and rng.random() < np.exp((new_score - current_score) / max(t, 1e-10))):
                points = p_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            t *= alpha
        return best_points, best_score

    def nelder_mead_refine(points, max_iter=200):
        """
        Nelder-Mead refinement on the flattened point set, maximizing
        the score (minimizing negative score).
        """
        from scipy.optimize import minimize
        x0 = points.flatten()

        def objective(x):
            p = x.reshape(n, 2)
            return -compute_score(p)

        try:
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            p = result.x.reshape(n, 2)
            sc = compute_score(p)
            if sc > compute_score(points):
                return p, sc
        except Exception:
            pass
        return points, compute_score(points)

    # Start with a regular polygon
    angles = 2 * np.pi * np.arange(n) / n
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])

    best_points = x0.copy()
    best_score = compute_score(best_points)

    rng = np.random.default_rng(seed=123)

    # Multiple restarts with different initializations
    for run in range(30):
        if time.time() > deadline - 10:
            break

        if run == 0:
            points = x0.copy()
        elif run == 1:
            points = x0 + rng.normal(0, 0.05, size=x0.shape)
        elif run == 2:
            points = x0 + rng.normal(0, 0.15, size=x0.shape)
        elif run == 3:
            # Slightly irregular polygon
            angles_pert = angles + rng.normal(0, 0.1, size=n)
            radii = 1.0 + rng.normal(0, 0.05, size=n)
            points = np.column_stack([radii * np.cos(angles_pert), radii * np.sin(angles_pert)])
        else:
            # Random points in unit disk
            r = rng.uniform(0, 1, size=n)
            theta = rng.uniform(0, 2*np.pi, size=n)
            points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        # Coordinate descent
        points, sc = coordinate_descent(points, max_iter=200, step_init=0.1)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()

        # Simulated annealing refinement
        if time.time() < deadline - 30:
            points, sc = simulated_annealing(points, t0=0.3, t_end=1e-5, alpha=0.997, max_iter=1500)
            if sc > best_score:
                best_score = sc
                best_points = points.copy()

        # Nelder-Mead final polish
        if time.time() < deadline - 15:
            points, sc = nelder_mead_refine(points, max_iter=100)
            if sc > best_score:
                best_score = sc
                best_points = points.copy()

    # Final high-precision polish
    if time.time() < deadline - 5:
        points, sc = coordinate_descent(best_points, max_iter=500, step_init=0.05)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()

    if time.time() < deadline - 2:
        points, sc = nelder_mead_refine(best_points, max_iter=300)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()

    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = x0

    return best_points

# EVOLVE-BLOCK-END
