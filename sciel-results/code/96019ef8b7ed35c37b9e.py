import numpy as np
from math import cos, sin, pi, sqrt
import time

def construct():
    start_time = time.time()
    deadline = 148.0  # safety margin before 150s kill

    n = 8

    def score_of(points):
        """Compute Delta / n^n for a set of n points.
        Delta = product over all ordered pairs (i != j) of distances,
        after scaling the set so that diameter = 2.
        """
        pts = np.asarray(points, dtype=float)
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        # Diameter
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        # Scale to diameter 2
        scale = 2.0 / d_max
        d_scaled = d * scale
        # Product over ordered pairs (i != j): that's n*(n-1) terms
        # Use log to avoid underflow
        mask = ~np.eye(n, dtype=bool)
        log_prod = np.sum(np.log(d_scaled[mask] + 1e-300))
        score = np.exp(log_prod) / (n ** n)
        return score

    # === Strategy: Multi-start with diverse initializations + Nelder-Mead + local polish ===

    # Generate diverse initializations
    def gen_regular_octagon():
        angles = np.linspace(0, 2 * pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))

    def gen_perturbed_octagon(sigma):
        base = gen_regular_octagon()
        return base + np.random.normal(0, sigma, size=(n, 2))

    def gen_random_in_circle():
        r = np.sqrt(np.random.rand(n))
        theta = np.random.rand(n) * 2 * pi
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    def gen_two_clusters():
        # Two clusters of 4 points, separated
        c1 = np.random.normal(0, 0.3, size=(4, 2))
        c2 = np.random.normal(2, 0.3, size=(4, 2))
        return np.vstack([c1, c2])

    def gen_square_corners():
        # 4 points near corners of a square, 4 near midpoints
        corners = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
        midpoints = np.array([[0,1.2],[-1.2,0],[0,-1.2],[1.2,0]])
        pts = np.vstack([corners + np.random.normal(0,0.1,size=(4,2)),
                          midpoints + np.random.normal(0,0.1,size=(4,2))])
        return pts

    def gen_alternating_radii():
        # Alternating inner and outer radii
        angles = np.linspace(0, 2*pi, n, endpoint=False)
        radii = np.where(np.arange(n) % 2 == 0, 1.0, 0.6)
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    np.random.seed(12345)

    initializations = [gen_regular_octagon()]
    for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        initializations.append(gen_perturbed_octagon(s))
    for _ in range(15):
        initializations.append(gen_random_in_circle())
    for _ in range(5):
        initializations.append(gen_two_clusters())
    for _ in range(3):
        initializations.append(gen_square_corners())
    for _ in range(3):
        initializations.append(gen_alternating_radii())

    best_score = -1.0
    best_points = gen_regular_octagon()

    def nelder_mead_optimize(pts0, max_iter=5000, tol=1e-10):
        """Custom Nelder-Mead on 16-dimensional parameter space."""
        # Flatten
        x = pts0.flatten().copy()
        # Initial simplex
        eps = 0.1
        simplex = [x.copy()]
        for i in range(16):
            xi = x.copy()
            xi[i] += eps
            simplex.append(xi)

        def f(x):
            pts = x.reshape(n, 2)
            return -score_of(pts)  # minimize negative score

        # Evaluate initial simplex
        fvals = [f(s) for s in simplex]

        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        for iteration in range(max_iter):
            if time.time() > deadline:
                break

            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if abs(fvals[0] - fvals[-1]) < tol:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])

        return np.argmin(fvals), simplex[np.argmin(fvals)]

    # Process each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline:
            break

        # First try Nelder-Mead
        try:
            _, best_x = nelder_mead_optimize(init, max_iter=3000, tol=1e-10)
            pts = best_x.reshape(n, 2)
            sc = score_of(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts
        except Exception:
            pass

        # Also try a few gradient-free random restarts around the best found
        if time.time() > deadline - 10:
            break

        # Local refinement with smaller Nelder-Mead steps
        try:
            _, best_x2 = nelder_mead_optimize(pts, max_iter=2000, tol=1e-11)
            pts2 = best_x2.reshape(n, 2)
            sc2 = score_of(pts2)
            if sc2 > best_score:
                best_score = sc2
                best_points = pts2
        except Exception:
            pass

    # Final polish: try to improve with a different approach - coordinate descent
    if time.time() < deadline - 5:
        # Coordinate ascent on log-product with diameter constraint handled by scaling
        def log_product(pts):
            d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            mask = ~np.eye(n, dtype=bool)
            return np.sum(np.log(d[mask] + 1e-300))

        def log_score(pts):
            d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            d_max = np.max(d)
            if d_max < 1e-12:
                return -np.inf
            scale = 2.0 / d_max
            d_scaled = d * scale
            mask = ~np.eye(n, dtype=bool)
            log_prod = np.sum(np.log(d_scaled[mask] + 1e-300))
            return log_prod - n * np.log(n)

        # Simulated annealing style perturbations
        pts = best_points.copy()
        cur_score = score_of(pts)
        T = 0.1
        for sa_iter in range(500):
            if time.time() > deadline - 2:
                break
            # Random perturbation
            p_idx = np.random.randint(n)
            coord = np.random.randint(2)
            delta = np.random.normal(0, T)
            pts_new = pts.copy()
            pts_new[p_idx, coord] += delta

            new_score = score_of(pts_new)
            if new_score > cur_score or np.random.rand() < np.exp((new_score - cur_score) / max(T, 1e-6)):
                pts = pts_new
                cur_score = new_score

            if cur_score > best_score:
                best_score = cur_score
                best_points = pts

            T *= 0.995  # Cool down

    return [[float(x), float(y)] for x, y in best_points]


if __name__ == "__main__":
    construct()
