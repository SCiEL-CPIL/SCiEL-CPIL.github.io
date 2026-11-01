import numpy as np
import scipy.spatial as spat
import scipy.optimize as opt
import time

N = 14

def calc_score(points):
    # points is (N, 2)
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume  # in 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area / hull_area


def _all_triangle_areas(points):
    """Compute all C(n,3) triangle areas as a flat array."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    return np.array(areas)


def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0


def _soft_min_area(points, beta):
    """
    Smooth surrogate: -1/beta * log(sum(exp(-beta * area_i)))
    As beta -> inf, this approaches the true minimum area.
    Returns (soft_min, gradient) with respect to the flattened point coordinates.
    """
    n = len(points)
    areas = []
    # Store indices and compute areas
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
                triples.append((i, j, k))

    areas = np.array(areas)
    if areas.max() < 1e-15:
        return 0.0, np.zeros(2 * n)

    # Log-sum-exp for numerical stability
    beta_val = beta
    neg_beta_areas = -beta_val * areas
    max_exp = np.max(neg_beta_areas)
    log_sum = max_exp + np.log(np.sum(np.exp(neg_beta_areas - max_exp)))
    soft_min = -log_sum / beta_val

    # Gradient: d(soft_min)/d(area_i) = exp(-beta * area_i) / sum(exp(-beta * area_j))
    weights = np.exp(neg_beta_areas - max_exp)
    weights /= np.sum(weights)  # normalized

    # Now compute gradient w.r.t. coordinates
    # soft_min = -1/beta * log(sum_i exp(-beta * A_i))
    # d(soft_min)/d(x) = -1/beta * sum_i [exp(-beta*A_i)/sum_j exp(-beta*A_j)] * d(-beta*A_i)/dx
    #                   = sum_i [w_i * dA_i/dx]
    # where w_i = exp(-beta*A_i) / sum_j exp(-beta*A_j)

    grad = np.zeros(2 * n)
    for idx, (i, j, k) in enumerate(triples):
        w = weights[idx]
        if w < 1e-15:
            continue
        pi = points[i]
        pj = points[j]
        pk = points[k]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        # A_i = 0.5 * |cross|, assume cross > 0 for gradient (handle sign)
        # dA/dpi = 0.5 * |-(v1[1]*d... )| -- use signed
        # For |cross|, d|cross|/d(pi) = -sign(cross) * (v2, -v1) direction... 
        # Simpler: A = 0.5*|cross|, dA/d(pi_x) = 0.5 * sign(cross) * (-(v1[1]*0 - v2[1]*1))... 
        # Let's use: cross = (pj_x-pi_x)(pk_y-pi_y) - (pk_x-pi_x)(pj_y-pi_y)
        # dcross/dpi_x = -(pk_y-pi_y) + (pj_y-pi_y) = pj_y - pk_y
        # dcross/dpi_y = (pk_x-pi_x) - (pj_x-pi_x) = pj_x - pk_x
        # dcross/dpj_x = (pk_y-pi_y)
        # dcross/dpj_y = -(pk_x-pi_x)
        # dcross/dpk_x = -(pj_y-pi_y)
        # dcross/dpk_y = (pj_x-pi_x)
        s = np.sign(cross)
        if s == 0:
            s = 1.0

        dA_dpi_x = 0.5 * s * (pj[1] - pk[1])
        dA_dpi_y = 0.5 * s * (pj[0] - pk[0])
        dA_dpj_x = 0.5 * s * (pk[1] - pi[1])
        dA_dpj_y = 0.5 * s * (-(pk[0] - pi[0]))
        dA_dpk_x = 0.5 * s * (-(pj[1] - pi[1]))
        dA_dpk_y = 0.5 * s * (pj[0] - pi[0])

        grad[2 * i + 0] += w * dA_dpi_x
        grad[2 * i + 1] += w * dA_dpi_y
        grad[2 * j + 0] += w * dA_dpj_x
        grad[2 * j + 1] += w * dA_dpj_y
        grad[2 * k + 0] += w * dA_dpk_x
        grad[2 * k + 1] += w * dA_dpk_y

    return soft_min, grad


def _smooth_neg_score(x, beta):
    """Negative smooth score for L-BFGS-B."""
    p = x.reshape((N, 2))
    soft_min, grad = _soft_min_area(p, beta)
    ha = _hull_area(p)
    if ha < 1e-10:
        return 1e6, np.zeros(2 * N)
    score = soft_min / ha
    # Gradient of -score = -grad(soft_min)/ha (ignoring derivative of hull_area for speed)
    g = -grad / ha
    return -score, g


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    best_points = None
    best_score = 0.0

    # Generate diverse initial guesses
    def _regular_polygon(n, offset=0.0, scale=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
        return scale * np.column_stack((np.cos(angles), np.sin(angles)))

    def _two_row_grid(n):
        # Arrange in 2 rows
        half = n // 2
        pts = []
        for i in range(half):
            pts.append((i / (half - 1) * 2 - 1, -0.5))
        for i in range(n - half):
            pts.append((i / max(1, n - half - 1) * 2 - 1, 0.5))
        return np.array(pts, dtype=float)

    def _hex_lattice(n):
        # Hexagonal packing subset
        pts = []
        # Generate hex lattice points
        for row in range(-3, 4):
            for col in range(-3, 4):
                x = col + 0.5 * (row % 2)
                y = row * np.sqrt(3) / 2
                pts.append((x, y))
        pts = np.array(pts, dtype=float)
        # Select n closest to center
        dists = np.sum(pts ** 2, axis=1)
        idx = np.argsort(dists)[:n]
        selected = pts[idx]
        # Center and scale
        selected -= selected.mean(axis=0)
        max_r = np.max(np.sqrt(np.sum(selected ** 2, axis=1)))
        if max_r > 0:
            selected /= max_r
        return selected

    def _random_points(n):
        return np.random.randn(n, 2) * 0.8

    initial_guesses = []
    # Regular 14-gon
    initial_guesses.append(_regular_polygon(N))
    # Rotated regular 14-gon
    initial_guesses.append(_regular_polygon(N, offset=np.pi / 14))
    # Two-row grid
    initial_guesses.append(_two_row_grid(N))
    # Hex lattice subset
    initial_guesses.append(_hex_lattice(N))
    # A few random
    for _ in range(3):
        initial_guesses.append(_random_points(N))

    # Multi-start L-BFGS-B with annealed beta
    n_guesses = len(initial_guesses)
    # Budget per guess: we'll do a few beta levels
    betas = [10.0, 50.0, 200.0, 1000.0]

    for gi, guess in enumerate(initial_guesses):
        if time.time() > deadline:
            break

        # Evaluate initial
        s = calc_score(guess)
        if s > best_score:
            best_score = s
            best_points = guess.copy()

        current = guess.copy()

        for bi, beta in enumerate(betas):
            if time.time() > deadline:
                break

            # L-BFGS-B optimization
            try:
                res = opt.minimize(
                    _smooth_neg_score,
                    current.flatten(),
                    args=(beta,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                             'maxls': 20, 'maxfun': 500}
                )
                if time.time() > deadline:
                    break
                opt_pts = res.x.reshape((N, 2))
                opt_s = calc_score(opt_pts)
                if opt_s > best_score:
                    best_score = opt_s
                    best_points = opt_pts.copy()
                    current = opt_pts.copy()
            except Exception:
                pass

    # Final polish: use Nelder-Mead on the best found with exact score
    if best_points is not None and time.time() < deadline - 20:
        try:
            def neg_exact(x):
                p = x.reshape((N, 2))
                return -calc_score(p)

            res = opt.minimize(
                neg_exact,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11,
                         'adaptive': True}
            )
            if time.time() < deadline:
                polish_pts = res.x.reshape((N, 2))
                polish_s = calc_score(polish_pts)
                if polish_s > best_score:
                    best_score = polish_s
                    best_points = polish_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _regular_polygon(N)

    return best_points
