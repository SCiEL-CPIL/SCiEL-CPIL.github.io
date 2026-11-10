import numpy as np
import scipy.optimize
import scipy.spatial.distance
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit

    # ---- Smooth surrogate objective and analytic gradient ----
    # We maximize (d_min / d_max)^2 via a soft-min/soft-max surrogate.
    # For a vector of distances d, soft_min = -t*log(mean(exp(-d/t))) and
    # soft_max = t*log(mean(exp(d/t))).  The ratio r = soft_min/soft_max is
    # smooth in the coordinates, and dr/dx_i = (1/r)( (dr_min/dx_i)/r_min -
    # (dr_max/dx_i)/r_max ).  We minimize -r^2.

    def _surrogate_and_grad(x_flat, t):
        pts = x_flat.reshape(n, 2)
        # Pairwise distances, vectorized
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        # Upper-triangular mask (exclude diagonal)
        iu = np.triu_indices(n, k=1)
        dvec = d[iu]
        m = len(dvec)
        if m == 0:
            return 0.0, np.zeros_like(x_flat)

        # Soft min / soft max (log-sum-exp, stabilized)
        dmin0 = np.min(dvec)
        dmax0 = np.max(dvec)
        # log(mean(exp(-d/t))) = -dmin0/t + log(mean(exp(-(d-dmin0)/t)))
        a = np.exp(-(dvec - dmin0) / t)
        log_mean_neg = -dmin0 / t + np.log(np.mean(a))
        r_min = -t * log_mean_neg
        # log(mean(exp(d/t))) = dmax0/t + log(mean(exp((d-dmax0)/t)))
        b = np.exp((dvec - dmax0) / t)
        log_mean_pos = dmax0 / t + np.log(np.mean(b))
        r_max = t * log_mean_pos

        if r_max <= 1e-12:
            return 0.0, np.zeros_like(x_flat)
        ratio = r_min / r_max
        val = -(ratio * ratio)

        # Weights for the soft min: w_min_j = exp(-(d_j - dmin0)/t) / sum
        w_min = a / np.sum(a)
        # Weights for the soft max: w_max_j = exp((d_j - dmax0)/t) / sum
        w_max = b / np.sum(b)

        # d r_min / d d_j = -w_min_j
        # d r_max / d d_j =  w_max_j
        # d r / d d_j = (1/ratio) * ( -w_min_j / r_min - w_max_j / r_max )
        # (derivative of (r_min/r_max) wrt d_j = (dr_min/d_j * r_max - r_min * dr_max/d_j)/r_max^2
        #  = ( (-w_min_j) * r_max - r_min * (w_max_j) ) / r_max^2
        #  = ( -w_min_j / r_max - r_min * w_max_j / r_max^2 )
        #  = (1/ratio) * ( -w_min_j / r_min - w_max_j / r_max )   [since ratio=r_min/r_max]
        # Let's verify: (1/ratio)(-w_min/r_min - w_max/r_max)
        #  = (r_max/r_min)(-w_min/r_min - w_max/r_max)
        #  = -w_min*r_max/r_min^2 - w_max/r_min
        # Direct: (-w_min*r_max - r_min*w_max)/r_max^2 = -w_min/r_max - r_min*w_max/r_max^2
        #  = -w_min/r_max - (r_min/r_max)*w_max/r_max = -w_min/r_max - ratio*w_max/r_max
        #  = (1/r_max)(-w_min - ratio*w_max)
        # The (1/ratio) form: (r_max/r_min)(-w_min/r_min - w_max/r_max)
        #  = -w_min*r_max/r_min^2 - w_max/r_min
        # These differ. Use the direct form to be safe.
        # d ratio / d d_j = (-w_min_j * r_max - r_min * w_max_j) / r_max**2

        d_ratio_dd = (-w_min * r_max - r_min * w_max) / (r_max * r_max)

        # d val / d d_j = d(-ratio^2)/d d_j = -2*ratio * d_ratio_dd
        d_val_dd = -2.0 * ratio * d_ratio_dd

        # Now propagate to coordinates via d d_j / d x.
        # d_j is the distance for pair (i,k). Build index map.
        # d[i,k] = ||pts[i]-pts[k]|| ; for i<k.
        # d d_j / d pts[i] = (pts[i]-pts[k])/d_j
        # d d_j / d pts[k] = (pts[k]-pts[i])/d_j
        grad = np.zeros_like(x_flat)
        # Map each upper-triangular index j to (i,k)
        # We have iu[0], iu[1] arrays.
        ii = iu[0]
        kk = iu[1]
        # d_j values
        dj = dvec
        # Avoid divide by zero
        safe_dj = np.where(dj > 1e-12, dj, 1e-12)
        # unit vectors
        # For each pair, contribution to grad of pts[i] and pts[k]
        # d_val_dd[j] * (pts[i]-pts[k])/dj  added to grad[i], and negative to grad[k]
        # Vectorize: accumulate
        # delta[j] = d_val_dd[j] / safe_dj[j]
        delta = d_val_dd / safe_dj
        # pts[i]-pts[k]
        # We'll scatter-add
        # Build arrays of indices and vectors
        # For point i: add delta[j]*(pts[i]-pts[k])
        # For point k: add delta[j]*(pts[k]-pts[i]) = -delta[j]*(pts[i]-pts[k])
        # Use np.add.at
        diff_vec = pts[ii] - pts[kk]  # shape (m,2)
        contrib = delta[:, None] * diff_vec  # shape (m,2)
        np.add.at(grad.reshape(n, 2), ii, contrib)
        np.add.at(grad.reshape(n, 2), kk, -contrib)

        return val, grad

    # ---- Initializations ----
    def get_hex_points(n_points):
        # 4x4 hexagonal grid
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 1.0 + (i % 2) * 0.5
                y = i * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        center = points.mean(axis=0)
        points = points - center
        max_norm = np.max(np.linalg.norm(points, axis=1))
        if max_norm > 0:
            points = points / max_norm
        return points

    def get_circle_points(n_points):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def get_square_points(n_points):
        points = []
        for i in range(4):
            for j in range(4):
                points.append([j * 1.0, i * 1.0])
        points = np.array(points)
        center = points.mean(axis=0)
        points = points - center
        max_norm = np.max(np.linalg.norm(points, axis=1))
        if max_norm > 0:
            points = points / max_norm
        return points

    def get_concentric(n_points):
        # 8 on outer, 8 on inner
        angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles2 = angles1 + np.pi / 8
        outer = np.column_stack([np.cos(angles1), np.sin(angles1)])
        inner = np.column_stack([np.cos(angles2), np.sin(angles2)]) * 0.5
        return np.vstack([outer, inner])

    def get_random(seed, n_points):
        rng = np.random.default_rng(seed)
        r = rng.random(n_points) ** 0.5
        theta = rng.random(n_points) * 2 * np.pi
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    base_inits = [
        get_hex_points(n),
        get_circle_points(n),
        get_square_points(n),
        get_concentric(n),
        get_random(1, n),
        get_random(2, n),
        get_random(3, n),
    ]

    # Add jittered hexagonal variants
    rng = np.random.default_rng(7)
    for _ in range(4):
        jitter = rng.normal(0, 0.1, get_hex_points(n).shape)
        base_inits.append(get_hex_points(n) + jitter)

    # ---- Score function (true metric) ----
    def true_score(points):
        dists = scipy.spatial.distance.pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    best_score = -np.inf
    best_points = None

    def consider(pts):
        nonlocal best_score, best_points
        s = true_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---- Optimization loop: L-BFGS-B with analytic gradient on smooth surrogate ----
    for init in base_inits:
        if time.time() > deadline - 30:
            break
        for t in [0.15, 0.08, 0.04]:
            if time.time() > deadline - 15:
                break
            x0 = init.flatten()
            try:
                res = scipy.optimize.minimize(
                    lambda x: _surrogate_and_grad(x, t)[0],
                    x0,
                    jac=lambda x: _surrogate_and_grad(x, t)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50},
                )
                if res.fun < best_score or best_points is None:
                    pass  # surrogate value not directly comparable; use true score
                consider(res.x.reshape(n, 2))
            except Exception:
                pass

    # ---- Final coordinate-descent polish on the true nonsmooth objective ----
    # Move one point at a time, small steps, accept if true score improves.
    if best_points is not None:
        pts = best_points.copy()
        cur = true_score(pts)
        # Normalize scale (metric is scale-invariant, but keep dmax ~ 1 for stability)
        dists = scipy.spatial.distance.pdist(pts)
        dmax = np.max(dists)
        if dmax > 0:
            pts = pts / dmax
            cur = true_score(pts)

        rng2 = np.random.default_rng(99)
        step = 0.05
        no_improve = 0
        while time.time() < deadline - 5 and no_improve < 2000:
            idx = rng2.integers(n)
            # Try a few candidate directions
            improved = False
            for _ in range(8):
                delta = rng2.normal(0, step, 2)
                new_pts = pts.copy()
                new_pts[idx] += delta
                # Renormalize
                d2 = scipy.spatial.distance.pdist(new_pts)
                dm = np.max(d2)
                if dm > 0:
                    new_pts = new_pts / dm
                ns = true_score(new_pts)
                if ns > cur + 1e-12:
                    pts = new_pts
                    cur = ns
                    improved = True
                    break
            if improved:
                no_improve = 0
                # Occasionally shrink step
                if no_improve == 0 and step > 0.001:
                    step *= 0.98
            else:
                no_improve += 1
                step *= 0.995
                if step < 1e-5:
                    step = 1e-5

        consider(pts)
        best_points = pts

    if best_points is None:
        best_points = get_hex_points(n)

    # Ensure shape (16,2) and finite
    best_points = np.asarray(best_points, dtype=float).reshape(16, 2)
    if not np.all(np.isfinite(best_points)):
        best_points = get_hex_points(n)

    return best_points
