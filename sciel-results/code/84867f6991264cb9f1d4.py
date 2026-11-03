"""Combined approach: topology-aware local search + gradient polish + graph-based violation minimization."""
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

N = 50
KNOWN_R = 0.071377103865


def _true_radius(centers):
    """Compute the true maximum packing radius for given centers in [-0.5,0.5]^2."""
    centers = np.asarray(centers)
    n = centers.shape[0]
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _local_search(centers, max_iter=3000, time_limit=15.0, step_init=0.015):
    """
    Hill-climbing local search: move each circle away from its nearest constraint.
    Returns (centers, best_radius).
    """
    centers = centers.copy()
    best_r = _true_radius(centers)
    step_size = step_init
    no_improve = 0
    start = time.time()

    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        indices = np.random.permutation(N)
        for idx in indices:
            orig = centers[idx].copy()
            # Current local min for this circle
            wall_d = 0.5 - np.abs(orig)
            min_wall = np.min(wall_d)
            diffs = centers - orig
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            dists[idx] = np.inf
            min_pair = np.min(dists) / 2.0
            cur_local = min(min_wall, min_pair)

            # Determine push direction: away from closest constraint
            closest_idx = np.argmin(dists)
            if dists[closest_idx] / 2.0 < min_wall:
                direction = orig - centers[closest_idx]
            else:
                # Move toward center (away from nearest wall)
                direction = -orig
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                direction = np.random.normal(0, 1, 2)
                norm = np.linalg.norm(direction)
            direction /= norm

            # Try several perturbations
            best_local = cur_local
            best_new = orig.copy()
            for _ in range(30):
                noise = np.random.normal(0, 1, 2)
                move = direction * step_size + noise * step_size * 0.4
                new_c = np.clip(orig + move, -0.4999, 0.4999)
                # Quick local min
                nw = 0.5 - np.abs(new_c)
                nmin_wall = np.min(nw)
                nd = centers - new_c
                ndists = np.sqrt(np.sum(nd ** 2, axis=1))
                ndists[idx] = np.inf
                nmin_pair = np.min(ndists) / 2.0
                nlocal = min(nmin_wall, nmin_pair)
                if nlocal > best_local + 1e-12:
                    best_local = nlocal
                    best_new = new_c
            if best_local > cur_local + 1e-12:
                centers[idx] = best_new
                improved = True
        if not improved:
            no_improve += 1
            step_size *= 0.9
            if no_improve > 40:
                break
        else:
            no_improve = 0
        r = _true_radius(centers)
        if r > best_r:
            best_r = r
    return centers, best_r


def _softmin_objective(pts_flat, k=500.0):
    """
    Smooth surrogate: -soft_min(min_pairwise_dist/2, min_wall_dist).
    """
    pts = pts_flat.reshape(-1, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    wall_dists = 0.5 - np.abs(pts)
    all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
    all_terms = all_terms[np.isfinite(all_terms)]
    if len(all_terms) == 0:
        return 0.0
    m = np.min(all_terms)
    shifted = all_terms - m
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 50.0 / k)
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min = m - (1.0 / k) * log_sum
    return -soft_min


def _gradient_polish(centers, time_limit=20.0):
    """Polish with L-BFGS-B using soft-min surrogate."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    start = time.time()
    best_c = centers.copy()
    best_r = _true_radius(centers)

    for k in [200.0, 500.0, 1000.0]:
        if time.time() - start > time_limit:
            break
        try:
            result = minimize(
                _softmin_objective,
                best_c.flatten(),
                args=(k,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_c = result.x.reshape(N, 2)
            new_c = np.clip(new_c, -0.499999, 0.499999)
            r = _true_radius(new_c)
            if r > best_r:
                best_r = r
                best_c = new_c
        except Exception:
            pass
    return best_c, best_r


def _graph_violation_minimize(centers, time_limit=15.0):
    """
    Graph-based violation minimization using projected gradient descent.
    Edges connect pairs closer than a dynamic threshold.
    Minimize sum of squared violations of minimum separation constraint.
    """
    centers = centers.copy()
    best_r = _true_radius(centers)
    start = time.time()
    lr = 0.001
    threshold_mult = 1.5  # dynamic threshold multiplier
    
    for iteration in range(5000):
        if time.time() - start > time_limit:
            break
        
        # Compute current radius
        r = _true_radius(centers)
        if r > best_r:
            best_r = r
        
        # Dynamic threshold
        threshold = threshold_mult * r
        
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find violating pairs (distance < 2*r)
        # Also consider wall constraints
        wall_dists = 0.5 - np.abs(centers)
        
        # Build gradient: push apart violating pairs, push inward from walls
        grad = np.zeros_like(centers)
        
        # Pair violations
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 2 * r and d > 1e-9:
                    # Violation: need distance >= 2*r
                    # Push apart
                    direction = (centers[i] - centers[j]) / d
                    violation = 2 * r - d
                    grad[i] += direction * violation
                    grad[j] -= direction * violation
        
        # Wall violations
        for i in range(N):
            for k in range(2):
                wall_d = wall_dists[i, k]
                if wall_d < r:
                    # Push toward center
                    grad[i, k] -= (r - wall_d) * np.sign(centers[i, k])
        
        # Projected gradient step
        centers = centers - lr * grad
        centers = np.clip(centers, -0.5, 0.5)
        
        # Adapt learning rate
        if iteration % 100 == 0:
            lr *= 0.95
    
    return centers, best_r


def _hex_init(seed=42):
    """Hexagonal grid init."""
    np.random.seed(seed)
    s = 0.14
    pts = []
    for i, x in enumerate(np.arange(-0.8, 0.8, s)):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in np.arange(-0.8, 0.8, s):
            pts.append((x + offset, y))
    pts = np.array(pts)
    mask = (np.abs(pts[:, 0]) <= 0.49) & (np.abs(pts[:, 1]) <= 0.49)
    pts = pts[mask]
    if len(pts) < N:
        return np.random.uniform(-0.45, 0.45, (N, 2))
    idx = np.random.choice(len(pts), N, replace=False)
    return pts[idx].copy()


def _square_grid_init(seed=123):
    """Square grid init."""
    np.random.seed(seed)
    gx = np.linspace(-0.45, 0.45, 7)
    gy = np.linspace(-0.45, 0.45, 7)
    pts = np.array([[x, y] for x in gx for y in gy])
    if len(pts) < N:
        return np.random.uniform(-0.45, 0.45, (N, 2))
    idx = np.random.choice(len(pts), N, replace=False)
    return pts[idx].copy()


def _random_init(seed=999):
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 2))


def _jittered_hex(seed=43):
    base = _hex_init(seed)
    np.random.seed(seed)
    jittered = base + np.random.normal(0, 0.02, (N, 2))
    return np.clip(jittered, -0.49, 0.49)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    inits = [
        _hex_init(42),
        _square_grid_init(123),
        _random_init(999),
        _jittered_hex(43),
        _random_init(1000),
        _hex_init(777),
    ]

    best_pts = inits[0].copy()
    best_r = _true_radius(best_pts)

    # Phase 1: Local search on each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        remaining = deadline - time.time()
        ls_time = min(15.0, remaining * 0.6)
        if ls_time < 2.0:
            break
        ls_c, ls_r = _local_search(init, max_iter=3000, time_limit=ls_time)
        if ls_r > best_r:
            best_r = ls_r
            best_pts = ls_c.copy()

    # Phase 2: Graph-based violation minimization
    if time.time() < deadline - 20.0:
        gv_c, gv_r = _graph_violation_minimize(best_pts, time_limit=15.0)
        if gv_r > best_r:
            best_r = gv_r
            best_pts = gv_c.copy()

    # Phase 3: Gradient polish on best
    if time.time() < deadline - 5.0:
        pol_c, pol_r = _gradient_polish(best_pts, time_limit=20.0)
        if pol_r > best_r:
            best_r = pol_r
            best_pts = pol_c.copy()

    # Phase 4: If time remains, try one more round of local search from polished result
    if time.time() < deadline - 10.0:
        ls_c2, ls_r2 = _local_search(best_pts, max_iter=2000, time_limit=10.0)
        if ls_r2 > best_r:
            best_r = ls_r2
            best_pts = ls_c2.copy()

    # Final check: ensure coordinates are within bounds
    best_pts = np.clip(best_pts, -0.5, 0.5)

    return best_pts
