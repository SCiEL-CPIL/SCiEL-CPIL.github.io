"""Hexagonal-lattice init + soft-min (log-sum-exp) force relaxation + exact local polish."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
HALF = 0.5
DEADLINE = 140.0  # seconds; evaluator kills at 150s


def _score(pts):
    """True objective: min(pair_dist/2, min wall margin)."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    pair = d.min() / 2.0
    wall = HALF - np.abs(pts).max()
    return min(pair, wall)


def _softmin_grad(pts, t):
    """Gradient of soft-min pair distance s = -ln(sum_{i<j} exp(-d_ij/t)).
    Returns (s, grad) where grad has same shape as pts.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, 0.0)
    # mask upper triangle
    iu = np.triu_indices(N, k=1)
    du = d[iu]
    w = np.exp(-du / t)
    w /= w.sum()
    # gradient w.r.t. pts: for pair (i,j), d/dx_i = w * (x_i - x_j)/d_ij
    grad = np.zeros_like(pts)
    iu_i, iu_j = iu
    for k in range(len(du)):
        i, j = iu_i[k], iu_j[k]
        if du[k] < 1e-12:
            continue
        vec = (pts[i] - pts[j]) / du[k]
        grad[i] += w[k] * vec
        grad[j] -= w[k] * vec
    s = -np.log(w.sum() * t)  # not used directly; grad is what matters
    return s, grad


def _wall_grad(pts):
    """Gradient of smooth wall penalty: encourage staying inside.
    penalty = sum over points of softplus(HALF - |x|) + softplus(HALF - |y|)
    We return negative gradient (i.e., direction to increase margin).
    """
    # For each coordinate, margin = HALF - |coord|
    # We want to push points inward when margin is small.
    # Use: force = -tanh((|coord| - HALF + eps) / t_wall) * sign(coord)
    # Simpler: gradient of -softplus(|coord| - HALF + delta)
    t_wall = 0.02
    delta = 0.01
    grad = np.zeros_like(pts)
    for axis in range(2):
        c = pts[:, axis]
        margin = HALF - np.abs(c)
        # push inward: force proportional to -margin when margin small
        # use smooth: f = -tanh((HALF - margin) / t_wall) * sign(c)
        # = -tanh((|c|) / t_wall) * sign(c) when near boundary
        # Actually let's just use: if |c| > HALF - t_wall, push back
        overshoot = np.abs(c) - (HALF - t_wall)
        mask = overshoot > 0
        force = np.zeros_like(c)
        force[mask] = -np.tanh(overshoot[mask] / t_wall) * np.sign(c[mask])
        grad[:, axis] += force
    return grad


def _hex_init():
    """Initialize 200 points on a hexagonal lattice clipped to the square."""
    # Hex lattice: spacing s in x, s*sqrt(3)/2 in y, odd rows offset by s/2
    # We want ~200 points in [-0.5, 0.5]^2
    # Try spacing ~ 1/sqrt(200*2/sqrt(3)) ≈ 1/sqrt(230.9) ≈ 0.0656
    # But we'll generate a grid and pick the best 200.
    s = 0.065  # hex spacing
    dx = s
    dy = s * np.sqrt(3) / 2
    # generate points
    pts = []
    nx = int(np.ceil(1.0 / dx)) + 2
    ny = int(np.ceil(1.0 / dy)) + 2
    for j in range(-ny, ny + 1):
        y = j * dy
        if y > HALF or y < -HALF:
            continue
        offset = (dx / 2) if (j % 2 != 0) else 0.0
        for i in range(-nx, nx + 1):
            x = i * dx + offset
            if x > HALF or x < -HALF:
                continue
            pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        # pick the N closest to origin for symmetry
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        return pts[idx]
    else:
        # fallback: pad with small random jitter
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-HALF, HALF, 2)])
        return pts[:N]


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # --- Phase 1: Soft-min gradient ascent with adaptive temperature ---
    pts = _hex_init().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(best_pts)

    t = 0.02  # temperature for soft-min
    lr = 0.001
    max_iter = 5000

    for it in range(max_iter):
        if time.time() - t0 > DEADLINE:
            break

        s, grad = _softmin_grad(pts, t)
        wall_g = _wall_grad(pts)
        total_grad = grad + 2.0 * wall_g

        # Adaptive step: cap displacement
        norm_g = np.linalg.norm(total_grad)
        if norm_g > 0:
            step = total_grad / norm_g * min(lr, 0.005)
        else:
            break

        new_pts = pts + step
        # Clamp to square
        new_pts = np.clip(new_pts, -HALF + 1e-9, HALF - 1e-9)

        # Check no pair is too close (hard constraint)
        d = cdist(new_pts, new_pts)
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.004:
            # reject this step, shrink lr
            lr *= 0.7
            continue

        pts = new_pts
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()

        # Adaptive temperature: decrease as we converge
        t = max(0.005, t * 0.999)
        # Slightly increase lr if improving
        if sc > best_score - 1e-8:
            lr = min(lr * 1.01, 0.005)

    # --- Phase 2: Exact local polish via Nelder-Mead on min-distance ---
    if time.time() - t0 < DEADLINE - 20:
        from scipy.optimize import minimize

        def neg_obj(x):
            p = x.reshape(N, 2)
            p = np.clip(p, -HALF + 1e-9, HALF - 1e-9)
            return -_score(p)

        # Try a few perturbed starts
        for trial in range(3):
            if time.time() - t0 > DEADLINE - 5:
                break
            if trial == 0:
                x0 = best_pts.copy()
            else:
                x0 = best_pts + rng.normal(0, 0.002, (N, 2))
                x0 = np.clip(x0, -HALF + 1e-9, HALF - 1e-9)

            try:
                res = minimize(neg_obj, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12,
                                        'adaptive': True})
                p = res.x.reshape(N, 2)
                p = np.clip(p, -HALF + 1e-9, HALF - 1e-9)
                sc = _score(p)
                if sc > best_score:
                    best_score = sc
                    best_pts = p.copy()
            except Exception:
                pass

    return best_pts
