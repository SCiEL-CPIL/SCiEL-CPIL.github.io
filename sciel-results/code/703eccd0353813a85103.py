"""Combined strategy: Spectral/rotated initializations + smooth log-sum-exp surrogate
with analytic gradients, Nesterov active-set gradient ascent, and beta-annealed
L-BFGS-B polish. For 50 equal spheres in [-1/2,1/2]^3."""
import time
import numpy as np
import scipy.optimize as opt

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 142.0  # Safety margin before 150s kill


def _score(pts):
    """Exact radius: min(min_pair_dist/2, min_wall_dist)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3) or not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    min_wall = np.min(0.5 - np.abs(pts))
    return float(min(min_pair * 0.5, min_wall))


def _smooth_loss_and_grad(c_flat, beta):
    """
    Smooth surrogate: -soft_min of all radius-candidates (pairwise half-distances + wall distances).
    Returns (value, gradient) for minimization.
    """
    c = c_flat.reshape(N, 3)
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    # Pairwise half-distances
    diff = c[tri_i] - c[tri_j]
    d_pairs = np.linalg.norm(diff, axis=1) / 2.0

    # Wall distances: 6 per point
    d_x_pos = 0.5 - c[:, 0]
    d_x_neg = c[:, 0] + 0.5
    d_y_pos = 0.5 - c[:, 1]
    d_y_neg = c[:, 1] + 0.5
    d_z_pos = 0.5 - c[:, 2]
    d_z_neg = c[:, 2] + 0.5

    wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
    g_all = np.concatenate([d_pairs, wall_dists])

    # Soft min via log-sum-exp
    scaled = -beta * g_all
    m = np.max(scaled)
    lse = m + np.log(np.sum(np.exp(scaled - m)))
    val = -lse / beta

    # Weights for gradient
    weights = np.exp(scaled - lse)

    grad_c = np.zeros_like(c)

    # Pairwise gradient contribution
    safe_d = np.maximum(d_pairs, 1e-12)
    factors = weights[:n_pairs] / (4.0 * safe_d)
    factors = np.where(d_pairs > 1e-12, factors, 0.0)
    weighted_diffs = factors[:, None] * diff
    np.add.at(grad_c, tri_i, weighted_diffs)
    np.add.at(grad_c, tri_j, -weighted_diffs)

    # Wall gradient contribution
    w0 = n_pairs
    grad_c[:, 0] -= weights[w0: w0 + N]
    grad_c[:, 0] += weights[w0 + N: w0 + 2 * N]
    grad_c[:, 1] -= weights[w0 + 2 * N: w0 + 3 * N]
    grad_c[:, 1] += weights[w0 + 3 * N: w0 + 4 * N]
    grad_c[:, 2] -= weights[w0 + 4 * N: w0 + 5 * N]
    grad_c[:, 2] += weights[w0 + 5 * N: w0 + 6 * N]

    return -val, -grad_c.flatten()


def _nesterov_active_set(c_init, maxiter=400, lr_start=0.003, lr_end=0.0001):
    """
    Nesterov-accelerated gradient ascent on the exact min-distance objective.
    Uses active-set (k-NN) subgradient approximation for efficiency.
    """
    c = c_init.copy()
    v = np.zeros_like(c)
    best_c = c.copy()
    best_r = _score(c)

    for it in range(maxiter):
        if time.time() > DEADLINE_SECONDS - 3:
            break

        lr = lr_start * (lr_end / lr_start) ** (it / max(1, maxiter))

        # Compute active-set subgradient
        diff = c[:, None, :] - c[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)

        # Wall distances
        wall_d = np.min(0.5 - np.abs(c), axis=1)

        current_min = min(np.min(d) / 2.0, np.min(wall_d))
        threshold = max(1.2 * current_min, 0.01)

        # Pairwise active set
        active_pair_mask = (d < 2.0 * threshold)
        active_pair_mask = active_pair_mask & (np.triu(np.ones((N, N), dtype=bool), 1))
        ai, aj = np.where(active_pair_mask)

        grad = np.zeros_like(c)
        if len(ai) > 0:
            d_active = d[ai, aj]
            d_safe = np.maximum(d_active, 1e-12)
            unit_diffs = (c[ai] - c[aj]) / d_safe[:, None]
            np.add.at(grad, ai, unit_diffs)
            np.add.at(grad, aj, -unit_diffs)

        # Wall active set
        for i in range(N):
            for dim in range(3):
                if c[i, dim] > 0:
                    if 0.5 - c[i, dim] < threshold:
                        grad[i, dim] -= 1.0
                else:
                    if c[i, dim] + 0.5 < threshold:
                        grad[i, dim] += 1.0

        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm

        # Nesterov update
        v = 0.9 * v + lr * grad
        c_new = c + v
        c_new = np.clip(c_new, -0.499999, 0.499999)

        # Check improvement
        r_new = _score(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        c = c_new

    return best_c


def _lbgfsb_polish(pts, beta, iters_max, time_budget):
    """Run L-BFGS-B on the smooth surrogate, respecting time budget."""
    pts = np.asarray(pts, dtype=np.float64).copy()
    x0 = pts.ravel()

    def fun(x):
        val, _ = _smooth_loss_and_grad(x, beta)
        return val

    def jac(x):
        _, grad = _smooth_loss_and_grad(x, beta)
        return grad

    best_x = x0.copy()
    best_f = fun(x0)

    def cb(xk):
        nonlocal best_x, best_f
        f = fun(xk)
        if f < best_f:
            best_f = f
            best_x = xk.copy()
        if time.time() > time_budget:
            raise RuntimeError("timeout")

    try:
        res = opt.minimize(fun, x0, jac=jac, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (N * 3),
                           options={'maxiter': iters_max, 'maxfun': iters_max * 4,
                                    'ftol': 1e-12, 'gtol': 1e-10, 'callback': cb})
        if res.x is not None:
            f = fun(res.x)
            if f < best_f:
                best_f = f
                best_x = res.x.copy()
    except RuntimeError:
        pass
    except Exception:
        pass
    return best_x.reshape(N, 3)


def _anneal(pts, T0, T1, steps, rng):
    """Simulated annealing with adaptive kicks."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(steps):
        T = T0 * (T1 / T0) ** (i / steps)
        kick = 0.015 * (T / T0 + 0.05)
        n_move = max(1, N // 4)
        idx = rng.choice(N, size=n_move, replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(len(idx), 3))
        cand = np.clip(cand, -0.5 + 1e-12, 0.5 - 1e-12)
        cand_r = _score(cand)
        if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-9)):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 400 == 0 and time.time() > DEADLINE_SECONDS - 5:
            break
    return best, best_r


def _make_starts(rng):
    starts = []
    # 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    starts.append(grid[:N].copy())

    # 5x5x2 slab
    m2 = 5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    slab = np.array([(x, y, z) for z in ticks2[:2] for y in ticks2 for x in ticks2])
    if len(slab) >= N:
        starts.append(slab[:N].copy())

    # Spectral/rotated initializations: random orthogonal rotations of grid
    base_grid = starts[0].copy()
    for _ in range(3):
        # Generate random rotation matrix
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        rotated = base_grid @ Q.T
        # Scale to fit in cube
        max_abs = np.max(np.abs(rotated))
        if max_abs > 0:
            rotated = rotated * (0.45 / max_abs)
        rotated = np.clip(rotated, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(rotated)

    # BCC lattice
    positions = []
    a = 0.5
    spacing = (2 * a) / 3.0
    for iz in range(3):
        for iy in range(3):
            for ix in range(3):
                x = -a + (ix + 0.5) * spacing
                y = -a + (iy + 0.5) * spacing
                z = -a + (iz + 0.5) * spacing
                positions.append([x, y, z])
                # Add BCC center
                positions.append([x + spacing/2, y + spacing/2, z + spacing/2])
    bcc = np.array(positions[:N])
    bcc = np.clip(bcc, -0.5 + 1e-6, 0.5 - 1e-6)
    starts.append(bcc)

    # Random jittered grids
    for _ in range(3):
        s = starts[0] + rng.normal(0.0, 0.03, size=starts[0].shape)
        s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(s)

    return starts


def construct():
    start_time = time.time()
    rng = np.random.default_rng(777)
    starts = _make_starts(rng)

    best = starts[0].copy()
    best_r = _score(best)

    # Phase 1: anneal each start, then polish with L-BFGS-B
    for s in starts:
        if time.time() > DEADLINE_SECONDS - 35:
            break
        a, ar = _anneal(s, T0=0.04, T1=0.0005, steps=2500, rng=rng)
        if ar > best_r:
            best_r, best = ar, a

        # Nesterov active-set gradient ascent
        if time.time() > DEADLINE_SECONDS - 30:
            break
        n = _nesterov_active_set(a, maxiter=300, lr_start=0.003, lr_end=0.0001)
        nr = _score(n)
        if nr > best_r:
            best_r, best = nr, n

        # Polish with L-BFGS-B using a moderate beta
        if time.time() > DEADLINE_SECONDS - 25:
            break
        p = _lbgfsb_polish(n, beta=200.0, iters_max=200, time_budget=DEADLINE_SECONDS - 20)
        p = np.clip(p, -0.5 + 1e-12, 0.5 - 1e-12)
        pr = _score(p)
        if pr > best_r:
            best_r, best = pr, p

    # Phase 2: high-beta polish on the current best
    if time.time() < DEADLINE_SECONDS - 10:
        for beta in [400.0, 800.0, 1500.0]:
            if time.time() > DEADLINE_SECONDS - 8:
                break
            p = _lbgfsb_polish(best, beta=beta, iters_max=300, time_budget=DEADLINE_SECONDS - 5)
            p = np.clip(p, -0.5 + 1e-12, 0.5 - 1e-12)
            pr = _score(p)
            if pr > best_r:
                best_r, best = pr, p

    # Phase 3: final Nesterov polish
    if time.time() < DEADLINE_SECONDS - 5:
        n = _nesterov_active_set(best, maxiter=200, lr_start=0.001, lr_end=0.00005)
        nr = _score(n)
        if nr > best_r:
            best_r, best = nr, n

    # Final safety: ensure strictly inside the cube
    eps = 1e-12
    best = np.clip(best, -0.5 + eps, 0.5 - eps)

    return best
