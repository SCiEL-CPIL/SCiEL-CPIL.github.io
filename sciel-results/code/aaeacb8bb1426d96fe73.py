"""Hex seed + smooth soft-min gradient ascent + exact local polish for 200-circle packing."""
import time
import numpy as np

N = 200
HALF = 0.5
SEED = 1234567
EPS = 1e-12


def _radius_of(c):
    """Exact packing radius for centers c (N,2)."""
    w = HALF - np.abs(c)
    wall = float(np.min(w))
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = float(np.min(dist)) / 2.0
    return min(wall, pair)


def _softmin_sq(c, beta):
    """Smooth surrogate for (2r)^2 = min( min_pair d_ij^2, min_wall w_i^2 )."""
    d = c[:, None, :] - c[None, :, :]
    d2 = np.sum(d * d, axis=2)
    np.fill_diagonal(d2, np.inf)
    w = HALF - np.abs(c)
    w2 = np.sum(w * w, axis=1)
    a = np.concatenate([d2.ravel(), w2.ravel()])
    a = np.where(a > 1e-18, a, 1e-18)
    loga = np.log(a)
    m = loga.max()
    LSE = m + np.log(np.sum(np.exp(loga - m)))
    return 0.5 * LSE, a


def _softmin_sq_grad(c, beta):
    """Gradient of the soft-min surrogate (smooth min over squared distances/walls)."""
    d = c[:, None, :] - c[None, :, :]
    d2 = np.sum(d * d, axis=2)
    np.fill_diagonal(d2, np.inf)
    w = HALF - np.abs(c)
    w2 = np.sum(w * w, axis=1)
    a = np.concatenate([d2.ravel(), w2.ravel()])
    a = np.where(a > 1e-18, a, 1e-18)
    loga = np.log(a)
    m = loga.max()
    LSE = m + np.log(np.sum(np.exp(loga - m)))
    p = np.exp(loga - LSE)
    p = p / p.sum()
    d2f = d2.ravel()
    wf = w.ravel()
    dp = p[: d2f.size].reshape(N, N)
    dw = p[d2f.size:].reshape(N, 2)
    g = np.zeros((N, 2))
    for i in range(N):
        for j in range(N):
            g[i] -= 2.0 * dp[i, j] * d[i, j]
    for i in range(N):
        g[i] -= 2.0 * dw[i] * np.sign(c[i])
    return 0.5 * LSE, g


def _hexagon_seed(n):
    """Staggered hexagonal rows inside [-1/2,1/2]^2, return n centers."""
    pts = []
    r0 = 0.036
    for _ in range(400):
        h = 2 * r0
        v = r0 * np.sqrt(3)
        x0 = -HALF + r0
        y0 = -HALF + r0
        n_rows = int((1 - 2 * r0) / v) + 1
        pts = []
        for i in range(n_rows):
            y = y0 + i * v
            x_start = x0 + (h / 2 if i % 2 else 0.0)
            if x_start > HALF - r0:
                continue
            n_cols = int((HALF - r0 - x_start) / h) + 1
            for j in range(n_cols):
                x = x_start + j * h
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n:
            return pts[:n]
        r0 *= 0.997
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _ascent(c0, rng, deadline, beta0, step0):
    """Gradient ascent on the soft-min surrogate with backtracking line search."""
    c = c0.copy().astype(float)
    beta = beta0
    step = step0
    best = c.copy()
    best_r = _radius_of(c)
    best_soft = _softmin_sq(c, beta)[0]
    t0 = time.time()
    for it in range(200000):
        if time.time() - t0 > deadline:
            break
        soft, g = _softmin_sq_grad(c, beta)
        if soft > best_soft - 1e-9:
            r = _radius_of(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            best_soft = soft
        # adaptive temperature: anneal beta, occasional reheat to escape minima
        if it % 400 == 399:
            beta *= 0.82
        if it % 2000 == 1999:
            beta = min(beta / 0.6, beta0 * 4.0)
        gnorm = float(np.linalg.norm(g))
        if gnorm < 1e-10:
            break
        g = g / gnorm
        # backtracking line search
        moved = False
        s = step
        for _ in range(12):
            cand = c + s * g
            if np.any(cand < -HALF) or np.any(cand > HALF):
                s *= 0.5
                continue
            soft_c, _ = _softmin_sq(cand, beta)
            if soft_c > soft + 1e-12:
                c = cand
                soft = soft_c
                moved = True
                break
            s *= 0.5
        if moved:
            step = min(step * 1.05, 2e-2)
        else:
            step *= 0.6
    return best, best_r


def _polish(c, deadline):
    """Exact coordinate-wise line search polish (only active constraint pairs)."""
    c = c.copy().astype(float)
    best_r = _radius_of(c)
    t0 = time.time()
    step = 1e-4
    while time.time() - t0 < deadline:
        d = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        w = HALF - np.abs(c)
        wall = float(np.min(w))
        pair = float(np.min(dist)) / 2.0
        r = min(wall, pair)
        # active pairs: those within 2r + margin
        thr = 2.0 * r + 1e-4
        active = dist < thr
        improved = False
        for i in range(N):
            for j in np.where(active[i])[0]:
                if j <= i:
                    continue
                for k in range(2):
                    orig = c[i, k]
                    for sgn in (1.0, -1.0):
                        s = step
                        for _ in range(10):
                            cand = c.copy()
                            cand[i, k] = orig + sgn * s
                            if cand[i, k] < -HALF or cand[i, k] > HALF:
                                break
                            rr = _radius_of(cand)
                            if rr > r + 1e-13:
                                c = cand
                                r = rr
                                improved = True
                                break
                            s *= 0.5
            if time.time() - t0 > deadline:
                break
        if r > best_r:
            best_r = r
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
        else:
            step = min(step * 1.1, 1e-3)
    return c, best_r


def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    beta0 = 200.0
    step0 = 1e-3
    restarts = 0
    while time.time() - t_start < deadline - 12.0:
        restarts += 1
        scale = 0.02 / (1.0 + 0.08 * restarts)
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)
        d = c_start[:, None, :] - c_start[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-4:
            c_start = best_c.copy()
        rem = deadline - 12.0 - (time.time() - t_start)
        if rem < 3.0:
            break
        c_opt, r_opt = _ascent(c_start, rng, rem - 2.0, beta0, step0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    # final high-precision exact polish
    rem = deadline - (time.time() - t_start)
    if rem > 1.0:
        best_c, best_r = _polish(best_c, rem - 0.5)

    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return best_c


if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
