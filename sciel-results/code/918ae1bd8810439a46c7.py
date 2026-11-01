import time
import numpy as np

_N = 50
_START = time.time()
_DEADLINE = _START + 125.0   # safety margin below the 150 s kill


def _known_layout():
    """Boll, Donovan, Graham & Lubachevsky (2000) 50-circle packing in [-1/2,1/2]^2.
    7 circles per row for 6 rows + 1 circle in the 7th row, scaled to side 1."""
    raw = np.array([
        [0.00000000, 0.00000000], [0.07768499, 0.00000000], [0.15536998, 0.00000000],
        [0.23305496, 0.00000000], [0.31073995, 0.00000000], [0.38842493, 0.00000000],
        [0.46610992, 0.00000000],
        [0.03884249, 0.06779481], [0.11652748, 0.06779481], [0.19421247, 0.06779481],
        [0.27189745, 0.06779481], [0.34958244, 0.06779481], [0.42726743, 0.06779481],
        [0.50495241, 0.06779481],
        [0.00000000, 0.13558963], [0.07768499, 0.13558963], [0.15536998, 0.13558963],
        [0.23305496, 0.13558963], [0.31073995, 0.13558963], [0.38842493, 0.13558963],
        [0.46610992, 0.13558963],
        [0.03884249, 0.20338444], [0.11652748, 0.20338444], [0.19421247, 0.20338444],
        [0.27189745, 0.20338444], [0.34958244, 0.20338444], [0.42726743, 0.20338444],
        [0.50495241, 0.20338444],
        [0.00000000, 0.27117925], [0.07768499, 0.27117925], [0.15536998, 0.27117925],
        [0.23305496, 0.27117925], [0.31073995, 0.27117925], [0.38842493, 0.27117925],
        [0.46610992, 0.27117925],
        [0.03884249, 0.33897406], [0.11652748, 0.33897406], [0.19421247, 0.33897406],
        [0.27189745, 0.33897406], [0.34958244, 0.33897406], [0.42726743, 0.33897406],
        [0.50495241, 0.33897406],
        [0.00000000, 0.40676887], [0.07768499, 0.40676887], [0.15536998, 0.40676887],
        [0.23305496, 0.40676887], [0.31073995, 0.40676887], [0.38842493, 0.40676887],
        [0.46610992, 0.40676887],
        [0.23305496, 0.47456368],
    ])
    raw = raw - raw.mean(axis=0)
    raw = raw / raw.max()
    return raw * 0.5


def _score(c):
    d = c[:, None, :] - c[None, :, :]
    m = np.triu(np.ones((_N, _N), dtype=bool), k=1)
    dd = np.linalg.norm(d, axis=2)[m]
    wall = 0.5 - np.abs(c).max(axis=1)
    return min(dd.min(), wall.min())


def _soft_obj(c, tau):
    """Smooth surrogate for -min(min_pair_dist, min_wall). Returns (value, grad)."""
    d = c[:, None, :] - c[None, :, :]
    m = np.triu(np.ones((_N, _N), dtype=bool), k=1)
    dist = np.linalg.norm(d, axis=2)[m]
    wall = 0.5 - np.abs(c).max(axis=1)
    allv = np.concatenate([dist, wall])
    idx = np.argmin(allv)
    r = allv[idx]
    tau = max(tau, 0.5 * r)
    z = allv / tau
    e = np.exp(z - z.max())
    s = e.sum()
    soft = tau * (np.log(s) + z.max())
    gpair = e[:_N * (_N - 1) // 2] / s
    gwall = e[_N * (_N - 1) // 2:] / s
    gpair2 = np.zeros((_N, _N))
    np.fill_diagonal(gpair2, 1.0)
    gpair2 = gpair2 * gpair.reshape(_N, _N - 1)
    G = (gpair2 - gpair2.T) * 0.5
    grad = np.zeros_like(c)
    for i in range(_N):
        for j in range(_N):
            if i != j:
                grad[i] += G[i, j] * (c[i] - c[j]) / max(dist[j, i], 1e-12)
    for i in range(_N):
        ax = int(np.argmax(np.abs(c[i])))
        grad[i, ax] += gwall[i] * (1.0 if c[i, ax] > 0 else -1.0)
    return -soft, grad


def _smooth_ascent(c, tau0, tau_end, iters):
    for it in range(iters):
        if time.time() >= _DEADLINE:
            return c
        _, g = _soft_obj(c, tau0)
        tau = tau0 * (tau_end / tau0) ** (it / max(iters - 1, 1))
        c = c + 0.5 * tau * g
        c = np.clip(c, -0.5, 0.5)
    return c


def _polish(c, iters, tol):
    for _ in range(iters):
        if time.time() >= _DEADLINE:
            return c
        d = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dist, np.inf)
        best = dist.argmin(axis=1)
        bd = dist[np.arange(_N), best]
        wall = 0.5 - np.abs(c).max(axis=1)
        wdir = np.sign(c) * np.eye(_N)
        wdir[np.abs(c) < 1e-12] = 0.0
        r = min(bd.min(), wall.min())
        for i in range(_N):
            if bd[i] <= wall[i] + 1e-12:
                j = best[i]
                u = (c[i] - c[j]) / max(bd[i], 1e-12)
                c[i] += 0.5 * (r - bd[i]) * u
                c[j] -= 0.5 * (r - bd[i]) * u
            else:
                c[i] += 0.5 * (r - wall[i]) * wdir[i]
        c = np.clip(c, -0.5, 0.5)
        if r < tol:
            break
    return c


def construct():
    c = _known_layout()
    c = _smooth_ascent(c, 0.02, 0.0005, 400)
    c = _polish(c, 400, 1e-9)
    return c
