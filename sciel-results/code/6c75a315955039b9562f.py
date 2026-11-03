"""50 equal spheres in a unit cube [-1/2,1/2]^3. Maximize common radius.
Direct epigraph + active-set refinement + multi-start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_TRI_I, _TRI_J = np.triu_indices(N, k=1)
_NP = len(_TRI_I)          # 1225
_NC = _NP + 6 * N          # 2025 constraints
_BOUNDS = [(-0.49999, 0.49999)] * (N * 3)


def _radius(c):
    d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    mp = np.min(d) * 0.5
    dw = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
    return min(mp, float(np.min(dw)))


def _grads(c):
    """Return (g, J) where g[i]=constraint value, J[i]=grad w.r.t. c (shape 3N)."""
    diff = c[_TRI_I] - c[_TRI_J]
    d2 = np.einsum('ij,ij->i', diff, diff)
    d = np.sqrt(np.maximum(d2, 1e-24))
    g = np.empty(_NC)
    J = np.empty((_NC, 3 * N))
    g[:_NP] = 0.5 * d
    inv = 0.5 / d
    J[:_NP, 0] = 0.0
    J[:_NP, 1] = 0.0
    J[:_NP, 2] = 0.0
    np.add.at(J[:_NP, 0], _TRI_I, inv * diff[:, 0])
    np.add.at(J[:_NP, 0], _TRI_J, -inv * diff[:, 0])
    np.add.at(J[:_NP, 1], _TRI_I, inv * diff[:, 1])
    np.add.at(J[:_NP, 1], _TRI_J, -inv * diff[:, 1])
    np.add.at(J[:_NP, 2], _TRI_I, inv * diff[:, 2])
    np.add.at(J[:_NP, 2], _TRI_J, -inv * diff[:, 2])
    b = _NP
    for ax in range(3):
        g[b:b + N] = 0.5 - c[:, ax]
        J[b:b + N, ax] = -1.0
        b += N
        g[b:b + N] = c[:, ax] + 0.5
        J[b:b + N, ax] = 1.0
        b += N
    return g, J


def _epigraph(x, lam, kmax):
    c = x[:3 * N].reshape(N, 3)
    t = x[3 * N]
    g, J = _grads(c)
    viol = np.maximum(g - t, 0.0)
    # objective: minimize sum of (active violations) + lam*max_viol
    # smooth-ish: use softplus of violations for differentiability
    s = 20.0 * viol
    obj = np.sum(np.log1p(np.exp(-s)) / 20.0)  # softplus(s)/20 ~ viol
    # grad wrt x: d/dx [softplus(20*(g-t))/20] = 20*sigma(-20*viol) * d(g-t)/dx
    w = np.exp(-s) / (1.0 + np.exp(-s))  # sigma(-20*viol)
    w = np.where(viol > 0, w, 0.0)
    # d(g-t)/dx: for constraint i, d g_i/dx = J[i]; d(-t)/dx = -e_t
    Jx = J * w[:, None]
    gx = np.sum(Jx, axis=0)
    gx[3 * N] -= np.sum(w)
    return obj, gx


def _epi_opt(c0, deadline, max_iter=400):
    x0 = np.concatenate([c0.flatten(), [_radius(c0) - 1e-4]])
    try:
        res = minimize(_epigraph, x0, args=(1e-3, 30), method='L-BFGS-B',
                       bounds=_BOUNDS + [(0.0, 0.49999)],
                       options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-12,
                                'maxfun': max_iter * 20, 'maxls': 100, 'maxcor': 60})
        c = res.x[:3 * N].reshape(N, 3)
        return c
    except Exception:
        return c0.copy()


def _active_refine(c, deadline, max_iter=60):
    """Newton-like refinement on the k most-binding constraints (equality)."""
    c = c.copy()
    for _ in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        g, J = _grads(c)
        t = np.min(g)
        # active set: constraints within tolerance of the min
        tol = max(1e-6, 0.02 * (t + 1e-9))
        active = np.where(g <= t + tol)[0]
        if len(active) == 0:
            break
        if len(active) > 40:
            # keep the 40 most binding
            order = np.argsort(g)
            active = order[:40]
        Ja = J[active]                       # (m, 3N)
        ga = g[active] - t                   # (m,)
        # minimize sum((Ja c - (ga+t))^2)  -> solve  Ja^T Ja c = Ja^T (ga+t)
        A = Ja.T @ Ja
        rhs = Ja.T @ (ga + t)
        try:
            # damped Newton step toward satisfying constraints
            lam = 1e-4
            sol = np.linalg.solve(A + lam * np.eye(3 * N), rhs)
        except Exception:
            sol = np.linalg.lstsq(A + lam * np.eye(3 * N), rhs, rcond=None)[0]
        # sol is the point satisfying active constraints; move toward it
        step = sol - c.flatten()
        # damped
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]:
            c2 = c.flatten() + alpha * step
            c2 = np.clip(c2, -0.49999, 0.49999)
            c2 = c2.reshape(N, 3)
            r2 = _radius(c2)
            if r2 >= t - 1e-12:
                c = c2
                break
        else:
            break
    return c


def _init_grid():
    m = 4
    t = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in t for y in t for x in t])
    return g[:N].copy()


def _init_bcc():
    nc = 4
    cs = 1.0 / nc
    pts = []
    for iz in range(nc):
        for iy in range(nc):
            for ix in range(nc):
                cx = -0.5 + cs * (ix + 0.5)
                cy = -0.5 + cs * (iy + 0.5)
                cz = -0.5 + cs * (iz + 0.5)
                pts.append([cx, cy, cz])
                pts.append([cx + cs / 2, cy + cs / 2, cz + cs / 2])
    a = np.array(pts)
    a = a[np.all(np.abs(a) <= 0.5, axis=1)]
    if len(a) >= N:
        return a[:N].copy()
    extra = np.random.uniform(-0.45, 0.45, size=(N - len(a), 3))
    return np.vstack([a, extra])


def _init_fcc():
    a = 1.0 / 3
    pts = []
    for iz in range(4):
        for iy in range(4):
            for ix in range(4):
                cx = -0.5 + a * ix
                cy = -0.5 + a * iy
                cz = -0.5 + a * iz
                pts.append([cx, cy, cz])
                if ix < 3:
                    pts.append([cx + a / 2, cy, cz])
                if iy < 3:
                    pts.append([cx, cy + a / 2, cz])
                if iz < 3:
                    pts.append([cx, cy, cz + a / 2])
                if ix < 3 and iy < 3 and iz < 3:
                    pts.append([cx + a / 2, cy + a / 2, cz + a / 2])
    a_arr = np.array(pts)
    a_arr = a_arr[np.all(np.abs(a_arr) <= 0.5, axis=1)]
    if len(a_arr) >= N:
        return a_arr[:N].copy()
    extra = np.random.uniform(-0.45, 0.45, size=(N - len(a_arr), 3))
    return np.vstack([a_arr, extra])


def _init_layered():
    pts = []
    for layer in range(5):
        z = -0.4 + 0.2 * layer
        for j in range(10):
            x = -0.45 + 0.09 * j
            y_offset = 0.03 * (j % 2)
            y = -0.3 + y_offset
            pts.append([x, y, z])
    a = np.array(pts)
    a = a[np.all(np.abs(a) <= 0.5, axis=1)]
    if len(a) >= N:
        return a[:N].copy()
    extra = np.random.uniform(-0.45, 0.45, size=(N - len(a), 3))
    return np.vstack([a, extra])


def _init_random(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(N, 3))


def _perturb(c, seed, scale=0.03):
    rng = np.random.RandomState(seed)
    c2 = c + rng.normal(0, scale, size=c.shape)
    return np.clip(c2, -0.49999, 0.49999)


def construct():
    start = time.time()
    deadline = start + 143.0  # safety margin before 150s kill

    np.random.seed(12345)

    inits = [
        _init_grid(), _init_bcc(), _init_fcc(), _init_layered(),
        _init_random(1), _init_random(7), _init_random(13), _init_random(42),
    ]

    best_c = inits[0].copy()
    best_r = _radius(best_c)

    # Phase 1: epigraph optimize from each init (time-sliced)
    for k, init in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
        rem = deadline - 25.0 - time.time()
        max_iter = int(min(350, max(60, rem * 25)))
        c = _epi_opt(init, deadline, max_iter)
        r = _radius(c)
        if r > best_r:
            best_r, best_c = r, c.copy()

    # Phase 2: active-set refinement on best
    if time.time() < deadline - 15.0:
        c = _active_refine(best_c, deadline)
        r = _radius(c)
        if r > best_r:
            best_r, best_c = r, c.copy()

    # Phase 3: perturb-and-refine around best to escape local minima
    if time.time() < deadline - 10.0:
        for seed in [101, 202, 303, 404, 505]:
            if time.time() > deadline - 8.0:
                break
            c0 = _perturb(best_c, seed, scale=0.02)
            c = _epi_opt(c0, deadline, max_iter=200)
            c = _active_refine(c, deadline)
            r = _radius(c)
            if r > best_r:
                best_r, best_c = r, c.copy()

    # Phase 4: final high-precision polish
    if time.time() < deadline - 4.0:
        c = _active_refine(best_c, deadline, max_iter=80)
        r = _radius(c)
        if r > best_r:
            best_r, best_c = r, c.copy()

    best_c = np.clip(best_c, -0.49999, 0.49999)
    if not np.all(np.isfinite(best_c)):
        best_c = _init_grid()
    return best_c
