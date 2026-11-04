import numpy as np
import time

N = 100

def _fcc_init(n):
    """Generate an FCC-like initial configuration inside the unit cube."""
    if n <= 0:
        return np.empty((0, 3))
    a = 0.44  # lattice spacing chosen so spheres stay inside cube with margin
    pts = []
    count = 0
    idx = 0
    while len(pts) < n and idx < 300:
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    x = (i + 0.5) * a - 0.5
                    y = (j + 0.5) * a - 0.5
                    z = (k + 0.5) * a - 0.5
                    pts.append([x, y, z])
                    count += 1
                    if len(pts) == n:
                        break
                if len(pts) == n:
                    break
            if len(pts) == n:
                break
        idx += 1
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            pts.append(rng.uniform(-0.45, 0.45, size=3).tolist())
    return np.array(pts[:n], dtype=np.float64)


def _compute_radius(c):
    """Return the feasible common radius for centres c (N,3) in [-1/2,1/2]^3."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    wall = 0.5 - np.max(np.abs(c), axis=1)
    r = min(min_pair / 2.0, wall.min())
    return r


def _softmin_value_and_grad(v, beta):
    """Soft-min of vector v with log-sum-exp form. Returns (value, grad wrt v)."""
    beta = float(beta)
    neg = -beta * v
    # numerical stability
    m = np.max(neg)
    shifted = neg - m
    exps = np.exp(shifted)
    s = np.sum(exps)
    sm = -(m + np.log(s)) / beta
    p = exps / s
    grad = -beta * p
    return sm, grad


def _energy_value_and_grad(c, beta):
    """
    Differentiable objective: energy = -soft_min(pair distances) - soft_min(wall distances).
    We want to maximize the minimum, so we minimize this energy.
    c: (N,3). Returns (energy, grad).
    """
    N = c.shape[0]
    diff = c[:, None, :] - c[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    # flatten unique pairs
    iu = np.triu_indices(N, k=1)
    pair_d = d[iu]
    sm_pair, grad_pair = _softmin_value_and_grad(pair_d, beta)
    # wall distances: 0.5 - max_k |c_ik|
    amax = np.max(np.abs(c), axis=1)
    wall_d = 0.5 - amax
    sm_wall, grad_wall = _softmin_value_and_grad(wall_d, beta)

    energy = -sm_pair - sm_wall

    # gradient wrt c
    # d sm_pair / d d = grad_pair (shape npair)
    # d d / d c via diff
    # Build gradient tensor
    g = np.zeros_like(c)
    # For each pair (i,j), d_d/d_c_i = (c_i - c_j)/d, d_d/d_c_j = (c_j - c_i)/d
    idx_i, idx_j = iu
    for t in range(len(idx_i)):
        i = idx_i[t]
        j = idx_j[t]
        dij = pair_d[t]
        if dij <= 0:
            continue
        w = grad_pair[t]
        v = (c[i] - c[j]) / dij
        g[i] += w * v
        g[j] -= w * v
    # wall gradient
    # d wall_d_i / d c_i = -sign(c_i) * (1/3) component-wise where argmax attained; use soft argmax
    for i in range(N):
        abs_c = np.abs(c[i])
        # argmax of abs_c
        maxv = np.max(abs_c)
        # if ties, distribute
        ties = abs_c == maxv
        if not np.any(ties):
            k = int(np.argmax(abs_c))
            dwall_dc = -np.zeros(3)
            dwall_dc[k] = -np.sign(c[i, k])
        else:
            dwall_dc = np.zeros(3)
            for k in range(3):
                if ties[k]:
                    dwall_dc[k] = -0.5 * np.sign(c[i, k])
        g[i] += grad_wall[i] * dwall_dc

    return energy, g


def _adam_step(state, c, grad, lr, t, beta1, beta2, eps):
    m = state['m']
    v = state['v']
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    c_new = c - lr * m_hat / (np.sqrt(v_hat) + eps)
    state['m'] = m
    state['v'] = v
    return c_new


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin

    rng = np.random.default_rng(0)

    # Initialization: FCC lattice for symmetry
    c = _fcc_init(N)
    # ensure inside cube
    c = np.clip(c, -0.45, 0.45)

    best_c = c.copy()
    best_r = _compute_radius(best_c)

    state = {
        'm': np.zeros_like(c),
        'v': np.zeros_like(c),
    }
    t = 0
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Anneal temperature: start high (smooth) then decrease
    schedule = [
        # (beta_start, beta_end, steps, lr)
        (40.0, 20.0, 400, 0.010),
        (20.0, 10.0, 600, 0.008),
        (10.0, 5.0, 800, 0.006),
        (5.0, 3.0, 800, 0.004),
        (3.0, 2.0, 1000, 0.003),
        (2.0, 1.5, 1000, 0.002),
        (1.5, 1.2, 1000, 0.0015),
        (1.2, 1.0, 1000, 0.001),
    ]

    for (b0, b1, steps, lr) in schedule:
        for s in range(steps):
            if time.time() > deadline - 20.0:
                break
            beta = b0 + (b1 - b0) * (s / max(1, steps - 1))
            t += 1
            energy, grad = _energy_value_and_grad(c, beta)
            c = _adam_step(state, c, grad, lr, t, beta1, beta2, eps)
            # project back into cube
            c = np.clip(c, -0.48, 0.48)
            # periodic evaluation
            if s % 50 == 0:
                r = _compute_radius(c)
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
        if time.time() > deadline - 20.0:
            break

    # Local refinement: simple coordinate-wise / pair-wise hill climbing on hard objective
    if time.time() < deadline - 5.0:
        c = best_c.copy()
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            r_cur = _compute_radius(c)
            # random perturbations
            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                c_trial = c.copy()
                idx = rng.integers(0, N)
                delta = rng.normal(0, 0.005, size=3)
                c_trial[idx] = np.clip(c_trial[idx] + delta, -0.48, 0.48)
                r_trial = _compute_radius(c_trial)
                if r_trial > r_cur + 1e-12:
                    c = c_trial
                    r_cur = r_trial
                    improved = True

        r_final = _compute_radius(c)
        if r_final > best_r:
            best_r = r_final
            best_c = c.copy()

    # Final hard-constraint projection: ensure strictly inside
    best_c = np.clip(best_c, -0.4999, 0.4999)
    return best_c
