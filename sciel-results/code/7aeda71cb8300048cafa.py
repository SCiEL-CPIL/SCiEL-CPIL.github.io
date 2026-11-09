# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))
    idx_a = X.astype(np.int64).ravel()
    idx_b = (X + Y).astype(np.int64).ravel()
    idx_c = (X - Y + 9).astype(np.int64).ravel()
    idx_d = (X + 2 * Y).astype(np.int64).ravel()
    return idx_a, idx_b, idx_c, idx_d

def _softmax(z):
    z2 = z - np.max(z)
    e = np.exp(z2)
    s = e.sum()
    if s < 1e-300 or not np.isfinite(s):
        return np.full_like(z, 1.0 / len(z))
    return e / s

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _entropy_grad(q):
    m = q > 1e-300
    H = -np.sum(q[m] * np.log2(q[m]))
    dq = np.zeros_like(q)
    dq[m] = -(np.log2(q[m]) + 1.0 / np.log(2.0))
    return H, dq

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M < 1e-12:
        return -1e9
    return D / M

def _make_f_and_grad(idx_a, idx_b, idx_c, idx_d):
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    log2 = 1.0 / np.log(2.0)

    def f_and_grad(z):
        P = _softmax(z)
        qa = np.bincount(idx_a, weights=P, minlength=la)
        qb = np.bincount(idx_b, weights=P, minlength=lb)
        qc = np.bincount(idx_c, weights=P, minlength=lc)
        qd = np.bincount(idx_d, weights=P, minlength=ld)

        D, dD_dc = _entropy_grad(qc)
        Ha, dHa_da = _entropy_grad(qa)
        Hb, dHb_db = _entropy_grad(qb)
        Hd, dHd_dd = _entropy_grad(qd)

        M = max(Ha, Hb, Hd)
        if M < 1e-12:
            return -1e9, np.zeros_like(z)

        val = D / M

        # Gradient via chain rule
        # dval/dP_i for each pooled variable
        g_a = np.zeros(100)
        g_b = np.zeros(100)
        g_c = np.zeros(100)
        g_d = np.zeros(100)

        # dD/dP_i = dD/dqc * (1 if idx_c[i] is the bin)
        g_c[idx_c] = dD_dc
        g_a[idx_a] = -dHa_da / M
        g_b[idx_b] = -dHb_db / M
        g_d[idx_d] = -dHd_dd / M

        # Only the argmax M contributes to the gradient of M
        if Ha == M:
            g_a[idx_a] = -dHa_da / M
            g_b[idx_b] = np.zeros(100)
            g_d[idx_d] = np.zeros(100)
        elif Hb == M:
            g_a[idx_a] = np.zeros(100)
            g_b[idx_b] = -dHb_db / M
            g_d[idx_d] = np.zeros(100)
        else:
            g_a[idx_a] = np.zeros(100)
            g_b[idx_b] = np.zeros(100)
            g_d[idx_d] = -dHd_dd / M

        gP = g_c + g_a + g_b + g_d

        # dP/dz = P * (I - P)  (for softmax)
        g_z = P * (gP - np.dot(P, gP))

        return -val, -g_z

    def fun(z):
        v, _ = f_and_grad(z)
        return v

    return f_and_grad, fun

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def _make_structured_starts():
    """Generate diverse structured initializations in log-prob space."""
    starts = []
    n = 10
    X, Y = np.meshgrid(np.arange(n), np.arange(n))

    # 1. Uniform
    starts.append(np.zeros(100))

    # 2. Diagonal bands: emphasize cells where x-y is in a specific range
    for center in range(-4, 5):
        z = np.full(100, -20.0)
        diff = X - Y
        mask = np.abs(diff - center) <= 1
        z[mask.ravel()] = 0.0
        starts.append(z)

    # 3. L-shapes: emphasize rows and columns
    for x0 in range(0, 10, 3):
        for y0 in range(0, 10, 3):
            z = np.full(100, -20.0)
            z[x0, :] = 0.0   # row
            z[:, y0] = 0.0   # column
            starts.append(z)

    # 4. Corner-heavy
    for cx in [0, 9]:
        for cy in [0, 9]:
            z = np.full(100, -20.0)
            z[cy, cx] = 0.0
            starts.append(z)

    # 5. Ring pattern
    z = np.full(100, -20.0)
    for i in range(10):
        for j in range(10):
            d = max(abs(i - 4.5), abs(j - 4.5))
            if 2.5 <= d <= 4.5:
                z[i, j] = 0.0
    starts.append(z)

    # 6. Random structured
    rng = np.random.default_rng(123)
    for _ in range(30):
        z = rng.normal(0, 2.0, 100)
        starts.append(z)

    # 7. Anti-diagonal emphasis
    for k in range(0, 19, 3):
        z = np.full(100, -20.0)
        mask = (X + Y == k) | (X + Y == k - 1) | (X + Y == k + 1)
        z[mask.ravel()] = 0.0
        starts.append(z)

    # 8. Sparse random subsets
    for _ in range(15):
        z = np.full(100, -30.0)
        subset = rng.choice(100, size=rng.integers(8, 25), replace=False)
        z[subset] = 0.0
        starts.append(z)

    return starts

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    f_and_grad, fun = _make_f_and_grad(idx_a, idx_b, idx_c, idx_d)

    start_time = time.time()
    deadline = start_time + 620.0

    best_z = None
    best_val = -1e9

    def update_best(z, val):
        nonlocal best_z, best_val
        if val > best_val:
            best_val = val
            best_z = z.copy()

    # Evaluate all structured starts
    starts = _make_structured_starts()
    for z in starts:
        P = _softmax(z).reshape(n, n)
        v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
        update_best(z, v)

    # Multi-restart L-BFGS-B
    max_restarts = 40
    for r in range(max_restarts):
        if time.time() > deadline - 120.0:
            break

        # Choose starting point
        if r < len(starts):
            z0 = starts[r].copy()
        else:
            # Perturb best found so far
            z0 = best_z + np.random.default_rng(r).normal(0, 0.5, 100)

        try:
            result = minimize(
                f_and_grad, z0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            z_opt = result.x
            P = _softmax(z_opt).reshape(n, n)
            v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            update_best(z_opt, v)
        except Exception:
            pass

    # Local polish: fine-grained local search around best
    if time.time() < deadline - 60.0:
        rng = np.random.default_rng(999)
        for _ in range(200):
            if time.time() > deadline - 60.0:
                break
            pert = best_z + rng.normal(0, 0.05, 100)
            P = _softmax(pert).reshape(n, n)
            v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            update_best(pert, v)

    # Final gradient polish
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                f_and_grad, best_z,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            z_opt = result.x
            P = _softmax(z_opt).reshape(n, n)
            v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            update_best(z_opt, v)
        except Exception:
            pass

    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P

# EVOLVE-BLOCK-END
