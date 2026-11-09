# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d


def _softmax(z):
    z2 = z - np.max(z)
    e = np.exp(z2)
    return e / e.sum()


def _entropy_grad(q):
    m = q > 1e-300
    H = -np.sum(q[m] * np.log2(q[m]))
    dq = np.zeros_like(q)
    dq[m] = -(np.log2(q[m]) + 1.0 / np.log(2.0))
    return H, dq


def _make_f_and_grad(idx_a, idx_b, idx_c, idx_d):
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1

    def f_and_grad(z):
        P = _softmax(z).reshape(10, 10)

        qa = np.zeros(la); qa[idx_a] += P.ravel()
        qb = np.zeros(lb); qb[idx_b] += P.ravel()
        qc = np.zeros(lc); qc[idx_c] += P.ravel()
        qd = np.zeros(ld); qd[idx_d] += P.ravel()

        Hc, dHc_dqc = _entropy_grad(qc)
        Ha, dHa_dqa = _entropy_grad(qa)
        Hb, dHb_dqb = _entropy_grad(qb)
        Hd, dHd_dqd = _entropy_grad(qd)

        Hs = np.array([Ha, Hb, Hd])
        M = np.max(Hs)
        active = Hs == M

        D = Hc
        value = D / M if M > 1e-12 else 0.0

        # d(value)/d(P) via chain rule
        # value = D/M
        # dvalue/dP = (dD/dP * M - D * dM/dP) / M^2
        dD_dP = np.zeros(100)
        for i in range(100):
            dD_dP[i] = dHc_dqc[idx_c[i]]

        dM_dP = np.zeros(100)
        for i in range(100):
            contrib = 0.0
            if active[0]:
                contrib += dHa_dqa[idx_a[i]]
            if active[1]:
                contrib += dHb_dqb[idx_b[i]]
            if active[2]:
                contrib += dHd_dqd[idx_d[i]]
            dM_dP[i] = contrib

        dvalue_dP = (dD_dP * M - D * dM_dP) / (M * M)

        # dP/dz for softmax: P_j * (delta_ij - P_i)
        # grad_z = P * (dvalue_dP - sum(P * dvalue_dP))
        s = np.sum(P.ravel() * dvalue_dP)
        grad_z = P.ravel() * (dvalue_dP - s)

        return -value, -grad_z

    return f_and_grad


def _objective(z, f_and_grad):
    val, grad = f_and_grad(z)
    return val, grad


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    The baseline is the uniform distribution over the grid.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds

    idx_a, idx_b, idx_c, idx_d = _build_idx()
    f_and_grad = _make_f_and_grad(idx_a, idx_b, idx_c, idx_d)

    # Start from uniform
    z0 = np.zeros(100)
    best_z = z0.copy()
    best_val, _ = f_and_grad(z0)
    best_val = -best_val  # convert back to positive value

    # Try multiple random restarts with different initializations
    rng = np.random.default_rng(42)

    def check_time():
        return time.time() < deadline - safety_margin

    # First, do a single optimization from uniform
    try:
        res = minimize(
            _objective, z0, args=(f_and_grad,),
            method='L-BFGS-B', jac=True,
            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if check_time():
            val, _ = f_and_grad(res.x)
            if -val > best_val:
                best_val = -val
                best_z = res.x.copy()
    except Exception:
        pass

    # Multiple random restarts
    n_restarts = 20
    for r in range(n_restarts):
        if not check_time():
            break

        # Random initialization with varying scales
        scale = 0.1 + 0.5 * r / n_restarts
        z_init = rng.normal(0, scale, size=100)

        try:
            res = minimize(
                _objective, z_init, args=(f_and_grad,),
                method='L-BFGS-B', jac=True,
                options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if check_time():
                val, _ = f_and_grad(res.x)
                if -val > best_val:
                    best_val = -val
                    best_z = res.x.copy()
        except Exception:
            pass

    # Also try some structured initializations
    # Diagonal concentration
    if check_time():
        z_diag = np.zeros(100)
        for i in range(10):
            z_diag[i * 10 + i] = 2.0
        try:
            res = minimize(
                _objective, z_diag, args=(f_and_grad,),
                method='L-BFGS-B', jac=True,
                options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if check_time():
                val, _ = f_and_grad(res.x)
                if -val > best_val:
                    best_val = -val
                    best_z = res.x.copy()
        except Exception:
            pass

    # Anti-diagonal concentration
    if check_time():
        z_adiag = np.zeros(100)
        for i in range(10):
            z_adiag[i * 10 + (9 - i)] = 2.0
        try:
            res = minimize(
                _objective, z_adiag, args=(f_and_grad,),
                method='L-BFGS-B', jac=True,
                options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if check_time():
                val, _ = f_and_grad(res.x)
                if -val > best_val:
                    best_val = -val
                    best_z = res.x.copy()
        except Exception:
            pass

    # Row/column concentration patterns
    if check_time():
        z_rows = np.zeros(100)
        for i in range(10):
            z_rows[i * 10 + (i % 5)] = 1.5
        try:
            res = minimize(
                _objective, z_rows, args=(f_and_grad,),
                method='L-BFGS-B', jac=True,
                options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if check_time():
                val, _ = f_and_grad(res.x)
                if -val > best_val:
                    best_val = -val
                    best_z = res.x.copy()
        except Exception:
            pass

    # Final refinement from best solution
    if check_time() and best_val > 1.0:
        try:
            res = minimize(
                _objective, best_z, args=(f_and_grad,),
                method='L-BFGS-B', jac=True,
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val, _ = f_and_grad(res.x)
            if -val > best_val:
                best_val = -val
                best_z = res.x.copy()
        except Exception:
            pass

    P = _softmax(best_z).reshape(10, 10)

    # Ensure validity: clip negatives, normalize
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / total

    return P


# EVOLVE-BLOCK-END
