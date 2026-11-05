"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed high-accuracy reference quadrature (Gauss-Legendre, 64 pts)
_NREF = 64
_REF_T, _REF_W = np.polynomial.legendre.leggauss(_NREF)
_REF_T = 0.5 * (_REF_T + 1.0)
_REF_W = 0.5 * _REF_W

# z-samples used to measure worst-case error
_ZGRID = np.linspace(0.1, 2.0, 24)

# Portfolio of cosmologies to design the rule for
_PORTFOLIO = np.array([
    [0.30, -1.0,  0.0],
    [0.30, -1.0,  0.3],
    [0.30, -1.0, -0.3],
    [0.27, -1.0,  0.0],
    [0.27, -1.2,  0.5],
    [0.27, -0.8, -0.5],
    [0.31, -1.1,  0.2],
    [0.31, -0.9, -0.2],
    [0.30, -1.0,  0.5],
    [0.30, -1.0, -0.5],
    [0.30, -0.9,  0.3],
    [0.30, -1.1, -0.3],
], dtype=float)
_OM = _PORTFOLIO[:, 0]
_W0 = _PORTFOLIO[:, 1]
_WA = _PORTFOLIO[:, 2]
_NP = len(_OM)


def _E_inv(a, Om, w0, wa):
    """1/E(a) for a CPL model, vectorized over a."""
    Ode = 1.0 - Om
    if Ode < 0.0:
        Ode = 0.0
    term1 = Om * a ** (-3.0)
    term2 = Ode * (a ** (-3.0 * w0)) * np.exp(-3.0 * wa * (1.0 - a))
    E2 = term1 + term2
    E2 = np.maximum(E2, 1e-300)
    return 1.0 / np.sqrt(E2)


def _quad_value(nodes, weights, z, Om, w0, wa):
    """(1+z)*z * sum_i weights[i]/E(a_i), a_i = 1/(1+z*nodes[i])."""
    a = 1.0 / (1.0 + z * nodes)
    einv = _E_inv(a, Om, w0, wa)
    return (1.0 + z) * z * np.dot(weights, einv)


def _ref_value(z, Om, w0, wa):
    """High-accuracy reference via 64-point Gauss-Legendre on [0,1]."""
    a = 1.0 / (1.0 + z * _REF_T)
    einv = _E_inv(a, Om, w0, wa)
    return (1.0 + z) * z * np.dot(_REF_W, einv)


def _worst_case_error(nodes, weights):
    """Worst relative error (percent) across the portfolio and z-grid."""
    if np.any(nodes <= 0.0) or np.any(nodes >= 1.0) or np.any(np.diff(nodes) <= 0.0):
        return 1e9
    if np.any(weights <= 0.0):
        return 1e9
    z = _ZGRID
    a = 1.0 / (1.0 + np.outer(z, nodes))          # (nz, 8)
    max_err = 0.0
    for p in range(_NP):
        einv = _E_inv(a, _OM[p], _W0[p], _WA[p])  # (nz, 8)
        qv = (1.0 + z)[:, None] * z[:, None] * (einv @ weights)   # (nz,)
        rv = np.array([_ref_value(zi, _OM[p], _W0[p], _WA[p]) for zi in z])
        rel = np.abs(qv - rv) / np.maximum(np.abs(rv), 1e-12) * 100.0
        m = float(rel.max())
        if m > max_err:
            max_err = m
    return max_err


def _fix(x):
    """Project a candidate onto the feasible set (strictly increasing nodes,
    positive weights summing to 1)."""
    nodes = x[:8].copy()
    weights = x[8:].copy()
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    for i in range(1, 8):
        if nodes[i] - nodes[i - 1] < 1e-9:
            nodes[i] = nodes[i - 1] + 1e-9
    weights = np.maximum(weights, 1e-10)
    weights = weights / weights.sum()
    return nodes, weights


def construct():
    t0 = time.time()
    deadline = 80.0  # wall-clock guard (evaluator kills at 90s)

    from scipy.optimize import minimize

    # Candidate starting points
    starts = []
    gl_t, gl_w = np.polynomial.legendre.leggauss(8)
    starts.append(np.concatenate([(gl_t + 1.0) / 2.0, gl_w / 2.0]))
    starts.append(np.concatenate([np.linspace(0.08, 0.92, 8), np.ones(8) / 8.0]))
    starts.append(np.concatenate([np.array([0.05, 0.18, 0.32, 0.46, 0.60, 0.74, 0.86, 0.97]),
                                  np.ones(8) / 8.0]))
    # a couple of random-feasible starts
    rng = np.random.default_rng(12345)
    for _ in range(4):
        n = np.sort(rng.uniform(0.02, 0.98, 8))
        w = rng.dirichlet(np.ones(8))
        starts.append(np.concatenate([n, w]))

    def _obj(x):
        if time.time() - t0 > deadline:
            return 1e10
        nodes, weights = _fix(x)
        return _worst_case_error(nodes, weights)

    best_x = starts[0].copy()
    best_err = _obj(best_x)

    for s in starts:
        if time.time() - t0 > deadline - 6.0:
            break
        cur = s.copy()
        cur_err = _obj(cur)
        if cur_err < best_err:
            best_err = cur_err
            best_x = cur
        # Nelder-Mead
        try:
            res = minimize(_obj, cur, method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-9})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass
        if time.time() - t0 > deadline - 4.0:
            break
        # L-BFGS-B from the best so far
        try:
            res = minimize(_obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 200, 'gtol': 1e-7})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final constrained polish (SLSQP) to tighten weight-sum and monotonicity
    if time.time() - t0 < deadline - 2.0:
        try:
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0}]
            bounds = [(1e-8, 1.0 - 1e-8)] * 8 + [(1e-10, 1.0)] * 8
            res = minimize(_obj, best_x, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-14})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass

    nodes, weights = _fix(best_x)

    # Final cleanup: guarantee strict increase, positivity, and sum==1 to 1e-12
    nodes = np.sort(nodes)
    for i in range(1, 8):
        if nodes[i] - nodes[i - 1] < 1e-10:
            nodes[i] = nodes[i - 1] + 1e-10
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
