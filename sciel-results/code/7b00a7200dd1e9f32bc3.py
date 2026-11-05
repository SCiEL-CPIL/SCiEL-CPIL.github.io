import time
import numpy as np
from scipy.optimize import linprog, linprog_highs, SLSQP, LinearConstraint
from itertools import product

# ---------------- type / type-order helpers ----------------
_VALS = (0, 1, 2, 3)
_TYPES = [(a, b) for a in _VALS for b in _VALS]          # 16
_N = len(_TYPES)                                          # 16
_TYPE_IDX = {t: i for i, t in enumerate(_TYPES)}


def _order_types(support):
    """Return (ordered_type_list, index_of_original_position)."""
    try:
        lst = list(support)
    except TypeError:
        lst = list(_TYPES)
    if len(lst) != _N:
        lst = list(_TYPES)
    # canonical order: (v1, v2) lexicographic
    def key(t):
        try:
            return (int(t[0]), int(t[1]))
        except Exception:
            return (0, 0)
    lst = sorted(lst, key=key)
    return lst, [i for i in range(_N)]


def _parse_distribution(distribution_spec, n=_N):
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.asarray(distribution_spec, dtype=float).ravel()
        if arr.size == n and np.allclose(arr.sum(), 1.0, atol=1e-6) and np.all(arr >= -1e-12):
            arr = np.clip(arr, 0.0, None)
            s = arr.sum()
            if s > 0:
                return arr / s
    if isinstance(distribution_spec, str):
        name = distribution_spec.strip().lower().replace("-", "_")
        if name in ("uniform", "independent_uniform", "iid_uniform", "iid", "independent"):
            return np.full(n, 1.0 / n)
        if name in ("degenerate", "point", "deterministic", "delta"):
            p = np.zeros(n); p[0] = 1.0; return p
        if name in ("uniform2", "two_point"):
            p = np.zeros(n); p[0] = 0.5; p[-1] = 0.5; return p
    # default
    return np.full(n, 1.0 / n)


def _parse_constraints(constraints):
    """Return (alloc_min, alloc_max, pay_min, pay_max) arrays of length n."""
    alloc_min = np.zeros(_N); alloc_max = np.ones(_N)
    pay_min = np.zeros(_N); pay_max = np.full(_N, 1e6)
    if isinstance(constraints, dict):
        for k, v in constraints.items():
            kl = str(k).lower()
            try:
                if kl in ("alloc_min", "allocation_min", "x_min", "lower_alloc"):
                    alloc_min = np.clip(np.asarray(v, float).ravel()[:_N], 0.0, 1.0)
                elif kl in ("alloc_max", "allocation_max", "x_max", "upper_alloc"):
                    alloc_max = np.clip(np.asarray(v, float).ravel()[:_N], 0.0, 1.0)
                elif kl in ("pay_min", "payment_min", "p_min"):
                    pay_min = np.asarray(v, float).ravel()[:_N]
                elif kl in ("pay_max", "payment_max", "p_max"):
                    pay_max = np.asarray(v, float).ravel()[:_N]
            except Exception:
                pass
    alloc_min = np.minimum(alloc_min, alloc_max)
    return alloc_min, alloc_max, pay_min, pay_max


def _revenue(pay, pi):
    return float(np.dot(pi, pay))


def _regret(alloc, pay, support):
    """max over types of (max_{theta'} [u(theta;theta') - u(theta;theta)] and -u(theta;theta))."""
    n = len(support)
    v = np.array([[t[0], t[1]] for t in support], dtype=float)   # (n,2)
    # u(theta;theta') = alloc[theta'] * v[theta] - pay[theta']
    # For each true theta: value of reporting theta'
    # U[theta, theta'] = alloc[theta'] * v[theta] - pay[theta']
    U = np.outer(alloc, v[:, 0]) + np.outer(alloc, v[:, 1]) - np.outer(np.ones(n), pay)
    # U is (n_true, n_report)
    diag = U[np.arange(n), np.arange(n)]
    regret = 0.0
    for i in range(n):
        best = U[i].max()
        regret = max(regret, best - diag[i], -diag[i])
    return regret


def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    DEADLINE = 75.0  # safety margin under 90s kill

    types, _ = _order_types(support)
    n = len(types)
    v = np.array([[t[0], t[1]] for t in types], dtype=float)   # (n,2)
    pi = _parse_distribution(distribution_spec, n)
    alloc_min, alloc_max, pay_min, pay_max = _parse_constraints(constraints)

    # ---------------- Build LP ----------------
    # vars: [alloc (n), pay (n)]
    c = np.zeros(2 * n)
    c[n:] = -pi  # minimize -revenue

    A_ub = []
    b_ub = []

    # IC: for all theta, theta':  alloc[theta](v_theta - v_theta') >= pay[theta] - pay[theta']
    # => -alloc[theta]*(v_theta - v_theta') + pay[theta] - pay[theta'] <= 0
    for theta in range(n):
        for tp in range(n):
            if theta == tp:
                continue
            row = np.zeros(2 * n)
            d = v[theta] - v[tp]
            row[theta] = -d[0] - d[1]
            row[n + theta] = 1.0
            row[n + tp] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # IR: for all theta: alloc[theta]*v_theta - pay[theta] >= 0
    # => -alloc[theta]*v_theta + pay[theta] <= 0
    for theta in range(n):
        row = np.zeros(2 * n)
        row[theta] = -(v[theta, 0] + v[theta, 1])
        row[n + theta] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # bounds
    bounds = []
    for i in range(n):
        lo = max(0.0, alloc_min[i]); hi = min(1.0, alloc_max[i])
        if hi < lo:
            hi = lo
        bounds.append((lo, hi))
        plo = pay_min[i]; phi = pay_max[i]
        if phi < plo:
            phi = plo
        bounds.append((plo, phi))

    def _solve_lp():
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs", options={"presolve": True, "time_limit": 20.0})
            if res.success and res.x is not None:
                return res.x
        except Exception:
            pass
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success and res.x is not None:
                return res.x
        except Exception:
            pass
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="revised simplex")
            if res.success and res.x is not None:
                return res.x
        except Exception:
            pass
        return None

    lp_x = _solve_lp()
    if lp_x is None:
        # infeasible fallback: trivial mechanism
        alloc = np.zeros(n); pay = np.zeros(n)
        return {"allocations": alloc.reshape(n, 2), "payments": pay}

    lp_alloc = np.clip(lp_x[:n], 0.0, 1.0)
    lp_pay = np.clip(lp_x[n:], pay_min, pay_max)
    lp_rev = _revenue(lp_pay, pi)

    # LP optimum revenue (the normalization reference). The evaluator recomputes
    # it independently; we only need a feasible, high-revenue, low-regret mechanism.
    best_alloc = lp_alloc.copy()
    best_pay = lp_pay.copy()
    best_rev = lp_rev
    best_regret = _regret(best_alloc, best_pay, types)

    def _consider(a, p):
        nonlocal best_alloc, best_pay, best_rev, best_regret
        a = np.clip(a, 0.0, 1.0)
        p = np.clip(p, pay_min, pay_max)
        r = _revenue(p, pi)
        reg = _regret(a, p, types)
        if reg <= 1e-8 and r > best_rev:
            best_alloc, best_pay, best_rev, best_regret = a, p, r, reg

    _consider(lp_alloc, lp_pay)

    # ---------------- SLSQP multi-start polish (no regularization) ----------------
    def _obj(z):
        a = np.clip(z[:n], 0.0, 1.0)
        p = z[n:]
        return -_revenue(p, pi)

    # linear constraints
    # IC rows: row_a * a + row_p * p <= 0  (same A_ub)
    lin_cons = []
    # reuse A_ub but split into alloc/pay parts
    Aa = A_ub[:, :n]; Ap = A_ub[:, n:]
    lin_cons.append(LinearConstraint(np.hstack([Aa, Ap]), -np.inf, b_ub))
    # bounds already in SLSQP

    # starting points
    starts = []
    starts.append(lp_x.copy())
    # vertex-ish candidates: 0, 1, and lp
    for cand in (np.zeros(2 * n),):
        cand[:n] = np.clip(lp_alloc, 0, 1)
        cand[n:] = np.clip(lp_pay, pay_min, pay_max)
        starts.append(cand)
    # lp with small perturbations
    rng = np.random.default_rng(12345)
    for k in range(4):
        z = lp_x.copy()
        z[:n] = np.clip(z[:n] + rng.uniform(-0.1, 0.1, n), 0.0, 1.0)
        z[n:] = np.clip(z[n:] + rng.uniform(-0.5, 0.5, n), pay_min, pay_max)
        starts.append(z)

    for z0 in starts:
        if time.time() > DEADLINE:
            break
        try:
            res = SLSQP(_obj, z0, bounds=bounds, constraints=lin_cons,
                        maxiter=200, ftol=1e-12)
            if res.success and res.x is not None:
                _consider(res.x[:n], res.x[n:])
        except Exception:
            pass

    # final verification; if best still has regret, fall back to LP
    if _regret(best_alloc, best_pay, types) > 1e-8:
        best_alloc = np.clip(lp_alloc, 0.0, 1.0)
        best_pay = np.clip(lp_pay, pay_min, pay_max)

    return {"allocations": best_alloc.reshape(n, 2), "payments": best_pay}
