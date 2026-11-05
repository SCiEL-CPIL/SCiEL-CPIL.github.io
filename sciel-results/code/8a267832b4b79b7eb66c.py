"""Product-formula design for Heisenberg-chain time evolution.

Builds the odd/even-bond halves of a 6-site open XXZ chain locally,
then refines 12-stage product-formula coefficients with a fast
Nelder-Mead optimizer under palindromic symmetry and sum constraints.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    n = 6
    dim = 2 ** n
    d = 0.8
    h = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)

    # Precompute per-site single-site operators
    site_ops = {}
    for site in range(n):
        for name, op in (("sx", sx), ("sy", sy), ("sz", sz)):
            ops = [ident] * n
            ops[site] = op
            r = ops[0]
            for o in ops[1:]:
                r = np.kron(r, o)
            site_ops[(site, name)] = r

    def bond_op(i, j):
        return (site_ops[(i, "sx")] @ site_ops[(j, "sx")]
                + site_ops[(i, "sy")] @ site_ops[(j, "sy")]
                + d * site_ops[(i, "sz")] @ site_ops[(j, "sz")])

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    for i, j in odd_bonds:
        A += bond_op(i, j)
    for i, j in even_bonds:
        B += bond_op(i, j)

    half_field = 0.5 * h
    for site in range(n):
        A += half_field * site_ops[(site, "sz")]
        B += half_field * site_ops[(site, "sz")]

    return A, B


def _expm(M):
    """Matrix exponential via scaling-and-squaring with Taylor series."""
    n = M.shape[0]
    norm = np.linalg.norm(M, 2)
    if norm < 1e-14:
        return np.eye(n, dtype=complex)
    # Scale so that ||M/scale|| <= 1
    s = max(1, int(np.ceil(norm / 1.0)))
    Ms = M / s
    term = np.eye(n, dtype=complex)
    result = term.copy()
    for k in range(1, 60):
        term = term @ Ms / k
        result = result + term
        if np.linalg.norm(term, 2) < 1e-16:
            break
    for _ in range(s - 1):
        result = result @ result
    return result


def _symmetrize(params, n):
    """Expand half-parameters to full palindromic a, b lists."""
    half = n // 2
    a = np.zeros(n)
    b = np.zeros(n)
    for k in range(half):
        a[k] = params[k]
        a[n - 1 - k] = params[k]
        b[k] = params[half + k]
        b[n - 1 - k] = params[half + k]
    return a, b


def _compute_error(a, b, A, B, Uexact, times):
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    errors = []
    for t in times:
        U = I.copy()
        for k in range(len(a)):
            # Right-multiply in product order k=1..12
            U = _expm(-1j * b[k] * t * B) @ _expm(-1j * a[k] * t * A) @ U
        errors.append(np.linalg.norm(U - Uexact[t], 2))
    return float(np.mean(errors))


def _nelder_mead(objective, x0, max_iter=80, tol=1e-7, deadline=None):
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.1 if x0[i] == 0 else 0.1 * x0[i]
        x_new[i] += step
        simplex.append(x_new)
    fvals = [objective(x) for x in simplex]

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        if fr < fvals[0]:
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    return simplex[0], fvals[0]


def coefficients(stages):
    t_start = time.time()
    deadline = t_start + 135.0  # safety margin before 150 s wall-clock

    n = stages
    A, B = _build_hamiltonians()
    H = A + B
    times = [0.25, 0.5, 1.0]
    Uexact = {t: _expm(-1j * t * H) for t in times}

    half = n // 2
    # Initial guess: uniform Strang-like
    x0 = np.array([1.0 / n] * half + [1.0 / n] * half)

    def objective(params):
        a, b = _symmetrize(params, n)
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
        sa, sb = a.sum(), b.sum()
        if abs(sa - 1.0) > 1e-6 or abs(sb - 1.0) > 1e-6:
            return 1e10
        return _compute_error(a, b, A, B, Uexact, times)

    best_a, best_b = _symmetrize(x0, n)
    best_err = objective(x0)

    # A few Nelder-Mead restarts with perturbations
    rng = np.random.default_rng(0)
    n_restarts = 4
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        if r == 0:
            x_cur = x0.copy()
        else:
            x_cur = x0 + rng.normal(0, 0.05, size=len(x0))
        try:
            x_opt, f_opt = _nelder_mead(objective, x_cur, max_iter=60,
                                        tol=1e-6, deadline=deadline - 15.0)
            if f_opt < best_err:
                best_err = f_opt
                best_a, best_b = _symmetrize(x_opt, n)
        except Exception:
            pass

    # Final constraint cleanup: renormalize and clamp
    for arr_name in ("a", "b"):
        arr = best_a if arr_name == "a" else best_b
        arr = np.clip(arr, -50.0, 50.0)
        s = arr.sum()
        if abs(s) > 1e-12:
            arr = arr / s
        else:
            arr = np.ones(n) / n
        if arr_name == "a":
            best_a = arr
        else:
            best_b = arr

    return best_a.tolist(), best_b.tolist()
# EVOLVE-BLOCK-END
