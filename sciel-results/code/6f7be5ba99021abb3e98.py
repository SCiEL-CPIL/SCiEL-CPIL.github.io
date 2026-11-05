import numpy as np
import time
from scipy.linalg import expm

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


def _compute_error(a, b, A, B, Uexact, times):
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    errors = []
    for t in times:
        U = I.copy()
        for k in range(len(a)):
            # Product order: exp(-i b_k t B) exp(-i a_k t A)
            U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
        errors.append(np.linalg.norm(U - Uexact[t], 2))
    return float(np.mean(errors))


def _nelder_mead(objective, x0, max_iter=100, tol=1e-8, deadline=None):
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05 if abs(x0[i]) < 1e-8 else 0.05 * x0[i]
        x_new[i] += step
        simplex.append(x_new)
    
    # Ensure initial simplex is valid
    valid_fvals = []
    for x in simplex:
        f = objective(x)
        if np.isfinite(f):
            valid_fvals.append(f)
        else:
            valid_fvals.append(1e10)
    fvals = valid_fvals

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        if not np.isfinite(fr):
            fr = 1e10

        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = objective(xe)
            if not np.isfinite(fe):
                fe = 1e10
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if not np.isfinite(fc):
                fc = 1e10
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
                    if not np.isfinite(fvals[i]):
                        fvals[i] = 1e10

    return simplex[0], fvals[0]


def coefficients(stages):
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s limit

    n = stages
    A, B = _build_hamiltonians()
    H = A + B
    times = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    Uexact = {}
    for t in times:
        Uexact[t] = expm(-1j * t * H)

    half = n // 2
    # Initial guess: uniform
    x0 = np.array([1.0 / n] * half + [1.0 / n] * half)

    def objective(params):
        # Expand to full palindromic a, b
        a = np.zeros(n)
        b = np.zeros(n)
        for k in range(half):
            a[k] = params[k]
            a[n - 1 - k] = params[k]
            b[k] = params[half + k]
            b[n - 1 - k] = params[half + k]
        
        # Check bounds
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
        
        # Check sum constraint (loose tolerance for optimizer, strict for final)
        sa = np.sum(a)
        sb = np.sum(b)
        if abs(sa - 1.0) > 1e-6 or abs(sb - 1.0) > 1e-6:
            return 1e10
            
        return _compute_error(a, b, A, B, Uexact, times)

    best_a, best_b = None, None
    best_err = float('inf')

    # Restart 0: Uniform start
    x_opt, f_opt = _nelder_mead(objective, x0, max_iter=80, tol=1e-7, deadline=deadline - 10.0)
    if f_opt < best_err:
        best_err = f_opt
        # Reconstruct a, b from x_opt
        a_opt = np.zeros(n)
        b_opt = np.zeros(n)
        for k in range(half):
            a_opt[k] = x_opt[k]
            a_opt[n - 1 - k] = x_opt[k]
            b_opt[k] = x_opt[half + k]
            b_opt[n - 1 - k] = x_opt[half + k]
        best_a, best_b = a_opt, b_opt

    # Additional restarts with perturbations if time permits
    rng = np.random.default_rng(42)
    max_restarts = 5
    for r in range(1, max_restarts):
        if time.time() > deadline - 15.0:
            break
        x_cur = x0 + rng.normal(0, 0.1, size=len(x0))
        try:
            x_opt, f_opt = _nelder_mead(objective, x_cur, max_iter=50, tol=1e-6, deadline=deadline - 15.0)
            if f_opt < best_err:
                best_err = f_opt
                a_opt = np.zeros(n)
                b_opt = np.zeros(n)
                for k in range(half):
                    a_opt[k] = x_opt[k]
                    a_opt[n - 1 - k] = x_opt[k]
                    b_opt[k] = x_opt[half + k]
                    b_opt[n - 1 - k] = x_opt[half + k]
                best_a, best_b = a_opt, b_opt
        except Exception:
            pass

    # Final constraint cleanup
    if best_a is None:
        # Fallback to uniform if optimization failed completely
        best_a = np.ones(n) / n
        best_b = np.ones(n) / n
    else:
        # Clamp and renormalize
        best_a = np.clip(best_a, -50.0, 50.0)
        best_b = np.clip(best_b, -50.0, 50.0)
        
        sa = np.sum(best_a)
        sb = np.sum(best_b)
        
        if abs(sa) > 1e-12:
            best_a = best_a / sa
        else:
            best_a = np.ones(n) / n
            
        if abs(sb) > 1e-12:
            best_b = best_b / sb
        else:
            best_b = np.ones(n) / n

    return best_a.tolist(), best_b.tolist()
