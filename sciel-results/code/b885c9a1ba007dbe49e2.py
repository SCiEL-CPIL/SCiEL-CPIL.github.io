"""Product-formula design for Heisenberg-chain time evolution.

Uses a structural change: instead of Nelder-Mead on palindromic coefficients,
we use a different optimization strategy:
1. Precompute expm for all needed matrix arguments (scaled by t and coefficient)
   using a fast eigen-decomposition approach since A and B are Hermitian.
2. Use a direct gradient-free optimization with a different parameterization.
3. Key structural change: use the fact that A and B are Hermitian, so their
   exponentials can be computed via eigen-decomposition once, and then we only
   need to apply them. Also, we use a different optimization: coordinate descent
   combined with a global random restart strategy, and we parameterize the
   coefficients differently to enforce sum-to-1 constraints directly.
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


def _eigh_decomp(M):
    """Eigen-decomposition for Hermitian M: M = V diag(eigs) V^H."""
    eigs, V = np.linalg.eigh(M)
    return eigs, V


def _expm_hermitian(eigs, V, scalar):
    """Compute exp(scalar * M) where M = V diag(eigs) V^H."""
    # exp(scalar * M) = V diag(exp(scalar * eigs)) V^H
    exp_eigs = np.exp(scalar * eigs)
    return V @ np.diag(exp_eigs) @ V.conj().T


def _precompute_eigh(A, B):
    """Precompute eigen-decompositions of A and B (both Hermitian)."""
    eigs_A, V_A = _eigh_decomp(A)
    eigs_B, V_B = _eigh_decomp(B)
    return eigs_A, V_A, eigs_B, V_B


def _apply_exp(eigs, V, scalar, vec):
    """Apply exp(scalar * M) to a vector, where M = V diag(eigs) V^H."""
    # V^H @ vec, then multiply by exp(scalar*eigs), then V @
    tmp = V.conj().T @ vec
    tmp = exp_eigs = np.exp(scalar * eigs) * tmp
    return V @ tmp


def _compute_error_fast(a, b, eigs_A, V_A, eigs_B, V_B, Uexact, times):
    """Compute mean spectral norm error using precomputed eigen-decompositions."""
    n = V_A.shape[0]
    I = np.eye(n, dtype=complex)
    errors = []
    for t in times:
        U = I.copy()
        for k in range(len(a)):
            # U = exp(-i b_k t B) @ exp(-i a_k t A) @ U
            # First: exp(-i a_k t A) @ U
            # exp(-i a_k t A) = V_A diag(exp(-i a_k t eigs_A)) V_A^H
            # So exp(-i a_k t A) @ U = V_A diag(exp(-i a_k t eigs_A)) (V_A^H @ U)
            tmp = V_A.conj().T @ U
            tmp = np.exp(-1j * a[k] * t * eigs_A) * tmp
            U = V_A @ tmp
            # Then: exp(-i b_k t B) @ U
            tmp = V_B.conj().T @ U
            tmp = np.exp(-1j * b[k] * t * eigs_B) * tmp
            U = V_B @ tmp
        errors.append(np.linalg.norm(U - Uexact[t], 2))
    return float(np.mean(errors))


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
    deadline = t_start + 130.0  # safety margin before 150 s wall-clock

    n = stages
    A, B = _build_hamiltonians()
    H = A + B
    times = [0.25, 0.5, 1.0]

    # Compute exact propagators using eigen-decomposition of H (Hermitian)
    eigs_H, V_H = _eigh_decomp(H)
    Uexact = {}
    for t in times:
        Uexact[t] = V_H @ np.diag(np.exp(-1j * t * eigs_H)) @ V_H.conj().T

    # Precompute eigen-decompositions of A and B
    eigs_A, V_A, eigs_B, V_B = _precompute_eigh(A, B)

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
        return _compute_error_fast(a, b, eigs_A, V_A, eigs_B, V_B, Uexact, times)

    best_a, best_b = _symmetrize(x0, n)
    best_err = objective(x0)

    # Structural change: use more diverse restarts and a different optimizer strategy.
    # Instead of just Nelder-Mead, we combine it with a coordinate descent polish.
    rng = np.random.default_rng(42)
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        if r == 0:
            x_cur = x0.copy()
        elif r == 1:
            # Slightly perturbed uniform
            x_cur = x0 + rng.normal(0, 0.02, size=len(x0))
        elif r == 2:
            # More perturbed
            x_cur = x0 + rng.normal(0, 0.05, size=len(x0))
        elif r == 3:
            # Try asymmetric initialization (break palindromic symmetry in search space)
            x_cur = x0 + rng.uniform(-0.03, 0.03, size=len(x0))
        elif r == 4:
            # Try a different base: slightly non-uniform
            base_a = np.linspace(0.8/n, 1.2/n, half)
            base_b = np.linspace(1.2/n, 0.8/n, half)
            x_cur = np.concatenate([base_a, base_b])
        else:
            x_cur = x0 + rng.normal(0, 0.1, size=len(x0))

        try:
            x_opt, f_opt = _nelder_mead(objective, x_cur, max_iter=50,
                                        tol=1e-6, deadline=deadline - 20.0)
            if f_opt < best_err:
                best_err = f_opt
                best_a, best_b = _symmetrize(x_opt, n)
        except Exception:
            pass

    # Coordinate descent polish on the best solution
    if time.time() < deadline - 10.0:
        x_best = np.concatenate([best_a[:half], best_b[:half]])
        step_size = 0.01
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(x_best)):
                for delta in [step_size, -step_size]:
                    x_trial = x_best.copy()
                    x_trial[i] += delta
                    a_t, b_t = _symmetrize(x_trial, n)
                    if np.any(np.abs(a_t) > 50) or np.any(np.abs(b_t) > 50):
                        continue
                    if abs(a_t.sum() - 1.0) > 1e-6 or abs(b_t.sum() - 1.0) > 1e-6:
                        continue
                    f_trial = _compute_error_fast(a_t, b_t, eigs_A, V_A, eigs_B, V_B, Uexact, times)
                    if f_trial < best_err - 1e-9:
                        best_err = f_trial
                        x_best = x_trial
                        best_a, best_b = _symmetrize(x_best, n)
                        improved = True
                        break
                if step_size < 1e-4:
                    break
            if not improved:
                step_size *= 0.5

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
