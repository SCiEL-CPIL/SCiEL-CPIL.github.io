import numpy as np
import time
import sys
import math

def _build_hamiltonians():
    """
    Build the Hamiltonian parts A and B for the 6-site XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    """
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    N = 6
    dim = 2 ** N

    def apply_op(op, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    def apply_two_op(op1, op2, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op1
        ops[k+1] = op2
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    J = 1.0
    delta = 0.8
    h = 0.3

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    field_sum = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        field_sum += apply_op(Sz, k)

    A += 0.5 * h * field_sum
    B += 0.5 * h * field_sum

    for k in range(0, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            A += term

    for k in range(1, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            B += term

    return A, B
    """
    Build the Hamiltonian parts A and B for the 6-site XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    """
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    N = 6
    dim = 2 ** N

    def apply_op(op, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    def apply_two_op(op1, op2, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op1
        ops[k+1] = op2
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    J = 1.0
    delta = 0.8
    h = 0.3

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    field_sum = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        field_sum += apply_op(Sz, k)

    A += 0.5 * h * field_sum
    B += 0.5 * h * field_sum

    for k in range(0, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            A += term

    for k in range(1, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            B += term

    return A, B

def _spectral_norm(M):
    return np.linalg.norm(M, 2)

def _frobenius_norm(M):
    return np.linalg.norm(M, 'fro')

def _normalize_coeffs(a, b):
    """Normalize a and b so that sum(a)=1 and sum(b)=1, preserving relative weights."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(len(a)) / len(a)
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(len(b)) / len(b)
    # Clip to [-50, 50]
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    # Re-normalize after clipping
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb
    return a, b

def _eigh_expm_cache(eigvals, eigvecs, scale):
    """Compute exp(scale * H) given eigendecomposition H = V diag(w) V^{-1}."""
    # For Hermitian: exp(scale*H) = V diag(exp(scale*w)) V^H
    # eigvals, eigvecs from np.linalg.eigh
    scaled = np.exp(scale * eigvals)
    return eigvecs @ np.diag(scaled) @ eigvecs.conj().T

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients using Nelder-Mead
    on the full (non-symmetric) coefficient space with constraint
    sum(a)=sum(b)=1 handled by normalization.
    Uses cached eigen-decompositions for fast expm and smooth Frobenius
    norm surrogate during optimization, with spectral-norm polish at end.
    """
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    try:
        A, B = _build_hamiltonians()
        H = A + B
        dim = A.shape[0]

        # Precompute eigen-decompositions for fast expm
        # A and B are Hermitian, so use eigh
        eigvals_A, eigvecs_A = np.linalg.eigh(A)
        eigvals_B, eigvecs_B = np.linalg.eigh(B)
        eigvals_H, eigvecs_H = np.linalg.eigh(H)

        times = [0.25, 0.5, 1.0]
        U_exacts = []
        for t in times:
            U_exacts.append(_eigh_expm_cache(eigvals_H, eigvecs_H, -1j * t))

        def fast_expm_A(scale):
            """Fast exp(scale * A) using cached eigendecomposition."""
            return _eigh_expm_cache(eigvals_A, eigvecs_A, scale)

        def fast_expm_B(scale):
            """Fast exp(scale * B) using cached eigendecomposition."""
            return _eigh_expm_cache(eigvals_B, eigvecs_B, scale)

        def compute_error_spectral(a, b):
            """Compute mean spectral-norm error for given coefficients."""
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            total_err = 0.0
            for i, t in enumerate(times):
                U = np.eye(dim, dtype=complex)
                for k in range(12):
                    Ea = fast_expm_A(-1j * a[k] * t)
                    Eb = fast_expm_B(-1j * b[k] * t)
                    U = Eb @ Ea @ U
                err = _spectral_norm(U - U_exacts[i])
                total_err += err
            return total_err / len(times)

        def compute_error_frobenius(a, b):
            """Compute mean Frobenius-norm error (smooth surrogate)."""
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            total_err = 0.0
            for i, t in enumerate(times):
                U = np.eye(dim, dtype=complex)
                for k in range(12):
                    Ea = fast_expm_A(-1j * a[k] * t)
                    Eb = fast_expm_B(-1j * b[k] * t)
                    U = Eb @ Ea @ U
                err = _frobenius_norm(U - U_exacts[i])
                total_err += err
            return total_err / len(times)

        def objective_smooth(x):
            """Smooth objective using Frobenius norm for Nelder-Mead."""
            if time.time() > deadline:
                return 1e10
            a = x[:12]
            b = x[12:]
            a_n, b_n = _normalize_coeffs(a, b)
            if not (np.all(np.isfinite(a_n)) and np.all(np.isfinite(b_n))):
                return 1e10
            return compute_error_frobenius(a_n, b_n)

        def objective_spectral(x):
            """Spectral norm objective for final polish."""
            if time.time() > deadline:
                return 1e10
            a = x[:12]
            b = x[12:]
            a_n, b_n = _normalize_coeffs(a, b)
            if not (np.all(np.isfinite(a_n)) and np.all(np.isfinite(b_n))):
                return 1e10
            return compute_error_spectral(a_n, b_n)

        # --- Phase 1: Grid-search warm start over structured families ---
        best_x0 = None
        best_err_frob = float('inf')
        best_err_spec = float('inf')

        # Family 1: Uniform (baseline)
        a_u = np.ones(12) / 12.0
        b_u = np.ones(12) / 12.0
        x_u = np.concatenate([a_u, b_u])
        err_u_frob = compute_error_frobenius(a_u, b_u)
        err_u_spec = compute_error_spectral(a_u, b_u)
        if err_u_frob < best_err_frob:
            best_err_frob = err_u_frob
            best_x0 = x_u.copy()
        best_err_spec = err_u_spec

        # Family 2: Symmetric with endpoint weighting
        for alpha in [0.0, 0.2, 0.5, 0.8, 1.0, -0.5, -1.0]:
            if time.time() > deadline - 60.0:
                break
            ks = np.arange(12)
            a_w = 1.0 + alpha * np.cos(np.pi * ks / 11.0)
            b_w = 1.0 + alpha * np.cos(np.pi * ks / 11.0)
            a_n, b_n = _normalize_coeffs(a_w, b_w)
            x_w = np.concatenate([a_n, b_n])
            err_w_frob = compute_error_frobenius(a_n, b_n)
            if err_w_frob < best_err_frob:
                best_err_frob = err_w_frob
                best_x0 = x_w.copy()

        # Family 3: Quadratic profile
        for beta in [0.0, 0.3, 0.6, 1.0, -0.5]:
            if time.time() > deadline - 60.0:
                break
            ks = np.arange(12)
            centered = (ks - 5.5) / 5.5
            a_q = 1.0 + beta * centered**2
            b_q = 1.0 + beta * centered**2
            a_n, b_n = _normalize_coeffs(a_q, b_q)
            x_q = np.concatenate([a_n, b_n])
            err_q_frob = compute_error_frobenius(a_n, b_n)
            if err_q_frob < best_err_frob:
                best_err_frob = err_q_frob
                best_x0 = x_q.copy()

        # Family 4: Asymmetric a and b profiles
        for alpha_a in [0.0, 0.5, 1.0]:
            for alpha_b in [0.0, 0.5, 1.0]:
                if time.time() > deadline - 60.0:
                    break
                ks = np.arange(12)
                a_w = 1.0 + alpha_a * np.cos(np.pi * ks / 11.0)
                b_w = 1.0 + alpha_b * np.cos(np.pi * ks / 11.0)
                a_n, b_n = _normalize_coeffs(a_w, b_w)
                x_w = np.concatenate([a_n, b_n])
                err_w_frob = compute_error_frobenius(a_n, b_n)
                if err_w_frob < best_err_frob:
                    best_err_frob = err_w_frob
                    best_x0 = x_w.copy()

        # Family 5: Random perturbations of uniform
        rng = np.random.RandomState(42)
        for trial in range(30):
            if time.time() > deadline - 50.0:
                break
            a_r = 1.0 / 12.0 + rng.normal(0, 0.03, 12)
            b_r = 1.0 / 12.0 + rng.normal(0, 0.03, 12)
            a_n, b_n = _normalize_coeffs(a_r, b_r)
            x_r = np.concatenate([a_n, b_n])
            err_r_frob = compute_error_frobenius(a_n, b_n)
            if err_r_frob < best_err_frob:
                best_err_frob = err_r_frob
                best_x0 = x_r.copy()

        # --- Phase 2: Nelder-Mead local optimization on smooth surrogate ---
        try:
            from scipy.optimize import minimize
            res = minimize(objective_smooth, best_x0, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < best_err_frob:
                best_err_frob = res.fun
                best_x0 = res.x
        except Exception:
            pass

        # --- Phase 3: Second Nelder-Mead from a different random start ---
        if time.time() < deadline - 30.0:
            try:
                rng2 = np.random.RandomState(123)
                a_r2 = 1.0 / 12.0 + rng2.normal(0, 0.06, 12)
                b_r2 = 1.0 / 12.0 + rng2.normal(0, 0.06, 12)
                a_n2, b_n2 = _normalize_coeffs(a_r2, b_r2)
                x_r2 = np.concatenate([a_n2, b_n2])
                res2 = minimize(objective_smooth, x_r2, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                         'adaptive': True})
                if res2.fun < best_err_frob:
                    best_err_frob = res2.fun
                    best_x0 = res2.x
            except Exception:
                pass

        # --- Phase 4: Final refinement on smooth surrogate ---
        if time.time() < deadline - 20.0:
            try:
                res3 = minimize(objective_smooth, best_x0, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14,
                                         'adaptive': True})
                if res3.fun < best_err_frob:
                    best_err_frob = res3.fun
                    best_x0 = res3.x
            except Exception:
                pass

        # --- Phase 5: Spectral-norm polish pass ---
        # Use the actual spectral norm objective for final local optimization
        if time.time() < deadline - 10.0:
            try:
                from scipy.optimize import minimize
                # Evaluate spectral error at current best
                a_cur = best_x0[:12]
                b_cur = best_x0[12:]
                a_cur_n, b_cur_n = _normalize_coeffs(a_cur, b_cur)
                spec_err_cur = compute_error_spectral(a_cur_n, b_cur_n)
                
                res_spec = minimize(objective_spectral, best_x0, method='Nelder-Mead',
                                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                             'adaptive': True})
                if res_spec.fun < spec_err_cur:
                    best_x0 = res_spec.x
                    spec_err_cur = res_spec.fun
            except Exception:
                pass

        # --- Phase 6: Tiny perturbation search around best point ---
        if time.time() < deadline - 5.0:
            try:
                a_best = best_x0[:12]
                b_best = best_x0[12:]
                a_best, b_best = _normalize_coeffs(a_best, b_best)
                best_spec = compute_error_spectral(a_best, b_best)
                
                rng3 = np.random.RandomState(999)
                for trial in range(20):
                    if time.time() > deadline - 3.0:
                        break
                    a_p = a_best + rng3.normal(0, 0.005, 12)
                    b_p = b_best + rng3.normal(0, 0.005, 12)
                    a_p, b_p = _normalize_coeffs(a_p, b_p)
                    err_p = compute_error_spectral(a_p, b_p)
                    if err_p < best_spec:
                        best_spec = err_p
                        best_x0 = np.concatenate([a_p, b_p])
            except Exception:
                pass

        # Extract and normalize final coefficients
        a_final = best_x0[:12]
        b_final = best_x0[12:]
        a_final, b_final = _normalize_coeffs(a_final, b_final)

        # Final validation
        if not (np.all(np.isfinite(a_final)) and np.all(np.isfinite(b_final))):
            return [1.0/12.0]*12, [1.0/12.0]*12

        return list(a_final), list(b_final)

    except Exception:
        return [1.0/12.0]*12, [1.0/12.0]*12
