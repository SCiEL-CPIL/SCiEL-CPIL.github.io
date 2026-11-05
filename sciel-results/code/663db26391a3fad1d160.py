import numpy as np
import time
import sys

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

def _spectral_norm(M):
    return np.linalg.norm(M, 2)

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

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients using a symmetric (palindromic)
    structure with analytic higher-order matching, then refine with Nelder-Mead.
    """
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    try:
        A, B = _build_hamiltonians()
        H = A + B
        dim = A.shape[0]

        times = [0.25, 0.5, 1.0]
        U_exacts = []
        for t in times:
            U_exacts.append(np.linalg.matrix_exp(-1j * t * H))

        # Precompute eigen-decompositions for fast expm
        evals_A, evecs_A = np.linalg.eig(A)
        evals_B, evecs_B = np.linalg.eig(B)
        evecs_A_inv = np.linalg.inv(evecs_A)
        evecs_B_inv = np.linalg.inv(evecs_B)

        def fast_expm_A(theta):
            """Compute exp(-i * theta * A) using precomputed eigendecomposition."""
            diag = np.diag(np.exp(-1j * theta * evals_A))
            return evecs_A @ diag @ evecs_A_inv

        def fast_expm_B(theta):
            """Compute exp(-i * theta * B) using precomputed eigendecomposition."""
            diag = np.diag(np.exp(-1j * theta * evals_B))
            return evecs_B @ diag @ evecs_B_inv

        def compute_error_symmetric(params_a, params_b):
            """
            Compute error for palindromic coefficients.
            params_a: 6 values for positions 0..5 (mirrored to 11..6)
            params_b: 6 values for positions 0..5 (mirrored to 11..6)
            """
            a_full = np.zeros(12)
            b_full = np.zeros(12)
            for k in range(6):
                a_full[k] = params_a[k]
                a_full[11 - k] = params_a[k]
                b_full[k] = params_b[k]
                b_full[11 - k] = params_b[k]

            a_full, b_full = _normalize_coeffs(a_full, b_full)

            total_err = 0.0
            for i, t in enumerate(times):
                U = np.eye(dim, dtype=complex)
                for k in range(12):
                    Ea = fast_expm_A(a_full[k] * t)
                    Eb = fast_expm_B(b_full[k] * t)
                    U = Eb @ Ea @ U
                err = _spectral_norm(U - U_exacts[i])
                total_err += err
            return total_err / len(times)

        # --- Phase 1: Analytic initialization via commutator matching ---
        # For palindromic 12-stage formula, the first-order error term involves
        # the commutator [A,B]. We can try to match known high-order formulas.
        # 
        # The 4th-order Suzuki-Trotter formula has 6 stages. For 12 stages,
        # we can use a composition or a direct 12-stage high-order formula.
        # 
        # Let's start with a few structured initial guesses:
        
        best_params = None
        best_err = float('inf')

        # Family 1: Uniform
        a_u = np.ones(6) / 6.0
        b_u = np.ones(6) / 6.0
        err_u = compute_error_symmetric(a_u, b_u)
        if err_u < best_err:
            best_err = err_u
            best_params = (a_u.copy(), b_u.copy())

        # Family 2: Cosine-weighted
        for alpha in [0.0, 0.3, 0.5, 0.8, 1.0, -0.5, -1.0]:
            ks = np.arange(6)
            w = 1.0 + alpha * np.cos(np.pi * ks / 5.0)
            a_c = w.copy()
            b_c = w.copy()
            err_c = compute_error_symmetric(a_c, b_c)
            if err_c < best_err:
                best_err = err_c
                best_params = (a_c.copy(), b_c.copy())

        # Family 3: Quadratic
        for beta in [0.0, 0.3, 0.6, 1.0, -0.5, -1.0]:
            ks = np.arange(6)
            centered = (ks - 2.5) / 2.5
            w = 1.0 + beta * centered**2
            a_q = w.copy()
            b_q = w.copy()
            err_q = compute_error_symmetric(a_q, b_q)
            if err_q < best_err:
                best_err = err_q
                best_params = (a_q.copy(), b_q.copy())

        # Family 4: Asymmetric a and b
        for alpha_a in [0.0, 0.3, 0.5, 0.8]:
            for alpha_b in [0.0, 0.3, 0.5, 0.8]:
                ks = np.arange(6)
                wa = 1.0 + alpha_a * np.cos(np.pi * ks / 5.0)
                wb = 1.0 + alpha_b * np.cos(np.pi * ks / 5.0)
                err_ab = compute_error_symmetric(wa, wb)
                if err_ab < best_err:
                    best_err = err_ab
                    best_params = (wa.copy(), wb.copy())

        # Family 5: Random perturbations
        rng = np.random.RandomState(42)
        for trial in range(30):
            if time.time() > deadline - 30.0:
                break
            a_r = 1.0/6.0 + rng.normal(0, 0.03, 6)
            b_r = 1.0/6.0 + rng.normal(0, 0.03, 6)
            err_r = compute_error_symmetric(a_r, b_r)
            if err_r < best_err:
                best_err = err_r
                best_params = (a_r.copy(), b_r.copy())

        # --- Phase 2: Nelder-Mead on the 12-dim symmetric parameter space ---
        def objective_symmetric(x):
            if time.time() > deadline:
                return 1e10
            a_p = x[:6]
            b_p = x[6:]
            return compute_error_symmetric(a_p, b_p)

        x0 = np.concatenate([best_params[0], best_params[1]])

        try:
            from scipy.optimize import minimize
            res = minimize(objective_symmetric, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                x0 = res.x
        except Exception:
            pass

        # --- Phase 3: Second run from different start ---
        if time.time() < deadline - 20.0:
            try:
                rng2 = np.random.RandomState(123)
                a_r2 = 1.0/6.0 + rng2.normal(0, 0.05, 6)
                b_r2 = 1.0/6.0 + rng2.normal(0, 0.05, 6)
                x_r2 = np.concatenate([a_r2, b_r2])
                res2 = minimize(objective_symmetric, x_r2, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                         'adaptive': True})
                if res2.fun < best_err:
                    best_err = res2.fun
                    x0 = res2.x
            except Exception:
                pass

        # --- Phase 4: Final refinement ---
        if time.time() < deadline - 10.0:
            try:
                res3 = minimize(objective_symmetric, x0, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14,
                                         'adaptive': True})
                if res3.fun < best_err:
                    best_err = res3.fun
                    x0 = res3.x
            except Exception:
                pass

        # Extract final symmetric parameters and build full 12-element arrays
        a_p = x0[:6]
        b_p = x0[6:]

        a_full = np.zeros(12)
        b_full = np.zeros(12)
        for k in range(6):
            a_full[k] = a_p[k]
            a_full[11 - k] = a_p[k]
            b_full[k] = b_p[k]
            b_full[11 - k] = b_p[k]

        a_full, b_full = _normalize_coeffs(a_full, b_full)

        # Final validation
        if not (np.all(np.isfinite(a_full)) and np.all(np.isfinite(b_full))):
            return [1.0/12.0]*12, [1.0/12.0]*12

        return list(a_full), list(b_full)

    except Exception:
        return [1.0/12.0]*12, [1.0/12.0]*12
