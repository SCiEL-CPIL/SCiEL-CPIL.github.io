import numpy as np
import time
import sys

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Build the Hamiltonians
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    Jz = 1.0
    Jxy = delta * Jz

    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)

    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)

    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)

    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term

    # Precompute eigendecompositions for A, B, and A+B
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_AB, vecs_AB = np.linalg.eigh(A + B)

    times = [0.25, 0.5, 1.0]

    # Precompute exp(-i * t * vals) for each time and each matrix
    expA = {}
    expB = {}
    expAB = {}
    for t in times:
        expA[t] = np.exp(-1j * t * vals_A)
        expB[t] = np.exp(-1j * t * vals_B)
        expAB[t] = np.exp(-1j * t * vals_AB)

    def expm_A(t, a_k):
        """exp(-i * a_k * t * A)"""
        return vecs_A @ np.diag(np.exp(-1j * a_k * t * vals_A)) @ vecs_A.conj().T

    def expm_B(t, b_k):
        """exp(-i * b_k * t * B)"""
        return vecs_B @ np.diag(np.exp(-1j * b_k * t * vals_B)) @ vecs_B.conj().T

    def expm_AB(t):
        """exp(-i * t * (A+B))"""
        return vecs_AB @ np.diag(expAB[t]) @ vecs_AB.conj().T

    def compute_error(a, b, t):
        """Compute spectral norm of U(t) - exp(-i t (A+B))"""
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            U = expm_B(t, b[k]) @ expm_A(t, a[k]) @ U
        U_exact = expm_AB(t)
        diff = U - U_exact
        # Use power iteration for spectral norm (faster than full SVD for single value)
        # Actually, for 64x64, SVD is fine. Use svd.
        s = np.linalg.svd(diff, compute_uv=False)
        return s[0]

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        Uses penalty for sum constraint.
        """
        if time.time() > deadline:
            return 1e10

        a_full = params[:12]
        b_full = params[12:24]

        # Penalty for sum constraint
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        penalty = 1e4 * (a_sum - 1.0) ** 2 + 1e4 * (b_sum - 1.0) ** 2

        total_err = 0.0
        for t in times:
            total_err += compute_error(a_full, b_full, t)
        return total_err / len(times) + penalty

    # Known good initial guesses based on Trotter/Suzuki-Trotter orders
    # 2nd order: a = [1/2, 0, ..., 0, 1/2], b = [0, ..., 1, ..., 0]
    # For 12 stages, use a symmetric 4th-order-like pattern
    def make_initial_guesses():
        guesses = []
        
        # Uniform
        guesses.append(np.ones(24) / 12.0)
        
        # 2nd-order-like: concentrate in middle
        a2 = np.zeros(12)
        b2 = np.zeros(12)
        a2[5] = 0.5
        a2[6] = 0.5
        b2[5] = 1.0
        guesses.append(np.concatenate([a2, b2]))
        
        # Symmetric spread
        a3 = np.zeros(12)
        b3 = np.zeros(12)
        a3[3] = 0.25
        a3[4] = 0.25
        a3[7] = 0.25
        a3[8] = 0.25
        b3[3] = 0.25
        b3[4] = 0.25
        b3[7] = 0.25
        b3[8] = 0.25
        guesses.append(np.concatenate([a3, b3]))
        
        # Asymmetric: front-loaded
        a4 = np.zeros(12)
        b4 = np.zeros(12)
        a4[1] = 0.3
        a4[2] = 0.3
        a4[3] = 0.2
        a4[4] = 0.2
        b4[1] = 0.25
        b4[2] = 0.25
        b4[3] = 0.25
        b4[4] = 0.25
        guesses.append(np.concatenate([a4, b4]))
        
        # Random with seed
        np.random.seed(123)
        for _ in range(3):
            a_r = np.abs(np.random.randn(12))
            b_r = np.abs(np.random.randn(12))
            a_r /= a_r.sum()
            b_r /= b_r.sum()
            guesses.append(np.concatenate([a_r, b_r]))
        
        return guesses

    best_x = None
    best_obj = np.inf

    # Try each initial guess with L-BFGS-B
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except ImportError:
        has_scipy = False

    guesses = make_initial_guesses()

    for gi, x0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break

        # Normalize initial guess
        a0 = x0[:12]
        b0 = x0[12:24]
        if a0.sum() != 0:
            a0 = a0 / a0.sum()
        else:
            a0 = np.ones(12) / 12
        if b0.sum() != 0:
            b0 = b0 / b0.sum()
        else:
            b0 = np.ones(12) / 12
        x0 = np.concatenate([a0, b0])

        obj0 = objective(x0)
        if obj0 < best_obj:
            best_obj = obj0
            best_x = x0.copy()

        if not has_scipy:
            continue

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

        # Also try SLSQP for a couple of guesses
        if gi < 3:
            try:
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    options={'maxiter': 150, 'ftol': 1e-10}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

    # Final refinement: small random perturbations around best
    np.random.seed(999)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        x_try = best_x + 0.02 * np.random.randn(24)
        obj_try = objective(x_try)
        if obj_try < best_obj:
            best_obj = obj_try
            best_x = x_try.copy()

    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]

    # Clip to [-50, 50]
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)

    # Renormalize to sum to 1
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    else:
        a_full = np.ones(12) / 12
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum
    else:
        b_full = np.ones(12) / 12

    return [float(x) for x in a_full], [float(x) for x in b_full]
