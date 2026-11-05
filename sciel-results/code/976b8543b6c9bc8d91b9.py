"""Optimized product-formula coefficients for 6-site XXZ Heisenberg chain.
Uses a model-and-optimize approach: build A and B matrices, then optimize
the 12-stage splitting coefficients via scipy to minimize the mean spectral
norm error at t = 0.25, 0.5, 1.0."""

import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize

_START = time.time()
DEADLINE = _START + 130.0  # safety margin before 150s kill


def _build_matrices():
    """Build A (odd bonds + half field) and B (even bonds + half field) for
    a 6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    dim = 2 ** N

    # Spin operators
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2

    def site_op(op, site):
        """Apply op on a specific site (0-indexed from left)."""
        result = np.array([[1]], dtype=complex)
        for s in range(N):
            if s == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2, dtype=complex))
        return result

    def pair_op(i, j):
        """Build S_i^+ S_j^- + S_i^- S_j^+ + anisotropy * S_i^z S_j^z."""
        sp = (sx + 1j * sy)
        sm = (sx - 1j * sy)
        # S_i^+ S_j^-
        term1 = np.zeros((dim, dim), dtype=complex)
        for a in range(dim):
            for b in range(dim):
                # Apply S_i^+ on site i, S_j^- on site j
                pass
        # Better: build via kron
        def kron_op(op_i, op_j):
            result = np.array([[1]], dtype=complex)
            for s in range(N):
                if s == i:
                    result = np.kron(result, op_i)
                elif s == j:
                    result = np.kron(result, op_j)
                else:
                    result = np.kron(result, np.eye(2, dtype=complex))
            return result

        term1 = kron_op(sp, sm)
        term2 = kron_op(sm, sp)
        term3 = anisotropy * kron_op(sz, sz)
        return term1 + term2 + term3

    # Field term: sum of sz on each site, split half into A and half into B
    field_op = np.zeros((dim, dim), dtype=complex)
    for s in range(N):
        field_op += site_op(sz, s)

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    # Odd bonds: (0,1), (2,3), (4,5)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        A += pair_op(i, j)
    A += 0.5 * field_op

    # Even bonds: (1,2), (3,4)
    for i, j in [(1, 2), (3, 4)]:
        B += pair_op(i, j)
    B += 0.5 * field_op

    return A, B


def _compute_error(coeffs, A, B, times):
    """Compute mean spectral norm error for given coefficients."""
    a, b = coeffs
    total_err = 0.0
    for t in times:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            U = U @ expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A)
        U_exact = expm(-1j * t * (A + B))
        diff = U - U_exact
        # Spectral norm
        if diff.size > 0:
            sv = np.linalg.svd(diff, compute_uv=False)
            err = sv[0]
        else:
            err = 0.0
        total_err += err
    return total_err / len(times)


def coefficients(stages):
    """Return (a, b) lists of length `stages` (12) with sum(a)=sum(b)=1,
    each element in [-50, 50], that minimize the mean spectral norm error."""
    A, B = _build_matrices()
    times = [0.25, 0.5, 1.0]
    dim = A.shape[0]

    # Precompute exact propagators
    exact_propagators = [expm(-1j * t * (A + B)) for t in times]

    def objective(params):
        """Objective: mean spectral norm error. params = [a_1..a_12, b_1..b_12]."""
        a = params[:stages]
        b = params[stages:2 * stages]
        # Enforce sum constraints via soft penalty (optimizers handle it better
        # with constraints, but we add a large penalty for violations)
        sum_a = sum(a)
        sum_b = sum(b)
        penalty = 100.0 * (sum_a - 1.0) ** 2 + 100.0 * (sum_b - 1.0) ** 2

        total_err = 0.0
        for idx, t in enumerate(times):
            U = np.eye(dim, dtype=complex)
            for k in range(stages):
                U = U @ expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A)
            diff = U - exact_propagators[idx]
            sv = np.linalg.svd(diff, compute_uv=False)
            total_err += sv[0]
        return total_err / len(times) + penalty

    # Initial guess: equal split
    x0 = [1.0 / stages] * (2 * stages)

    # Bounds: each coefficient in [-50, 50]
    bounds = [(-50, 50)] * (2 * stages)

    best_result = None
    best_score = float('inf')

    # Try differential evolution with limited iterations
    try:
        if time.time() < DEADLINE - 30:
            de_result = differential_evolution(
                objective, bounds, maxiter=50, popsize=15,
                tol=1e-6, seed=42, polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x
    except Exception:
        pass

    # Refine with L-BFGS-B
    try:
        if best_result is None:
            start = x0
        else:
            start = best_result

        if time.time() < DEADLINE - 10:
            res = minimize(
                objective, start, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
    except Exception:
        pass

    # Another DE run with different seed if time permits
    try:
        if time.time() < DEADLINE - 20:
            de_result2 = differential_evolution(
                objective, bounds, maxiter=30, popsize=10,
                tol=1e-5, seed=123, polish=False,
                init='sobol'
            )
            if de_result2.fun < best_score:
                best_score = de_result2.fun
                best_result = de_result2.x
    except Exception:
        pass

    # Final refinement
    try:
        if best_result is not None and time.time() < DEADLINE - 5:
            res2 = minimize(
                objective, best_result, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-14}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
    except Exception:
        pass

    if best_result is None:
        # Fallback: equal split
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
    else:
        a = list(best_result[:stages])
        b = list(best_result[stages:2 * stages])

        # Enforce sum constraints exactly
        sum_a = sum(a)
        sum_b = sum(b)
        if abs(sum_a) > 1e-15:
            a = [x / sum_a for x in a]
        else:
            a = [1.0 / stages] * stages
        if abs(sum_b) > 1e-15:
            b = [x / sum_b for x in b]
        else:
            b = [1.0 / stages] * stages

        # Clamp to [-50, 50] and re-normalize
        a = [max(-50, min(50, x)) for x in a]
        b = [max(-50, min(50, x)) for x in b]
        sum_a = sum(a)
        sum_b = sum(b)
        if abs(sum_a) > 1e-15:
            a = [x / sum_a for x in a]
        if abs(sum_b) > 1e-15:
            b = [x / sum_b for x in b]

    # Final check: ensure finite values
    a = [float(x) if np.isfinite(x) else 0.0 for x in a]
    b = [float(x) if np.isfinite(x) else 0.0 for x in b]

    return a, b
