"""Improved product-formula coefficients via numerical optimization."""
import numpy as np
import time
from scipy.linalg import expm

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    anisotropy = 0.8
    field = 0.3
    J = 1.0  # coupling constant

    # Build bond terms
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    # For a 6-site open chain, bonds are (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)

    def bond_term(i, j):
        """Build the XXZ interaction term for bond (i, j)."""
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        sx_sx = kron_all(ops)

        ops[i] = sy
        ops[j] = sy
        sy_sy = kron_all(ops)

        ops[i] = sz
        ops[j] = sz
        sz_sz = kron_all(ops)

        return J * (sx_sx + sy_sy + anisotropy * sz_sz) / 2

    def field_term(site):
        """Build the uniform field term for a single site."""
        ops = [I2] * N
        ops[site] = sz
        return field * kron_all(ops) / 2

    # A: odd bonds + half field
    A = np.zeros((dim, dim), dtype=complex)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        A += bond_term(i, j)
    for site in range(N):
        A += 0.5 * field_term(site)

    # B: even bonds + half field
    B = np.zeros((dim, dim), dtype=complex)
    for i, j in [(1, 2), (3, 4)]:
        B += bond_term(i, j)
    for site in range(N):
        B += 0.5 * field_term(site)

    return A, B

def _spectral_norm_error(a, b, A, B, t_values):
    """Compute mean spectral norm error over t_values."""
    dim = A.shape[0]
    total_error = 0.0
    for t in t_values:
        # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = expm(-1j * b[k] * t * B) @ U
            U = expm(-1j * a[k] * t * A) @ U

        # Exact propagator
        H = A + B
        U_exact = expm(-1j * t * H)

        # Spectral norm of error
        err = U - U_exact
        # For Hermitian matrices, spectral norm = max singular value
        # Use svd for general case
        s = np.linalg.svd(err, compute_uv=False)
        total_error += s[0]
    return total_error / len(t_values)

def coefficients(stages):
    """Optimize product formula coefficients via gradient-free optimization."""
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin

    # Build Hamiltonians
    A, B = _build_hamiltonians()
    t_values = [0.25, 0.5, 1.0]

    # Initial guess: uniform split
    a0 = [1.0 / stages] * stages
    b0 = [1.0 / stages] * stages

    # Use scipy's minimize with Nelder-Mead
    from scipy.optimize import minimize

    def objective(x):
        """Objective function: mean spectral norm error."""
        a = list(x[:stages])
        b = list(x[stages:])
        # Enforce sum constraints via projection
        sum_a = sum(a)
        sum_b = sum(b)
        if abs(sum_a) > 1e-12:
            a = [ai / sum_a for ai in a]
        if abs(sum_b) > 1e-12:
            b = [bi / sum_b for bi in b]
        # Clamp to [-50, 50]
        a = [max(-50, min(50, ai)) for ai in a]
        b = [max(-50, min(50, bi)) for bi in b]
        # Re-normalize after clamping
        sum_a = sum(a)
        sum_b = sum(b)
        if abs(sum_a) > 1e-12:
            a = [ai / sum_a for ai in a]
        if abs(sum_b) > 1e-12:
            b = [bi / sum_b for bi in b]
        return _spectral_norm_error(a, b, A, B, t_values)

    x0 = a0 + b0
    best_x = x0.copy()
    best_val = objective(x0)

    # Try Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    if time.time() > deadline - 30:
        # Not enough time for more optimization, return best so far
        a = list(best_x[:stages])
        b = list(best_x[stages:])
        sum_a = sum(a)
        sum_b = sum(b)
        if abs(sum_a) > 1e-12:
            a = [ai / sum_a for ai in a]
        if abs(sum_b) > 1e-12:
            b = [bi / sum_b for bi in b]
        a = [max(-50, min(50, ai)) for ai in a]
        b = [max(-50, min(50, bi)) for bi in b]
        return a, b

    # Second round: refine with different initial points
    np.random.seed(42)
    for restart in range(3):
        if time.time() > deadline - 20:
            break
        # Perturb the best solution
        x_start = best_x + np.random.randn(len(x0)) * 0.1
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9,
                                       'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

    a = list(best_x[:stages])
    b = list(best_x[stages:])
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-12:
        a = [ai / sum_a for ai in a]
    if abs(sum_b) > 1e-12:
        b = [bi / sum_b for bi in b]
    a = [max(-50, min(50, ai)) for ai in a]
    b = [max(-50, min(50, bi)) for bi in b]
    return a, b
