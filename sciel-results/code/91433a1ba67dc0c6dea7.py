"""Optimized product-formula coefficients for 6-site XXZ Heisenberg chain.
Uses a model-and-optimize approach with a tight runtime budget:
  1. Build A (odd bonds + half field) and B (even bonds + half field).
  2. Eigendecompose A, B, H=A+B once.
  3. Precompute exp(-i t A) and exp(-i t B) on a grid of time values.
  4. Optimize 12-stage coefficients via differential_evolution with a
     symmetric parameterization to cut the search space in half.
  5. Polish with L-BFGS-B for final refinement.
All within a strict wall-clock deadline."""

import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

def _remaining():
    return _DEADLINE - time.time()

def coefficients(stages):
    """Return (a, b) lists of length `stages` (12) with sum(a)=sum(b)=1,
    each element in [-50, 50], that minimize the mean spectral-norm error
    of the product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    vs exp(-i t (A+B)) at t = 0.25, 0.5, 1.0.
    """
    if stages != 12:
        # Fallback: simple uniform split
        n = max(1, stages)
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b

    try:
        import scipy.linalg as sla
        from scipy.optimize import differential_evolution, minimize
    except ImportError:
        return [1.0/12]*12, [1.0/12]*12

    # ---- Build the Hamiltonian ----
    N = 6
    dim = 1 << N  # 64
    aniso = 0.8
    field = 0.3

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def tensor(*ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    def site_op(site, mat):
        ops = []
        for i in range(N):
            ops.append(mat if i == site else I2)
        return tensor(*ops)

    def pair_op(i, j, mat_i, mat_j):
        ops = []
        for k in range(N):
            if k == i:
                ops.append(mat_i)
            elif k == j:
                ops.append(mat_j)
            else:
                ops.append(I2)
        return tensor(*ops)

    # Even bonds: 0-1, 2-3, 4-5
    H_evens = (pair_op(0, 1, sx, sx) + pair_op(0, 1, sy, sy) + aniso * pair_op(0, 1, sz, sz)
               + pair_op(2, 3, sx, sx) + pair_op(2, 3, sy, sy) + aniso * pair_op(2, 3, sz, sz)
               + pair_op(4, 5, sx, sx) + pair_op(4, 5, sy, sy) + aniso * pair_op(4, 5, sz, sz))

    # Odd bonds: 1-2, 3-4
    H_odds = (pair_op(1, 2, sx, sx) + pair_op(1, 2, sy, sy) + aniso * pair_op(1, 2, sz, sz)
              + pair_op(3, 4, sx, sx) + pair_op(3, 4, sy, sy) + aniso * pair_op(3, 4, sz, sz))

    # Field split evenly
    H_field_half = -field / 2.0 * sum(site_op(i, sz) for i in range(N))

    A_mat = H_odds + H_field_half
    B_mat = H_evens + H_field_half
    H_total = A_mat + B_mat

    # ---- Eigendecompositions ----
    evals_A, evecs_A = sla.eig(A_mat)
    evals_B, evecs_B = sla.eig(B_mat)
    evals_H, evecs_H = sla.eig(H_total)

    inv_evecs_A = sla.inv(evecs_A)
    inv_evecs_B = sla.inv(evecs_B)
    inv_evecs_H = sla.inv(evecs_H)

    eval_times = [0.25, 0.5, 1.0]

    def exp_negi_t_mat(evals, evecs, inv_evecs, t):
        phases = np.exp(-1j * t * evals)
        return (evecs * phases) @ inv_evecs

    # Precompute exact propagators
    exact_prop = [exp_negi_t_mat(evals_H, evecs_H, inv_evecs_H, t) for t in eval_times]

    # Precompute exp(-i t A) and exp(-i t B) on a fine grid for speed
    # We'll use a grid of time values that covers the range of a_k * t and b_k * t
    # a_k, b_k can be in roughly [-5, 5], t in [0.25, 1.0], so a_k*t in [-5, 5]
    # Use a grid from -5 to 5 with fine spacing
    grid_times = np.linspace(-5.0, 5.0, 1001)
    expA_grid = np.array([exp_negi_t_mat(evals_A, evecs_A, inv_evecs_A, t) for t in grid_times])
    expB_grid = np.array([exp_negi_t_mat(evals_B, evecs_B, inv_evecs_B, t) for t in grid_times])

    def exp_lookup(grid, grid_times, t_val):
        """Interpolate exp(-i t_val M) from precomputed grid."""
        idx = int(np.searchsorted(grid_times, t_val))
        if idx == 0:
            return grid[0]
        if idx >= len(grid_times):
            return grid[-1]
        t0 = grid_times[idx - 1]
        t1 = grid_times[idx]
        alpha = (t_val - t0) / (t1 - t0)
        return (1 - alpha) * grid[idx - 1] + alpha * grid[idx]

    def compute_error_fast(a, b):
        """Compute mean spectral-norm error using precomputed grids."""
        error_sum = 0.0
        I = np.eye(dim, dtype=complex)
        for idx, t in enumerate(eval_times):
            U = I.copy()
            for k in range(12):
                Ea = exp_lookup(expA_grid, grid_times, a[k] * t)
                Eb = exp_lookup(expB_grid, grid_times, b[k] * t)
                U = Eb @ Ea @ U
            diff = U - exact_prop[idx]
            # Spectral norm via SVD
            sn = np.linalg.svd(diff, compute_uv=False)[0]
            error_sum += sn
        return error_sum / 3.0

    # ---- Optimization ----
    # Symmetric parameterization: a_k = a_{13-k}, b_k = b_{13-k}
    # Free params: a_1..a_6, b_1..b_6 (12 values)
    # Constraint: sum(a) = 1, sum(b) = 1
    # With symmetry: 2*(a1+a2+a3+a4+a5+a6) = 1 => a1+...+a6 = 0.5
    # Similarly for b.
    # Reparameterize: use 5 free values + 1 determined by constraint.
    # a_6 = 0.5 - a1 - a2 - a3 - a4 - a5
    # b_6 = 0.5 - b1 - b2 - b3 - b4 - b5

    def reparam_sym(x):
        """x is 10 values: a1..a5, b1..b5.
        Returns (a, b) each of length 12 with symmetry and sum=1."""
        a1, a2, a3, a4, a5 = x[:5]
        a6 = 0.5 - a1 - a2 - a3 - a4 - a5
        b1, b2, b3, b4, b5 = x[5:10]
        b6 = 0.5 - b1 - b2 - b3 - b4 - b5

        a = np.array([a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1])
        b = np.array([b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1])

        # Clip to [-50, 50]
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)

        # Renormalize if needed (should preserve symmetry approximately)
        sa = np.sum(a)
        sb = np.sum(b)
        if abs(sa) > 1e-15:
            a = a / sa
        if abs(sb) > 1e-15:
            b = b / sb

        return a, b

    def objective_sym(x):
        a, b = reparam_sym(x)
        return compute_error_fast(a, b)

    # Initial guess: uniform
    x_init = np.full(10, 1.0/12)
    # Adjust: for uniform a_k=1/12, symmetry gives a1..a6 = 1/12 each, sum first 6 = 6/12=0.5 ✓

    bounds = [(-3, 3)] * 10

    best_x = x_init.copy()
    best_err = objective_sym(x_init)

    if _remaining() > 40:
        try:
            result = differential_evolution(
                objective_sym,
                bounds=bounds,
                seed=42,
                maxiter=150,
                tol=1e-12,
                popsize=12,
                polish=True,
                init='latinhypercube',
                mutation=(0.3, 1.0),
                recombination=0.8,
                seed=42,
            )
            if result.x is not None:
                err = objective_sym(result.x)
                if err < best_err:
                    best_err = err
                    best_x = result.x.copy()
        except Exception:
            pass

    # Polish with L-BFGS-B
    if _remaining() > 15:
        try:
            res = minimize(
                objective_sym,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 5000},
            )
            if res.x is not None:
                err = objective_sym(res.x)
                if err < best_err:
                    best_err = err
                    best_x = res.x.copy()
        except Exception:
            pass

    a, b = reparam_sym(best_x)

    # Final safety
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        a = np.ones(12) / 12
        b = np.ones(12) / 12

    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    a = a / np.sum(a)
    b = b / np.sum(b)

    return list(a), list(b)
