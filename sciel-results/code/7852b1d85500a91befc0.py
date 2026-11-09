"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# ---------------------------------------------------------------------------
# Symmetry helpers
# ---------------------------------------------------------------------------
def _enforce_symmetry(cell, size):
    """Enforce 4-fold rotational and mirror symmetry about center."""
    sym = np.zeros((size, size), dtype=int)
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            val = cell[i][j]
            sym[i][j] = val
            sym[size - 1 - i][j] = val
            sym[i][size - 1 - j] = val
            sym[size - 1 - i][size - 1 - j] = val
    return sym


# ---------------------------------------------------------------------------
# Plane-wave band solver (fast, matches evaluator operator for TM)
# ---------------------------------------------------------------------------
def _fourier_eps_inv(cell, size, Gx, Gy):
    """Exact Fourier coefficients of 1/epsilon for the given G vectors.

    f_G = (1/N^2) * sum_{i,j} f(i,j) * exp(-i 2pi (Gx*i/N + Gy*j/N))
    """
    f = np.array(cell, dtype=float)
    f[f > 0.5] = 1.0 / 12.0
    f[f <= 0.5] = 1.0
    # f(i,j) * exp(-i2pi Gx i/N) * exp(-i2pi Gy j/N)
    # Use einsum over (i,j,Gx) and (i,j,Gy)
    i_idx = np.arange(size)
    phase_x = np.exp(-1j * 2.0 * np.pi * np.outer(i_idx, Gx) / size)  # (N, nG)
    phase_y = np.exp(-1j * 2.0 * np.pi * np.outer(i_idx, Gy) / size)  # (N, nG)
    # f[i,j] * phase_x[i,ga] * phase_y[j,gb]  -> sum over i,j for each (ga,gb)
    # reshape: f (N,N), phase_x (N,nG), phase_y (N,nG)
    tmp = f[:, :, None] * phase_x[:, None, :]          # (N, N, nGx)
    tmp = tmp * phase_y[None, :, :]                     # (N, N, nGx, nGy)
    coeffs = tmp.sum(axis=(0, 1))                       # (nGx, nGy)
    return coeffs / (size * size)


def _quick_score(cell, size, Npw=13):
    """Fast approximate gap-midgap ratio using a reduced plane-wave basis."""
    half = (Npw - 1) // 2
    g = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(g, g, indexing='ij')
    Gx = Gx.ravel()
    Gy = Gy.ravel()
    nG = len(Gx)

    coeffs = _fourier_eps_inv(cell, size, Gx, Gy)  # (nG, nG)

    # Wave vectors in the irreducible Brillouin zone (square lattice, k in [0,1])
    k_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
                (0.5, 0.0), (0.5, 0.5), (1.0, 0.5),
                (0.25, 0.0), (0.25, 0.25), (0.75, 0.0), (0.75, 0.75)]

    min_omega5 = float('inf')
    max_omega4 = 0.0

    for (kx, ky) in k_points:
        kgx = Gx + kx
        kgy = Gy + ky
        kg2 = kgx * kgx + kgy * kgy  # (nG,)
        # A[g,h] = kg2_g * coeffs[h,g]
        A = np.outer(kg2, coeffs.T)
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(max(eigvals[3], 0.0))
                omega5 = np.sqrt(max(eigvals[4], 0.0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except Exception:
            pass

    if max_omega4 > 0 and min_omega5 > 0:
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return -1.0


# ---------------------------------------------------------------------------
# Level-set parameterization (C4v symmetric low-order Fourier/level-set)
# ---------------------------------------------------------------------------
def _field_from_params(params, size):
    """Evaluate the continuous level-set field on the pixel grid.

    field(x,y) = sum_{m,n} A[m,n] * cos(2pi m (x-0.5)) * cos(2pi n (y-0.5))
    with A[m,n] = A[n,m]  (enforces transpose symmetry about the center).
    cos terms about 0.5 are mirror-symmetric in x and y.
    """
    max_h = 3
    A = np.zeros((max_h + 1, max_h + 1))
    idx = 0
    for m in range(max_h + 1):
        for n in range(m, max_h + 1):
            A[m, n] = params[idx]
            A[n, m] = params[idx]
            idx += 1

    x = (np.arange(size) + 0.5) / size - 0.5
    Xc, Yc = np.meshgrid(x, x, indexing='ij')
    field = np.zeros((size, size))
    for m in range(max_h + 1):
        cm = np.cos(2.0 * np.pi * m * Xc)
        for n in range(max_h + 1):
            if A[m, n] != 0.0:
                cn = np.cos(2.0 * np.pi * n * Yc)
                field += A[m, n] * cm * cn
    return field


def _params_to_cell(params, size):
    field = _field_from_params(params, size)
    cell = np.zeros((size, size), dtype=int)
    cell[field > 0.0] = 1
    return _enforce_symmetry(cell, size)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200.0  # safety margin before the 230 s kill

    random.seed(7)
    np.random.seed(7)

    max_h = 3
    # (m,n) with m<=n
    pairs = [(m, n) for m in range(max_h + 1) for n in range(m, max_h + 1)]
    n_params = len(pairs)

    # --- Phase 1: optimize the continuous level-set coefficients ----------
    # Heuristic initial guess: a central dielectric blob -> A(1,1) positive,
    # A(2,2) negative to sharpen the interface (Chebyshev-like ringing).
    params = np.zeros(n_params)
    idx11 = pairs.index((1, 1))
    idx22 = pairs.index((2, 2))
    params[idx11] = 1.5
    params[idx22] = -1.0

    best_params = params.copy()
    best_cell = _params_to_cell(params, size)
    best_score = _quick_score(best_cell, size, Npw=13)

    # Adaptive step sizes per coefficient
    steps = np.full(n_params, 0.35)

    t1_deadline = deadline - 25.0  # leave time for the pixel polish phase

    while time.time() < t1_deadline:
        p = random.randrange(n_params)
        direction = random.choice([-1.0, 1.0])
        new_params = best_params.copy()
        new_params[p] += direction * steps[p]
        new_cell = _params_to_cell(new_params, size)
        new_score = _quick_score(new_cell, size, Npw=13)
        if new_score > best_score:
            best_params = new_params
            best_cell = new_cell
            best_score = new_score
            steps[p] *= 1.25          # accept -> grow step
            steps = np.clip(steps, 0.02, 2.5)
        else:
            steps[p] *= 0.75          # reject -> shrink step
            steps = np.clip(steps, 0.02, 2.5)

    # --- Phase 2: polish the pixel map with the fast pixel-flip search -----
    best_cell = _enforce_symmetry(best_cell, size)
    best_score = _quick_score(best_cell, size, Npw=13)

    unique_positions = [(i, j) for i in range(size // 2 + 1)
                        for j in range(size // 2 + 1)]
    pos_idx = list(range(len(unique_positions)))
    cell = best_cell.copy()

    while time.time() < deadline - 5.0:
        idx = random.choice(pos_idx)
        i, j = unique_positions[idx]
        old_val = cell[i][j]
        cell[i][j] = 1 - old_val
        cell = _enforce_symmetry(cell, size)
        score = _quick_score(cell, size, Npw=13)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        else:
            cell[i][j] = old_val
            cell = _enforce_symmetry(cell, size)

    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
