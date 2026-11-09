"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_SIZE = 32
_EPS_LOW = 1.0
_EPS_HIGH = 12.0
# local (fast) solver settings
_NPW = 15          # odd plane-wave count per direction -> (2*7+1)^2 = 225 PWs
_NLOW = 5          # number of lowest bands to resolve
_KPTS = 24         # k-points sampled on the irreducible BZ


def _make_cell_from_params(params, size=_SIZE):
    """Build a 4-fold-symmetric 32x32 cell from (cx, cy, r, sign) circles.

    Circles are defined in the upper-right quadrant (x,y in [0,0.5]) and are
    mirrored across both axes, giving a cell invariant under the square
    symmetry group about the centre.
    """
    size = int(size)
    cell = np.zeros((size, size), dtype=np.float64)
    coords = (np.arange(size) + 0.5) / size - 0.5
    x_abs = np.abs(coords)
    X, Y = np.meshgrid(x_abs, x_abs, indexing='ij')
    for (cx, cy, r, sign) in params:
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        mask = (dist2 < r * r).astype(np.float64)
        cell += sign * mask
    cell = (cell > 0).astype(np.float64)
    # enforce exact square symmetry about the cell centre
    cell = (cell
            + cell[::-1].copy()
            + cell[:, ::-1].copy()
            + cell[::-1].copy()[:, ::-1].copy()
            + cell.T
            + cell[::-1].copy().T
            + cell[:, ::-1].copy().T
            + cell[::-1].copy()[:, ::-1].copy().T) / 8.0
    cell = (cell > 0.5).astype(np.float64)
    return cell


def _rod_cell(size=_SIZE, radius=0.2):
    """Textbook centred dielectric rod (deterministic seed / fallback)."""
    coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    return ((X * X + Y * Y) < radius * radius).astype(np.float64)


def _fourier_inv_eps(cell, half):
    """Fourier coefficients of 1/eps on the needed reciprocal grid."""
    size = cell.shape[0]
    inv_eps = np.where(cell > 0.5, 1.0 / _EPS_HIGH, 1.0 / _EPS_LOW)
    x = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x, x, indexing='ij')
    m = np.arange(-half, half + 1)
    Gx = 2 * np.pi * m
    Gy = 2 * np.pi * m
    # F[m,n] = sum_{i,j} inv_eps(i,j) * exp(-i (Gx_m x_i + Gy_n y_j))
    # = (phase_x @ inv_eps @ phase_y.T)
    phase_x = np.exp(-1j * np.outer(Gx, x))          # (nM, size)
    phase_y = np.exp(-1j * np.outer(Gy, x))          # (nM, size)
    F = phase_x @ inv_eps @ phase_y.T
    return F / (size * size)


def _kpoint_set(n=_KPTS):
    """n k-points covering the irreducible triangle Gamma->X->M->Gamma."""
    pts = []
    for t in np.linspace(0, 1, n):
        pts.append((t * np.pi / 2, t * np.pi / 2))                      # Gamma -> X
    for t in np.linspace(0, 1, n):
        pts.append((np.pi / 2 + t * np.pi / 2, np.pi / 2 - t * np.pi / 2))  # X -> M
    for t in np.linspace(0, 1, n):
        pts.append((np.pi - t * np.pi, 0.0))                            # M -> Gamma
    # de-duplicate
    uniq = []
    for kx, ky in pts:
        if all(abs(kx - a) > 1e-9 or abs(ky - b) > 1e-9 for a, b in uniq):
            uniq.append((kx, ky))
    return uniq


def _pw_solver(cell, F, kpts, npw=_NPW, nlow=_NLOW):
    """Fast local TM band solver. Returns (max omega_4, min omega_5) or None."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh
    half = (npw - 1) // 2
    m = np.arange(-half, half + 1)
    nM = len(m)
    m2 = m * m
    Gx = 2 * np.pi * m
    Gy = 2 * np.pi * m
    n = nM * nM

    max4 = 0.0
    min5 = np.inf
    for kx, ky in kpts:
        # diagonal |k - G|^2
        dx = kx - Gx
        dy = ky - Gy
        diag = np.broadcast_to(dx[None, :] ** 2, (nM, nM)) + np.broadcast_to(dy[:, None] ** 2, (nM, nM))
        A = diags(diag.ravel(), 0, shape=(n, n), format='csr')
        # inv_eps coupling matrix B (block structure)
        B = np.empty((nM, nM, nM, nM), dtype=np.complex128)
        for a in range(nM):
            for b in range(nM):
                B[a, b] = F[m[b] - m[a], m[a] - m[b]]
        B = B.transpose(0, 2, 1, 3).reshape(n, n)
        B = 0.5 * (B + B.conj().T)
        try:
            vals, _ = eigsh(B, k=nlow, sigma=0, which='LM')
            vals = np.real(vals)
            vals = vals[vals > 1e-10]
            vals.sort()
            if len(vals) >= 5:
                w4 = np.sqrt(vals[3])
                w5 = np.sqrt(vals[4])
                if w4 > max4:
                    max4 = w4
                if w5 < min5:
                    min5 = w5
        except Exception:
            continue
    if min5 == np.inf:
        return None
    return max4, min5


def _score_cell(cell, kpts, npw=_NPW, nlow=_NLOW):
    """gap-midgap ratio 2*(min5-max4)/(min5+max4) for the 4th/5th TM bands."""
    F = _fourier_inv_eps(cell, (npw - 1) // 2)
    res = _pw_solver(cell, F, kpts, npw, nlow)
    if res is None:
        return -1.0
    max4, min5 = res
    if min5 <= max4:
        return -1.0
    mid = (min5 + max4) / 2.0
    if mid <= 0:
        return 0.0
    return 2.0 * (min5 - max4) / (min5 + max4)


def design(problem):
    """Search for a symmetric 32x32 unit cell maximising the 4-5 TM band gap."""
    size = int(problem["pixels"])
    start = time.time()
    deadline = start + 200.0          # explicit wall-clock safety margin (< 230 s)

    kpts = _kpoint_set(_KPTS)

    # deterministic seed: textbook centred rod (guaranteed valid fallback)
    rod = _rod_cell(size, 0.2)
    best_cell = rod
    best_score = _score_cell(rod, kpts)

    np.random.seed(1234)
    step = 0.08
    max_steps = 40

    # a few deterministic structured starting points around the rod
    starts = [
        [(0.12, 0.12, 0.12, 1)],
        [(0.06, 0.06, 0.06, 1), (0.30, 0.30, 0.14, 1)],
        [(0.0, 0.0, 0.10, 1), (0.25, 0.25, 0.12, 1)],
        [(0.16, 0.0, 0.16, 1)],
    ]
    for s in starts:
        if time.time() > deadline - 15.0:
            break
        cell = _make_cell_from_params(s, size)
        sc = _score_cell(cell, kpts)
        if sc > best_score:
            best_score, best_cell = sc, cell.copy()

    # hill climbing with random restarts over circle parameters
    while time.time() < deadline - 15.0:
        n_c = np.random.randint(1, 4)
        params = []
        for _ in range(n_c):
            cx = np.random.uniform(0.0, 0.5)
            cy = np.random.uniform(0.0, 0.5)
            r = np.random.uniform(0.05, 0.35)
            sign = 1 if np.random.rand() < 0.8 else -1
            params.append((cx, cy, r, sign))
        cell = _make_cell_from_params(params, size)
        cur = _score_cell(cell, kpts)
        if cur > best_score:
            best_score, best_cell = cur, cell.copy()

        for _ in range(max_steps):
            if time.time() > deadline - 15.0:
                break
            idx = np.random.randint(len(params))
            cx, cy, r, sign = params[idx]
            cx = np.clip(cx + np.random.normal(0, step), 0.0, 0.5)
            cy = np.clip(cy + np.random.normal(0, step), 0.0, 0.5)
            r = np.clip(r + np.random.normal(0, step * 0.5), 0.04, 0.42)
            new = list(params)
            new[idx] = (cx, cy, r, sign)
            ncell = _make_cell_from_params(new, size)
            nsc = _score_cell(ncell, kpts)
            if nsc > cur:
                params, cell, cur = new, ncell, nsc
                if cur > best_score:
                    best_score, best_cell = cur, cell.copy()
            step *= 0.97
        step = 0.08

    # convert best cell to nested list of 0/1
    result = []
    for i in range(size):
        row = [1 if best_cell[i, j] > 0.5 else 0 for j in range(size)]
        result.append(row)
    return result
# EVOLVE-BLOCK-END
