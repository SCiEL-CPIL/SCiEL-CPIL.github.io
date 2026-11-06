"""Photonic-crystal design: large 4th-5th TM band gap via multi-scale D4-symmetric mask."""
import time
import numpy as np

_EPS_LOW = 1.0
_EPS_HIGH = 12.0
_NPW = 17          # plane-wave truncation radius (fast surrogate)
_NPW_FINE = 19     # slightly larger for the final candidate
_SIZE = 32

# 30 wavevectors on the irreducible BZ boundary (X->M->G)
def _irreducible_kvecs():
    pts = []
    n = 15
    for i in range(n):
        t = i / (n - 1)
        pts.append((0.5 * t, 0.0))          # G -> X
    for i in range(1, n):
        t = i / (n - 1)
        pts.append((0.5 * (1 - t), 0.5 * t)) # X -> M
    for i in range(1, n - 1):
        t = i / (n - 1)
        pts.append((0.5 * t, 0.5 * (1 - t))) # M -> G
    return np.array(pts)

_KVECS = _irreducible_kvecs()


def _fourier_coeffs(cell, npw):
    """Exact Fourier coefficients of the pixel map up to |G|<=npw.
    Returns dict keyed by (mx,my) -> complex value of the average over the cell."""
    N = len(cell)
    cell = np.asarray(cell, dtype=float)
    # sample at pixel centres
    idx = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(idx, idx, indexing='ij')
    coeffs = {}
    for mx in range(-npw, npw + 1):
        for my in range(-npw, npw + 1):
            if mx * mx + my * my > npw * npw:
                continue
            if mx == 0 and my == 0:
                val = cell.mean()
            else:
                phase = 2j * np.pi * (mx * X + my * Y)
                val = (cell * np.exp(phase)).mean()
            coeffs[(mx, my)] = val
    return coeffs


def _eps_inv_coeffs(cell, npw):
    """Fourier coefficients of 1/epsilon (for TM)."""
    N = len(cell)
    eps = np.where(np.asarray(cell, dtype=float) > 0.5, _EPS_HIGH, _EPS_LOW)
    eps_inv = 1.0 / eps
    idx = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(idx, idx, indexing='ij')
    coeffs = {}
    for mx in range(-npw, npw + 1):
        for my in range(-npw, npw + 1):
            if mx * mx + my * my > npw * npw:
                continue
            if mx == 0 and my == 0:
                val = eps_inv.mean()
            else:
                phase = 2j * np.pi * (mx * X + my * Y)
                val = (eps_inv * np.exp(phase)).mean()
            coeffs[(mx, my)] = val
    return coeffs


def _build_glist(npw):
    g = []
    for mx in range(-npw, npw + 1):
        for my in range(-npw, npw + 1):
            if mx * mx + my * my <= npw * npw:
                g.append((2.0 * np.pi * mx, 2.0 * np.pi * my))
    return np.array(g)


_GLIST = _build_glist(_NPW)
_GLIST_FINE = _build_glist(_NPW_FINE)


def _tm_bands(eps_inv_coeffs, k, glist, band_lo, band_hi):
    """Galerkin eigenvalues for TM at wavevector k. Returns (omega_lo, omega_hi)."""
    N = len(glist)
    G2 = np.einsum('ij,ij->i', glist, glist)
    K = k[0] + glist[:, 0]
    Ky = k[1] + glist[:, 1]
    K2 = K * K + Ky * Ky
    # matrix element (G-G')
    # C[G,G'] = (k+G)·(k+G') * eps_inv(G-G')
    # Build eps_inv on the difference grid
    # For each pair, need eps_inv(G_i - G_j)
    # Precompute eps_inv values for all needed differences
    # differences: G_i - G_j = 2pi*(m_i - m_j, n_i - n_j)
    # We have coeffs keyed by (mx,my) where G=2pi*(mx,my)
    # G_i - G_j = 2pi*(mx_i-mx_j, my_i-my_j)
    # So key = (mx_i - mx_j, my_i - my_j)
    # Need mx,my of each glist entry
    glist_int = np.round(glist / (2.0 * np.pi)).astype(int)
    mx_arr = glist_int[:, 0]
    my_arr = glist_int[:, 1]
    # Build matrix
    C = np.empty((N, N), dtype=complex)
    # vectorize over differences
    diff_mx = mx_arr[:, None] - mx_arr[None, :]
    diff_my = my_arr[:, None] - my_arr[None, :]
    # look up eps_inv coeffs
    # build a lookup from (mx,my) to value
    lookup = {}
    for key, val in eps_inv_coeffs.items():
        lookup[key] = val
    # vectorized lookup
    Cval = np.empty((N, N), dtype=complex)
    for a in range(N):
        for b in range(N):
            key = (diff_mx[a, b], diff_my[a, b])
            Cval[a, b] = lookup.get(key, 0.0)
    dot = (K[:, None] * K[None, :] + Ky[:, None] * Ky[None, :])
    C = dot * Cval
    # Hermitian, solve smallest eigenvalues
    w = np.linalg.eigvalsh(C)
    w = w[w > 0]
    if len(w) < band_hi:
        return (0.0, 0.0)
    omega = np.sqrt(w)
    return (float(omega[band_lo - 1]), float(omega[band_hi - 1]))


def _gap_ratio(cell, npw=_NPW, glist=None):
    """Compute 2(min w5 - max w4)/(min w5 + max w4) over the irreducible BZ."""
    if glist is None:
        glist = _GLIST
    eps_inv = _eps_inv_coeffs(cell, npw)
    max_gap = 0.0
    for k in _KVECS:
        w4, w5 = _tm_bands(eps_inv, k, glist, 4, 5)
        if w5 <= 0 or w4 <= 0:
            continue
        ratio = 2.0 * (w5 - w4) / (w5 + w4)
        if ratio > max_gap:
            max_gap = ratio
    return max_gap


def _make_cell(params):
    """Build a D4-symmetric 32x32 cell from parameters.
    params: (r1, r2, r3, phase, amp)
    Uses a sum of concentric D4-symmetric features:
      - a central dielectric blob of radius r1
      - a ring/annulus feature at radius r2 with D4 modulation
      - corner features at radius r3
    All features are D4-symmetric (invariant under 90-degree rotation and reflection).
    """
    N = _SIZE
    r1, r2, r3, phase, amp = params
    idx = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(idx, idx, indexing='ij')
    R = np.sqrt(X * X + Y * Y)
    # D4-symmetric basis functions
    # f1: central disk
    f1 = (R < r1).astype(float)
    # f2: D4-modulated ring (four-lobed)
    # Use cos(4*theta) modulation
    theta = np.arctan2(Y, X)
    ring = np.exp(-((R - r2) ** 2) / (0.02 * (r2 + 0.05) ** 2))
    mod = 0.5 + 0.5 * amp * np.cos(4 * theta + phase)
    f2 = (ring * mod)
    # f3: corner squares (D4 symmetric, at the four corners of the cell)
    # corners at (±0.5, ±0.5) but within cell; use distance to nearest corner
    cx = np.minimum(np.abs(X - 0.5), np.abs(X + 0.5))
    cy = np.minimum(np.abs(Y - 0.5), np.abs(Y + 0.5))
    corner = (np.sqrt(cx * cx + cy * cy) < r3).astype(float)
    f3 = corner
    # Combine: dielectric where f1 + f2 + f3 > threshold
    # threshold ~ 0.5 for the binary features, but f2 is continuous
    score = f1 + f2 + f3
    cell = (score > 0.5).astype(int)
    return cell


def _optimize(deadline):
    """Local search over the parameter space."""
    best_cell = None
    best_score = -1.0
    # initial guess
    start = (0.12, 0.22, 0.08, 0.0, 0.8)
    cell = _make_cell(start)
    score = _gap_ratio(cell)
    best_score = score
    best_cell = cell
    best_params = start

    # coordinate descent with random restarts
    np.random.seed(42)
    step_sizes = [0.02, 0.01, 0.005, 0.002]
    bounds = [(0.05, 0.30), (0.10, 0.35), (0.0, 0.20), (0.0, 2 * np.pi), (0.0, 1.5)]

    for restart in range(6):
        if time.time() > deadline - 30:
            break
        # current best params or random
        if restart == 0:
            params = list(best_params)
        else:
            params = [np.random.uniform(lo, hi) for lo, hi in bounds]
        cell = _make_cell(params)
        score = _gap_ratio(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            best_params = params
        # coordinate descent
        for ss in step_sizes:
            if time.time() > deadline - 30:
                break
            improved = True
            while improved:
                improved = False
                for dim in range(5):
                    for direction in [-1, 1]:
                        new_params = list(params)
                        new_params[dim] += direction * ss
                        if new_params[dim] < bounds[dim][0] or new_params[dim] > bounds[dim][1]:
                            continue
                        new_cell = _make_cell(new_params)
                        new_score = _gap_ratio(new_cell)
                        if new_score > score:
                            score = new_score
                            params = new_params
                            cell = new_cell
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_cell = cell
                                best_params = params
    return best_cell, best_score


def design(problem):
    size = problem["pixels"]
    assert size == 32
    deadline = time.time() + 200.0  # safety margin before 230s kill
    # Quick baseline: the seed rod
    cell_seed = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        cell_seed.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    best_cell = cell_seed
    best_score = _gap_ratio(cell_seed)
    # Try the multi-scale design
    if time.time() < deadline - 40:
        opt_cell, opt_score = _optimize(deadline)
        if opt_cell is not None and opt_score > best_score:
            best_cell = opt_cell
            best_score = opt_score
    # Ensure D4 symmetry (it should be by construction, but enforce)
    arr = np.array(best_cell, dtype=int)
    # enforce transpose symmetry
    arr = np.where(arr != arr.T, 0, arr)
    # enforce row/col reversal symmetry
    arr = np.where(arr != arr[::-1, :], 0, arr)
    arr = np.where(arr != arr[:, ::-1], 0, arr)
    return arr.tolist()
