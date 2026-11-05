"""Photonic-crystal design: field-intensity-driven topological optimization
to maximise the TM gap between bands 4 and 5."""
import time
import numpy as np

# ---------- deadline guard ----------
_DEADLINE = time.time() + 215.0   # safety margin vs 230 s wall-clock kill
def _time_left():
    return _DEADLINE - time.time()

# ---------- symmetry helpers (4-fold square symmetry about cell centre) ----------
def _fold(cell):
    """Average the 4 symmetry-equivalent copies of a 32x32 cell."""
    c = cell
    return 0.25 * (c + c[::-1, ::-1] + c.T + c[::-1, ::-1].T)

def _sym(cell):
    """Enforce exact 4-fold symmetry (deterministic, no rounding noise)."""
    c = np.array(cell, dtype=float)
    return _fold(c).astype(int)

def _valid(cell):
    c = np.array(cell)
    return (c.shape == (32, 32)) and (np.array_equal(c, c[::-1, ::-1])) \
        and (np.array_equal(c, c.T))

# ---------- plane-wave TM band solver (exact Fourier coefficients) ----------
def _band_energies(cell, N=17, kx=0.0, ky=0.0, top=5):
    """Return the lowest `top` eigenvalues of the TM eigenproblem.
    Exact plane-wave method: build the (1/eps) matrix in the Fourier basis
    of the pixel map and diagonalize."""
    n = cell.shape[0]
    eps = np.where(np.array(cell, dtype=float) > 0.5, 12.0, 1.0)
    # Fourier coefficients of 1/eps at reciprocal vectors G = (2*pi/n)*(gx,gy)
    inv_eps = 1.0 / eps
    inv_eps = inv_eps - inv_eps.mean()   # subtract DC; DC handled analytically
    gs = list(range(-N, N + 1))
    G = np.array([(gx, gy) for gx in gs for gy in gs], dtype=float)  # (M,2) in units 2pi/n
    M = G.shape[0]
    G2 = (G[:, 0] ** 2 + G[:, 1] ** 2) * (2.0 * np.pi / n) ** 2      # |G|^2
    # Fourier coefficients of inv_eps at each G
    fx = np.exp(1j * (2.0 * np.pi / n) * G[:, 0][:, None] * np.arange(n))
    fy = np.exp(1j * (2.0 * np.pi / n) * G[:, 1][:, None] * np.arange(n))
    coeff = (fx @ inv_eps @ fy.T).real / (n * n)   # real by symmetry
    coeff[0] = 0.0                                  # DC term removed
    # matrix: A_ij = G2_i * delta_ij + coeff_(G_i - G_j)
    idx = {tuple(g): i for i, g in enumerate(G)}
    A = np.zeros((M, M), dtype=float)
    A[np.arange(M), np.arange(M)] = G2
    for i in range(M):
        gi = G[i]
        for j in range(M):
            d = tuple(gi - G[j])
            if d in idx:
                A[i, j] += coeff[idx[d]]
    # shift by k: E_i = |G_i + k|^2 ;  A'_ij = A_ij + (k·G_i + k·G_j + |k|^2) * delta? 
    # For TM: (G+k)^2 delta_ij + <1/eps>_{G_i-G_j}. We built A with G^2 on diag.
    kvec = np.array([kx, ky])
    k2 = kvec[0] ** 2 + kvec[1] ** 2
    kdotG = G @ kvec
    A[np.arange(M), np.arange(M)] = (G2 + 2.0 * kdotG + k2)
    w, _ = np.linalg.eigh(A)
    w = np.sort(w)
    return w[:top]

# ---------- score a cell ----------
def _score_cell(cell):
    """gap-midgap ratio between bands 4 and 5 over the irreducible BZ."""
    # 30 wave vectors over the irreducible wedge: x in [0,0.5], y in [0,x]
    kpts = []
    N = 6
    for i in range(N + 1):
        for j in range(i + 1):
            kx = 0.5 * i / N
            ky = 0.5 * j / N
            kpts.append((kx, ky))
    # add interior points
    for i in range(1, N):
        for j in range(1, i):
            kpts.append((0.5 * i / N, 0.5 * j / N))
    kpts = kpts[:30]
    max4 = 0.0
    min5 = np.inf
    for (kx, ky) in kpts:
        w = _band_energies(cell, N=15, kx=kx, ky=ky, top=5)
        if len(w) < 5:
            continue
        max4 = max(max4, w[3])
        min5 = min(min5, w[4])
    if min5 == np.inf or (max4 + min5) <= 0:
        return -1.0
    return 2.0 * (min5 - max4) / (min5 + max4)

# ---------- field-intensity-driven topological optimization ----------
def _field_intensity(cell, band, kx, ky, N=15):
    """Spatial |E|^2 of a given band (0-indexed) at wavevector (kx,ky)."""
    n = cell.shape[0]
    eps = np.where(np.array(cell, dtype=float) > 0.5, 12.0, 1.0)
    inv_eps = 1.0 / eps
    inv_eps = inv_eps - inv_eps.mean()
    gs = list(range(-N, N + 1))
    G = np.array([(gx, gy) for gx in gs for gy in gs], dtype=float)
    M = G.shape[0]
    G2 = (G[:, 0] ** 2 + G[:, 1] ** 2) * (2.0 * np.pi / n) ** 2
    fx = np.exp(1j * (2.0 * np.pi / n) * G[:, 0][:, None] * np.arange(n))
    fy = np.exp(1j * (2.0 * np.pi / n) * G[:, 1][:, None] * np.arange(n))
    coeff = (fx @ inv_eps @ fy.T).real / (n * n)
    coeff[0] = 0.0
    idx = {tuple(g): i for i, g in enumerate(G)}
    A = np.zeros((M, M), dtype=float)
    A[np.arange(M), np.arange(M)] = G2
    for i in range(M):
        gi = G[i]
        for j in range(M):
            d = tuple(gi - G[j])
            if d in idx:
                A[i, j] += coeff[idx[d]]
    kvec = np.array([kx, ky])
    k2 = kvec[0] ** 2 + kvec[1] ** 2
    kdotG = G @ kvec
    A[np.arange(M), np.arange(M)] = (G2 + 2.0 * kdotG + k2)
    w, v = np.linalg.eigh(A)
    vec = v[:, band]
    # reconstruct field in real space: E(r) = sum_G c_G e^{i(G+k).r}
    r = np.arange(n)
    EX = np.zeros((n, n), dtype=complex)
    for m in range(M):
        gx, gy = G[m]
        phx = np.exp(1j * (2.0 * np.pi / n) * (gx) * r)
        phy = np.exp(1j * (2.0 * np.pi / n) * (gy) * r)
        EX += vec[m] * np.outer(phx, phy)
    # |E|^2 (the k-phase factor has unit magnitude, drop it)
    inten = np.abs(EX) ** 2
    inten = inten / (inten.max() + 1e-12)
    return inten

def _topo_step(cell, band_lo=3, band_hi=4, kx=0.0, ky=0.0, rate=0.12):
    """Move dielectric toward the band that should be raised and away from the
    band that should be lowered, using their field intensities."""
    ilo = _field_intensity(cell, band_lo, kx, ky)
    ihi = _field_intensity(cell, band_hi, kx, ky)
    # gradient: add dielectric where ilo is large and ihi is small,
    # remove dielectric where ihi is large and ilo is small.
    eps = np.where(np.array(cell, dtype=float) > 0.5, 12.0, 1.0)
    # TM field weight: intensity is in 1/eps (displacement) space; use as-is.
    grad = ilo - ihi
    # continuous update
    eps_c = eps + rate * grad
    eps_c = np.clip(eps_c, 1.0, 12.0)
    # threshold to binary, then re-symmetrize
    cell_c = (eps_c > 6.5).astype(int)
    return _sym(cell_c)

def design(problem):
    size = problem["pixels"]
    best = np.zeros((size, size), dtype=int)
    best_score = -1.0

    # ---- seed: a few physically motivated starting points ----
    seeds = []
    # (a) central rod
    c = np.zeros((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    cx = (xx + 0.5) / size - 0.5
    cy = (yy + 0.5) / size - 0.5
    c[(cx ** 2 + cy ** 2) < 0.2 ** 2] = 1
    seeds.append(c.astype(int))
    # (b) checkerboard-ish (high contrast, strong gap)
    c = np.zeros((size, size))
    c[(xx // 4 + yy // 4) % 2 == 0] = 1
    seeds.append(_sym(c.astype(int)))
    # (c) ring
    c = np.zeros((size, size))
    r = np.sqrt(cx ** 2 + cy ** 2)
    c[(r > 0.15) & (r < 0.35)] = 1
    seeds.append(_sym(c.astype(int)))

    for s in seeds:
        if not _valid(s):
            s = _sym(s)
        sc = _score_cell(s)
        if sc > best_score:
            best_score = sc
            best = s.copy()
        if _time_left() < 60.0:
            break

    # ---- field-intensity-driven topological optimization ----
    # alternate which band we raise; use several k-points (including high-symmetry)
    kpts_opt = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.25), (0.5, 0.25)]
    it = 0
    while _time_left() > 45.0:
        it += 1
        # pick a k-point cyclically
        kp = kpts_opt[it % len(kpts_opt)]
        # decide direction: try raising band4 (lowering band5) via grad = ilo - ihi
        cand = _topo_step(best, band_lo=3, band_hi=4, kx=kp[0], ky=kp[1], rate=0.10)
        if not _valid(cand):
            cand = _sym(cand)
        sc = _score_cell(cand)
        if sc > best_score:
            best_score = sc
            best = cand.copy()
        # also try the opposite gradient occasionally
        if it % 3 == 0 and _time_left() > 45.0:
            cand2 = _topo_step(best, band_lo=4, band_hi=3, kx=kp[0], ky=kp[1], rate=0.10)
            if not _valid(cand2):
                cand2 = _sym(cand2)
            sc2 = _score_cell(cand2)
            if sc2 > best_score:
                best_score = sc2
                best = cand2.copy()

    # ---- final local refinement: single-pixel flips on the symmetric orbit ----
    if _time_left() > 30.0:
        for _ in range(40):
            if _time_left() < 30.0:
                break
            improved = False
            for i in range(size):
                for j in range(size):
                    if (i, j) in _sym_orbit(i, j):
                        continue  # only canonical representatives
                    cand = best.copy()
                    # flip the whole symmetric orbit
                    for (a, b) in _sym_orbit(i, j):
                        cand[a, b] = 1 - cand[a, b]
                    sc = _score_cell(cand)
                    if sc > best_score + 1e-9:
                        best_score = sc
                        best = cand.copy()
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    return best.tolist()

def _sym_orbit(i, j):
    """All 4-fold symmetric partners of pixel (i,j)."""
    n = 32
    return {(i, j), (n - 1 - i, n - 1 - j), (j, i), (n - 1 - j, n - 1 - i)}
