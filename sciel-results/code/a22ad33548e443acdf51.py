"""Photonic-crystal unit cell design for a large TM 4-5 band gap."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_SIZE = 32
_EPS = 12.0
_NPW = 8          # coarse plane-wave basis for the surrogate
_KPTS = 9         # k-points in the irreducible BZ (x from 0..0.5, y from 0..x)
_DEADLINE_FRAC = 0.92   # return by ~92% of the wall-clock budget


def _symmetrize(cell):
    """Enforce full C4v (reflection + transpose) symmetry about the centre."""
    n = len(cell)
    a = np.array(cell, dtype=float)
    a = 0.25 * (a + a[::-1, :] + a[:, ::-1] + a[::-1, ::-1])
    a = np.where(a >= 0.5, 1.0, 0.0)
    # transpose symmetry
    a = 0.5 * (a + a.T)
    a = np.where(a >= 0.5, 1.0, 0.0)
    return a


def _fourier_coeffs(cell):
    """Exact Fourier coefficients of the 32x32 pixel map (eps(x) = eps_low + (eps_high-eps_low)*cell)."""
    n = len(cell)
    c = np.array(cell, dtype=float)
    # pixel grid: pixel (i,j) covers [i/n,(i+1)/n] x [j/n,(j+1)/n]
    # Fourier coeff: (1/n^2) sum_{i,j} cell[i,j] * exp(-2 pi i (h i/n + k j/n))
    idx = np.arange(n)
    gh = np.exp(-2j * np.pi * np.outer(idx, idx) / n)  # gh[i,j] = exp(-2 pi i i j / n)
    # C[h,k] = (1/n^2) sum_i sum_j c[i,j] exp(-2 pi i (h i + k j)/n)
    # = (1/n^2) (c @ gh)[h,k]  where gh[h,i] = exp(-2 pi i h i / n)
    # gh is symmetric so c @ gh gives sum_i c[i,j] exp(-2 pi i i j / n); need index swap
    C = (c @ gh) / (n * n)
    # C[h,k] = sum_i sum_j c[i,j] exp(-2 pi i (h*i + k*j)/n) / n^2  -> correct
    return C


def _surrogate_gap(cell, npw=_NPW, kpts=_KPTS):
    """Fast surrogate: TM band 4-5 gap-midgap ratio using npw plane waves at kpts k-points."""
    n = len(cell)
    C = _fourier_coeffs(cell)
    eps_low = 1.0
    eps_high = _EPS
    # inverse-eps Fourier coefficients: 1/eps(x) = sum_G Einv[G] exp(i G.x)
    # For a binary material, 1/eps(x) is also a 2-valued function; compute via pixel average of 1/eps
    inv_eps = np.where(np.array(cell, dtype=float) > 0.5, 1.0 / eps_high, 1.0 / eps_low)
    idx = np.arange(n)
    gh = np.exp(-2j * np.pi * np.outer(idx, idx) / n)
    Cinv = (inv_eps @ gh) / (n * n)

    # plane wave list: |G| <= npw
    G = []
    for h in range(-npw, npw + 1):
        for k in range(-npw, npw + 1):
            if abs(h) + abs(k) <= npw + 1:  # diamond cutoff
                G.append((h, k))
    G = np.array(G, dtype=float)
    m = len(G)

    # k-points in irreducible BZ (square lattice, BZ is [0,1]x[0,1] in lattice units)
    ks = []
    for a in range(kpts):
        for b in range(a + 1):
            ks.append((a / kpts, b / kpts))

    best_ratio = 0.0
    for (kx, ky) in ks:
        # TM: H field, eigenvalue problem: (1/eps) (G+K) x (G+K) H = omega^2 H
        # For 2D: omega^2 = (1/eps) |G+K|^2, coupled by 1/eps Fourier series
        # Build matrix: M[p,q] = (G_p+K) x (G_q+K) * Cinv[G_p - G_q]
        # For TM (H_z): the operator is (1/eps) * (k_perp^2) where k_perp = G+K
        K = np.array([kx, ky])
        GpK = G + K
        k2 = np.einsum('ij,ij->i', GpK, GpK)  # |G+K|^2
        # M[p,q] = k2_p * Cinv[G_p - G_q]
        # Cinv indexed by difference
        diff = G[:, None, :] - G[None, :, :]  # m x m x 2
        # get Cinv for each diff: Cinv[h,k]
        dh = diff[:, :, 0].astype(int)
        dk = diff[:, :, 1].astype(int)
        # build lookup
        maxg = npw + 2
        cmat = np.zeros((2 * maxg + 1, 2 * maxg + 1), dtype=complex)
        for h in range(-npw, npw + 1):
            for k in range(-npw, npw + 1):
                if abs(h) + abs(k) <= npw + 1:
                    cmat[h + maxg, k + maxg] = Cinv[h, k]
        Cinvm = cmat[dh + maxg, dk + maxg]
        M = Cinvm * k2[:, None]
        # Hermitian: M should be Hermitian; symmetrize
        M = 0.5 * (M + M.conj().T)
        try:
            w2 = np.linalg.eigvalsh(M)
        except np.linalg.LinAlgError:
            continue
        w2 = np.sort(np.real(w2))
        if len(w2) < 5:
            continue
        w4 = np.sqrt(max(w2[3], 1e-12))
        w5 = np.sqrt(max(w2[4], 1e-12))
        ratio = 2.0 * (w5 - w4) / (w5 + w4)
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio


def _cell_from_radii(radii):
    """Build a 32x32 cell from a 4x4 superlattice of circular disks.
    radii: 4x4 array of disk radii in lattice units (0..0.5).
    Each disk is centred in its 8x8 supercell. Symmetrize for C4v."""
    n = _SIZE
    sc = 4          # superlattice
    block = n // sc  # 8
    cell = np.zeros((n, n))
    # For C4v about centre, the 4x4 supercell must be symmetric:
    # radii[i][j] == radii[3-i][j] == radii[i][3-j] == radii[3-i][3-j]
    # We'll just compute all 16 and symmetrize the final cell.
    for i in range(sc):
        for j in range(sc):
            r = radii[i, j]
            if r <= 0:
                continue
            cy = i * block + block / 2.0
            cx = j * block + block / 2.0
            for y in range(i * block, (i + 1) * block):
                for x in range(j * block, (j + 1) * block):
                    dy = (y + 0.5) - cy
                    dx = (x + 0.5) - cx
                    if dx * dx + dy * dy < r * r:
                        cell[y, x] = 1.0
    # symmetrize
    cell = _symmetrize(cell)
    return cell


def _objective(params):
    """Negative gap ratio (we minimize). params: 6 free radii (symmetric 4x4 has 6 unique)."""
    # 4x4 with C4v: unique entries are (0,0),(0,1),(0,2),(0,3),(1,1),(1,2) = 6
    r00, r01, r02, r03, r11, r12 = params
    radii = np.zeros((4, 4))
    radii[0, 0] = r00
    radii[0, 1] = r01
    radii[0, 2] = r02
    radii[0, 3] = r03
    radii[1, 1] = r11
    radii[1, 2] = r12
    # mirror
    radii[3, 0] = r00
    radii[3, 1] = r01
    radii[3, 2] = r02
    radii[3, 3] = r03
    radii[2, 0] = r00
    radii[2, 1] = r01
    radii[2, 2] = r02
    radii[2, 3] = r03
    radii[1, 0] = r01
    radii[1, 3] = r03
    radii[2, 1] = r12
    radii[2, 2] = r11
    radii[2, 3] = r12
    radii[1, 2] = r12
    radii[3, 1] = r01
    radii[3, 2] = r02
    cell = _cell_from_radii(radii)
    return -_surrogate_gap(cell)


def _default_cell():
    """A strong hand-tuned dielectric-background-with-holes cell (high fill fraction)."""
    n = _SIZE
    cell = np.ones((n, n))
    # 4x4 superlattice of air holes (radius ~0.28 in lattice units) at supercell centres
    sc = 4
    block = n // sc
    r = 0.27
    for i in range(sc):
        for j in range(sc):
            cy = i * block + block / 2.0
            cx = j * block + block / 2.0
            for y in range(i * block, (i + 1) * block):
                for x in range(j * block, (j + 1) * block):
                    dy = (y + 0.5) - cy
                    dx = (x + 0.5) - cx
                    if dx * dx + dy * dy < r * r:
                        cell[y, x] = 0.0
    cell = _symmetrize(cell)
    return cell


def design(problem):
    size = problem.get("pixels", 32)
    _SIZE_g = size
    t_start = time.time()
    deadline = t_start + 230.0 * _DEADLINE_FRAC

    # Strong default (dielectric background with 4x4 air holes)
    best_cell = _default_cell()
    best_score = _surrogate_gap(best_cell)

    # Try Nelder-Mead on 6 radii with a deadline guard
    try:
        # initial guess: holes radius ~0.27 in lattice units
        x0 = np.array([0.27, 0.27, 0.27, 0.27, 0.27, 0.27])
        bounds = [(0.05, 0.45)] * 6

        state = {"best": best_score, "best_params": x0.copy(), "done": False}

        def _obj_guarded(p):
            if state["done"]:
                return 0.0
            if time.time() > deadline:
                state["done"] = True
                return 0.0
            val = _objective(p)
            if -val > state["best"]:
                state["best"] = -val
                state["best_params"] = p.copy()
            return val

        res = minimize(
            _obj_guarded, x0, method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 200, "xatol": 0.01, "fatol": 0.005, "adaptive": True},
        )
        # also evaluate the final
        if not state["done"]:
            final_val = _objective(state["best_params"])
            if -final_val > state["best"]:
                state["best"] = -final_val

        # rebuild best cell from best params
        r00, r01, r02, r03, r11, r12 = state["best_params"]
        radii = np.zeros((4, 4))
        radii[0, 0] = r00; radii[0, 1] = r01; radii[0, 2] = r02; radii[0, 3] = r03
        radii[1, 1] = r11; radii[1, 2] = r12
        radii[3, 0] = r00; radii[3, 1] = r01; radii[3, 2] = r02; radii[3, 3] = r03
        radii[2, 0] = r00; radii[2, 1] = r01; radii[2, 2] = r02; radii[2, 3] = r03
        radii[1, 0] = r01; radii[1, 3] = r03
        radii[2, 1] = r12; radii[2, 2] = r11; radii[2, 3] = r12
        radii[1, 2] = r12; radii[3, 1] = r01; radii[3, 2] = r02
        opt_cell = _cell_from_radii(radii)
        opt_score = _surrogate_gap(opt_cell)
        if opt_score > best_score:
            best_cell = opt_cell
            best_score = opt_score
    except Exception:
        pass

    # Final symmetrize and return nested list
    best_cell = _symmetrize(best_cell)
    return [[int(v) for v in row] for row in best_cell]
# EVOLVE-BLOCK-END
