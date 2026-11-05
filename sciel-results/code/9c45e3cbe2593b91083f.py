import numpy as np
import time

_DEADLINE = 120.0
_t0 = None


def _superellipse_cell(N, r, exp):
    """Symmetric 4-fold unit cell: superellipse dielectric rod in air."""
    cell = []
    for i in range(N):
        x = (i + 0.5) / N - 0.5
        row = []
        for j in range(N):
            y = (j + 0.5) / N - 0.5
            if exp >= 50:
                v = max(abs(x), abs(y)) < r
            else:
                v = (abs(x) ** exp + abs(y) ** exp) < (r ** exp)
            row.append(1 if v else 0)
        cell.append(row)
    return cell


def design(problem):
    global _t0
    if _t0 is None:
        _t0 = time.time()
    N = int(problem["pixels"])
    M = max(1, N // 4)                      # 289 plane waves for N=32
    Ms = list(range(-M, M + 1))
    Glist = [(2.0 * np.pi * m, 2.0 * np.pi * n) for m in Ms for n in Ms]
    dim = len(Glist)
    Gp = np.array(Glist, dtype=float)
    gm = np.array([g[0] for g in Glist])
    gn = np.array([g[1] for g in Glist])

    # Irreducible BZ (a=1): Gamma->X->M->Gamma
    nseg = 10
    kpts = []
    for t in np.linspace(0, 1, nseg):
        kpts.append((np.pi * t, 0.0))
    for t in np.linspace(0, 1, nseg):
        kpts.append((np.pi, np.pi * t))
    for t in np.linspace(0, 1, nseg):
        kpts.append((np.pi * (1 - t), np.pi * (1 - t)))

    def score(cell):
        eps = np.where(np.array(cell, dtype=float) == 1, 12.0, 1.0)
        inv = 1.0 / eps
        F = np.fft.fft2(inv) / (N * N)
        im = (gm[:, None] - gm[None, :]) % N
        in_ = (gn[:, None] - gn[None, :]) % N
        C = F[im, in_]
        max4 = -np.inf
        min5 = np.inf
        for kx, ky in kpts:
            kp = Gp + np.array([kx, ky])
            A = (kp[:, 0:1] * kp[None, :, 0] + kp[:, 1:2] * kp[None, :, 1]) * C
            w2 = np.linalg.eigvalsh(A)
            w = np.sqrt(np.clip(np.real(w2), 0, None))
            if w[3] > max4:
                max4 = w[3]
            if w[4] < min5:
                min5 = w[4]
        if min5 <= 0:
            return -1.0
        return 2.0 * (min5 - max4) / (min5 + max4)

    best_cell = None
    best_s = -np.inf

    def try_cell(r, exp):
        nonlocal best_cell, best_s
        if time.time() - _t0 > _DEADLINE - 30:
            return False
        cell = _superellipse_cell(N, r, exp)
        try:
            s = score(cell)
        except Exception:
            s = -np.inf
        if s > best_s:
            best_s = s
            best_cell = cell
        return time.time() - _t0 < _DEADLINE - 30

    radii = [0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38]
    exponents = [2, 3, 4, 6, 100]
    for r in radii:
        for exp in exponents:
            if not try_cell(r, exp):
                break
        if time.time() - _t0 > _DEADLINE - 30:
            break

    if best_cell is None:
        best_cell = _superellipse_cell(N, 0.30, 2)
    return best_cell
