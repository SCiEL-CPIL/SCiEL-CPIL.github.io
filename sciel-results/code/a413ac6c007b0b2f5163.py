"""Design a symmetric 32x32 photonic-crystal unit cell maximizing the TM 4-5 band gap."""
import time
import math
import random

import numpy as np

# ---------- constants ----------
SIZE = 32
EPS = 12.0
N_G = 7          # 7x7 G-vectors -> 49 plane waves
TOL = 1e-10
DEADLINE_FRAC = 0.88   # safety margin before 230 s kill


def _build_cell(q):
    """Mirror an 8x8 quarter (top-left) to a full 32x32 symmetric cell."""
    top = np.empty((16, 32), dtype=np.int8)
    top[:, :16] = q
    top[:, 16:] = q[:, ::-1]
    full = np.empty((32, 32), dtype=np.int8)
    full[:16] = top
    full[16:] = top[::-1]
    return full


def _fourier_basis():
    """Return (M, g2) where M[i,j] = exp(i G_j . r_i) and g2[j] = |G_j|^2."""
    idx = np.arange(SIZE) + 0.5
    r = (idx / SIZE - 0.5) * 2.0 * np.pi          # positions in [0,1) units of 2pi
    M = np.empty((SIZE * SIZE, N_G * N_G), dtype=np.complex128)
    g2 = np.empty(N_G * N_G, dtype=np.float64)
    col = 0
    for gx in range(-N_G // 2, N_G // 2 + 1):
        for gy in range(-N_G // 2, N_G // 2 + 1):
            M[:, col] = np.exp(1j * (gx * r[:, None] + gy * r[None, :])).ravel()
            g2[col] = gx * gx + gy * gy
            col += 1
    return M, g2


def _inv_eps(cell, M):
    """Inverse-permittivity Fourier coefficients in the 49-PW basis."""
    c = cell.ravel().astype(np.float64)
    eps = 1.0 + (EPS - 1.0) * c
    inv = 1.0 / eps
    Mf = M.T.conj() @ inv
    return Mf / (SIZE * SIZE)


def _eigs(inv, kx, ky):
    """Eigenvalues (omega^2) for a k-point, ascending."""
    N = N_G * N_G
    A = np.zeros((N, N), dtype=np.complex128)
    for a in range(N):
        for b in range(N):
            ka = kx + a
            kb = ky + b
            A[a, b] = ka * ka + kb * kb + inv[(b - a) % N]
    w = np.linalg.eigvalsh(A)
    return w


def _score(cell, M):
    inv = _inv_eps(cell, M)
    kpts = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.0), (0.25, 0.25)]
    o4 = 0.0
    o5 = 1e18
    for kx, ky in kpts:
        w = _eigs(inv, kx, ky)
        o4 = max(o4, math.sqrt(w[3]))
        o5 = min(o5, math.sqrt(w[4]))
    if o5 <= 0.0:
        return -1.0
    return 2.0 * (o5 - o4) / (o5 + o4)


def _random_quarter(rng, density=0.5):
    q = np.zeros((8, 8), dtype=np.int8)
    for i in range(8):
        for j in range(8):
            q[i, j] = 1 if rng.random() < density else 0
    return q


def design(problem):
    t0 = time.time()
    deadline = t0 + 120.0 * DEADLINE_FRAC
    size = problem.get("pixels", SIZE)
    if size != SIZE:
        # Fallback: single central rod (safe default)
        cell = np.zeros((size, size), dtype=np.int8)
        cx = cy = size / 2.0
        for i in range(size):
            for j in range(size):
                if (i - cx) ** 2 + (j - cy) ** 2 < (0.2 * size) ** 2:
                    cell[i, j] = 1
        return cell.tolist()

    rng = random.Random(12345)
    M, g2 = _fourier_basis()

    # ---- candidate pool (each a symmetric quarter) ----
    pool = []

    # concentric rings
    for inner in range(0, 8):
        for outer in range(inner + 1, 8):
            q = np.zeros((8, 8), dtype=np.int8)
            for i in range(8):
                for j in range(8):
                    d2 = (i + 0.5) ** 2 + (j + 0.5) ** 2
                    if inner * inner <= d2 < outer * outer:
                        q[i, j] = 1
            pool.append(q.copy())

    # single central disk
    for rad in range(2, 8):
        q = np.zeros((8, 8), dtype=np.int8)
        for i in range(8):
            for j in range(8):
                d2 = (i + 0.5) ** 2 + (j + 0.5) ** 2
                if d2 < rad * rad:
                    q[i, j] = 1
        pool.append(q.copy())

    # four-corner rods
    for rad in range(1, 4):
        q = np.zeros((8, 8), dtype=np.int8)
        for cx, cy in [(0.5, 0.5), (0.5, 7.5), (7.5, 0.5), (7.5, 7.5)]:
            for i in range(8):
                for j in range(8):
                    if (i + 0.5 - cx) ** 2 + (j + 0.5 - cy) ** 2 < rad * rad:
                        q[i, j] = 1
        pool.append(q.copy())

    # cross
    q = np.zeros((8, 8), dtype=np.int8)
    q[3, :] = 1
    q[:, 3] = 1
    pool.append(q.copy())

    # checkerboards
    for shift in (0, 1):
        q = np.zeros((8, 8), dtype=np.int8)
        for i in range(8):
            for j in range(8):
                if (i + j + shift) % 2 == 0:
                    q[i, j] = 1
        pool.append(q.copy())

    # random quarters
    for dens in (0.3, 0.4, 0.5, 0.6, 0.7):
        for _ in range(60):
            pool.append(_random_quarter(rng, dens))

    # ---- evaluate pool ----
    best_q = None
    best_score = -1e18
    for q in pool:
        if time.time() > deadline:
            break
        cell = _build_cell(q)
        s = _score(cell, M)
        if s > best_score:
            best_score = s
            best_q = q.copy()

    # ---- local polish around the best candidate ----
    if best_q is not None:
        cur_q = best_q.copy()
        cur_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # propose a small number of random flips per iteration
            for _ in range(20):
                if time.time() > deadline:
                    break
                trial = cur_q.copy()
                nflip = rng.choice([1, 1, 2])
                for _ in range(nflip):
                    i = rng.randrange(8)
                    j = rng.randrange(8)
                    trial[i, j] = 1 - trial[i, j]
                cell = _build_cell(trial)
                s = _score(cell, M)
                if s > cur_score + TOL:
                    cur_score = s
                    cur_q = trial
                    improved = True
        if cur_score > best_score:
            best_score = cur_score
            best_q = cur_q

    if best_q is None:
        # absolute fallback
        best_q = np.zeros((8, 8), dtype=np.int8)
        best_q[3:5, 3:5] = 1

    cell = _build_cell(best_q)
    return cell.tolist()
