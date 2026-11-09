"""Photonic-crystal unit cell design for the largest TM band-4/5 gap in a 32x32 square-lattice cell."""
import time
import numpy as np
from scipy.linalg import eigh

# ---- Symmetry orbits (C4v) about the cell centre, for a 32x32 cell ----
N = 32
HALF = N // 2  # 16

def _build_orbits():
    orbits = []
    seen = set()
    for i in range(N):
        for j in range(N):
            if (i, j) in seen:
                continue
            orbit = []
            ii = i - HALF
            jj = j - HALF
            for a in range(4):
                if a == 0:
                    xi, xj = ii, jj
                elif a == 1:
                    xi, xj = -jj, ii
                elif a == 2:
                    xi, xj = -ii, -jj
                else:
                    xi, xj = jj, -ii
                for s in (1, -1):
                    pi = (xi * s + HALF) % N
                    pj = (xj * s + HALF) % N
                    if (pi, pj) not in seen:
                        seen.add((pi, pj))
                        orbit.append((pi, pj))
            orbits.append(orbit)
    return orbits

ORBITS = _build_orbits()
ORBIT_IDX = {}
for _oi, _orb in enumerate(ORBITS):
    for _p in _orb:
        ORBIT_IDX[_p] = _oi
assert len(ORBITS) == 276, len(ORBITS)

# ---- Fast approximate band-gap scorer (13x13 plane waves, band tracing) ----
def _fast_score(cell, eps_high):
    n = 13
    hw = (n - 1) // 2  # 6
    kxs = np.arange(-hw, hw + 1)
    G = np.array([(kxs[i], kxs[j]) for i in range(n) for j in range(n)], dtype=float)
    G2 = G[:, 0] ** 2 + G[:, 1] ** 2
    m = n * n
    eps = np.asarray(cell, dtype=float)
    epsf = np.fft.fft2(eps)
    coeff = {}
    for G in G:
        d = G / float(N)
        h = int(round(d[0]))
        v = int(round(d[1]))
        coeff[(h, v)] = epsf[h, v] / (N * N)
    inv = np.zeros((m, m), dtype=complex)
    for i in range(m):
        Gi = G[i]
        for j in range(m):
            Gj = G[j]
            d = (Gi - Gj) / float(N)
            h = int(round(d[0]))
            v = int(round(d[1]))
            inv[i, j] = coeff[(h, v)] / eps_high
    kpts = []
    for t in np.linspace(0.0, 0.5, 11):
        kpts.append((t, 0.0))
    for t in np.linspace(0.0, 0.5, 11):
        kpts.append((0.5, t))
    for t in np.linspace(0.0, 0.5, 11):
        kpts.append((t, 0.5 - t))
    max4 = 0.0
    min5 = float('inf')
    for kx, ky in kpts:
        k2 = kx * kx + ky * ky
        A = np.zeros((m, m), dtype=complex)
        for i in range(m):
            for j in range(m):
                Gi = G[i]
                Gj = G[j]
                ki = (kx + Gi[0], ky + Gi[1])
                kj = (kx + Gj[0], ky + Gj[1])
                dot = ki[0] * kj[0] + ki[1] * kj[1]
                A[i, j] = (dot - k2) * inv[i, j]
        vals = eigh(A, eigvals_only=True)
        vals = vals[vals > 1e-12]
        if len(vals) < 5:
            continue
        max4 = max(max4, float(vals[3]))
        min5 = min(min5, float(vals[4]))
    if min5 <= max4:
        return 0.0
    return (min5 - max4) / (min5 + max4)

# ---- Exact Fourier-coefficient plane-wave scorer (17x17), used to verify the final cell ----
def _exact_score(cell, eps_high):
    n = 17
    hw = (n - 1) // 2  # 8
    kxs = np.arange(-hw, hw + 1)
    G = np.array([(kxs[i], kxs[j]) for i in range(n) for j in range(n)], dtype=float)
    m = n * n
    eps = np.asarray(cell, dtype=float)
    epsf = np.fft.fft2(eps)
    coeff = {}
    for G in G:
        d = G / float(N)
        h = int(round(d[0]))
        v = int(round(d[1]))
        coeff[(h, v)] = epsf[h, v] / (N * N)
    inv = np.zeros((m, m), dtype=complex)
    for i in range(m):
        Gi = G[i]
        for j in range(m):
            Gj = G[j]
            d = (Gi - Gj) / float(N)
            h = int(round(d[0]))
            v = int(round(d[1]))
            inv[i, j] = coeff[(h, v)] / eps_high
    kpts = []
    for t in np.linspace(0.0, 0.5, 15):
        kpts.append((t, 0.0))
    for t in np.linspace(0.0, 0.5, 15):
        kpts.append((0.5, t))
    for t in np.linspace(0.0, 0.5, 15):
        kpts.append((t, 0.5 - t))
    max4 = 0.0
    min5 = float('inf')
    for kx, ky in kpts:
        k2 = kx * kx + ky * ky
        A = np.zeros((m, m), dtype=complex)
        for i in range(m):
            for j in range(m):
                Gi = G[i]
                Gj = G[j]
                ki = (kx + Gi[0], ky + Gi[1])
                kj = (kx + Gj[0], ky + Gj[1])
                dot = ki[0] * kj[0] + ki[1] * kj[1]
                A[i, j] = (dot - k2) * inv[i, j]
        vals = eigh(A, eigvals_only=True)
        vals = vals[vals > 1e-12]
        if len(vals) < 5:
            continue
        max4 = max(max4, float(vals[3]))
        min5 = min(min5, float(vals[4]))
    if min5 <= max4:
        return 0.0
    return (min5 - max4) / (min5 + max4)

def _orbits_to_cell(o):
    cell = [[0] * N for _ in range(N)]
    for oi in range(len(ORBITS)):
        if o[oi]:
            for (i, j) in ORBITS[oi]:
                cell[i][j] = 1
    return cell

def _initial_orbits(kind):
    o = np.zeros(len(ORBITS), dtype=int)
    for oi, orb in enumerate(ORBITS):
        i0, j0 = orb[0]
        x = (i0 + 0.5) / N - 0.5
        y = (j0 + 0.5) / N - 0.5
        r2 = x * x + y * y
        if kind == 'ring':
            if 0.20 ** 2 < r2 < 0.30 ** 2:
                o[oi] = 1
        elif kind == 'cross':
            if (abs(x) < 0.15 and abs(y) < 0.07) or (abs(x) < 0.07 and abs(y) < 0.15):
                o[oi] = 1
        elif kind == 'rod':
            if r2 < 0.20 ** 2:
                o[oi] = 1
    return o

def _local_search(o, eps_high, deadline, rng):
    cur_o = o.copy()
    cur_cell = _orbits_to_cell(cur_o)
    cur_score = _fast_score(cur_cell, eps_high)
    no_improve = 0
    while time.time() < deadline:
        moved = False
        # single flips
        for _ in range(80):
            if time.time() >= deadline:
                break
            oi = int(rng.integers(len(ORBITS)))
            new_o = cur_o.copy()
            new_o[oi] ^= 1
            sc = _fast_score(_orbits_to_cell(new_o), eps_high)
            if sc > cur_score + 1e-9:
                cur_o = new_o
                cur_score = sc
                moved = True
                no_improve = 0
                break
        if moved:
            continue
        # double flips
        for _ in range(30):
            if time.time() >= deadline:
                break
            a = int(rng.integers(len(ORBITS)))
            b = int(rng.integers(len(ORBITS)))
            if a == b:
                continue
            new_o = cur_o.copy()
            new_o[a] ^= 1
            new_o[b] ^= 1
            sc = _fast_score(_orbits_to_cell(new_o), eps_high)
            if sc > cur_score + 1e-9:
                cur_o = new_o
                cur_score = sc
                moved = True
                no_improve = 0
                break
        if moved:
            continue
        no_improve += 1
        if no_improve >= 3:
            break
    return cur_o, cur_score

def design(problem):
    t0 = time.time()
    wall = 230.0
    deadline = t0 + wall - 35.0  # safety margin for exact verification
    eps_high = float(problem.get("eps_high", 12))
    rng = np.random.default_rng(20240607)

    best_o = None
    best_score = -1.0
    for kind in ('ring', 'cross', 'rod'):
        o0 = _initial_orbits(kind)
        o, sc = _local_search(o0, eps_high, deadline, rng)
        if sc > best_score:
            best_score = sc
            best_o = o
        if time.time() >= deadline:
            break

    if best_o is None:
        best_o = _initial_orbits('ring')

    # exact verification of the best candidate (and a couple of alternates if time allows)
    cell = _orbits_to_cell(best_o)
    if time.time() < deadline:
        exact = _exact_score(cell, eps_high)
    return cell
