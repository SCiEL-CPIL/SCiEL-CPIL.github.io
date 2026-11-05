import time
import math
import numpy as np
from numpy.linalg import eigh

# Precompute plane-wave grid (radius 8 -> 289 PWs, matching the evaluator)
_PW_R = 8
_G_list = []
for _Gx in range(-_PW_R, _PW_R + 1):
    for _Gy in range(-_PW_R, _PW_R + 1):
        if _Gx * _Gx + _Gy * _Gy <= _PW_R * _PW_R:
            _G_list.append((_Gx, _Gy))
_G_arr = np.array(_G_list, dtype=float)
_n_pw = len(_G_list)
_size_d = 4 * _PW_R + 1
_off_d = 2 * _PW_R
# Precompute the difference-indexed Fourier matrix layout
_diff_i, _diff_j = np.indices((_n_pw, _n_pw))
_diff_ix = _G_arr[_diff_i, 0] - _G_arr[_diff_j, 0]
_diff_jy = _G_arr[_diff_i, 1] - _G_arr[_diff_j, 1]
_full_ix = _diff_ix + _off_d
_full_jy = _diff_jy + _off_d
# K-space dots for each sampled k (k + G) @ (k + G)^T
_k_points = [
    (0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.0), (0.5, 0.25),
    (0.25, 0.25), (0.0, 0.25), (0.35, 0.35), (0.45, 0.15), (0.2, 0.3),
    (0.1, 0.45), (0.4, 0.4), (0.15, 0.15), (0.3, 0.1), (0.2, 0.2),
    (0.45, 0.45), (0.05, 0.5), (0.5, 0.05), (0.1, 0.25), (0.25, 0.1),
]
_k_dots = []
for _k in _k_points:
    _kG = _k_arr = np.array(_k, dtype=float)[:, None] + _G_arr
    _k_dots.append(_kG @ _kG.T)


def _eval_gap(cell, n_k=10):
    """Fast proxy gap-midgap ratio for bands 4 and 5 (TM)."""
    N = 32
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    inv_eps_full = np.zeros((_size_d, _size_d), dtype=complex)
    inv_eps_full[_full_ix, _full_jy] = fft_inv[_diff_ix % N, _diff_jy % N]
    max_w4 = 0.0
    min_w5 = float('inf')
    for dots in _k_dots[:n_k]:
        M = dots * inv_eps_full
        M = (M + M.conj().T) / 2
        try:
            ev = np.sqrt(np.maximum(eigh(M, eigvals_only=True), 0.0))
            if len(ev) >= 5:
                if ev[3] > max_w4:
                    max_w4 = ev[3]
                if ev[4] < min_w5:
                    min_w5 = ev[4]
        except Exception:
            pass
    if min_w5 == float('inf') or max_w4 <= 0.0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _make_rod4(cx, cy, r, size=32):
    """Four quadrant rods (structural idea of the current program)."""
    cell = [[0] * size for _ in range(size)]
    centers = [(cx, cx), (cx, size - cx), (size - cx, cx), (size - cx, size - cx)]
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cxx, cyy in centers:
                if (x - cxx) ** 2 + (y - cyy) ** 2 < r * r:
                    cell[i][j] = 1
                    break
    return cell


def _make_square4(cx, cy, half, size=32):
    """Four quadrant squares (alternative shape, same 2x2 periodicity)."""
    cell = [[0] * size for _ in range(size)]
    centers = [(cx, cx), (cx, size - cx), (size - cx, cx), (size - cx, size - cx)]
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cxx, cyy in centers:
                if abs(x - cxx) < half and abs(y - cyy) < half:
                    cell[i][j] = 1
                    break
    return cell


def _mutate(cell, rate):
    """Flip symmetric pixel groups (fundamental quadrant mirrored 4 ways)."""
    size = len(cell)
    c = np.array(cell, dtype=int)
    for i in range(size // 2):
        for j in range(size // 2):
            if np.random.random() < rate:
                val = 1 - c[i, j]
                c[i, j] = val
                c[size - 1 - i, j] = val
                c[i, size - 1 - j] = val
                c[size - 1 - i, size - 1 - j] = val
    return c.tolist()


def design(problem):
    start = time.time()
    deadline = start + 220.0  # safety margin under the 230s kill

    # --- 1) Analytic seeds: tune the four-rod structure over radius/shape ---
    best_cell = None
    best_score = -1.0
    seeds = []
    for r in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]:
        seeds.append(_make_rod4(8.0, 8.0, r))
    for r in [5.0, 5.5, 6.0]:
        seeds.append(_make_rod4(7.0, 7.0, r))
    for half in [2.5, 3.0, 3.5, 4.0]:
        seeds.append(_make_square4(8.0, 8.0, half))
    for cell in seeds:
        if time.time() > deadline - 60:
            break
        s = _eval_gap(cell, n_k=10)
        if s > best_score:
            best_score = s
            best_cell = cell

    # --- 2) Local search: symmetric pixel mutations around the best ---
    if best_cell is not None:
        current = best_cell
        cur_score = best_score
        attempts = 0
        while time.time() < deadline - 20 and attempts < 400:
            attempts += 1
            mutated = _mutate(current, 0.03)
            s = _eval_gap(mutated, n_k=10)
            if s > cur_score:
                cur_score = s
                current = mutated
                best_score = s
                best_cell = current

    if best_cell is None:
        best_cell = _make_rod4(8.0, 8.0, 5.5)
    return best_cell
