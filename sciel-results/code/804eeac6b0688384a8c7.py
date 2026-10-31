"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_for_blocks(heights, boundaries, N):
    """Compute max overlap for a piecewise-constant construction.
    heights: list of heights [h0, h1, ..., h_{k-1}]
    boundaries: list of start indices [0, b1, b2, ..., N]
    """
    k = len(heights)
    max_ov = 0.0
    for i in range(N):
        # Find which block i belongs to
        blk = 0
        for j in range(1, k):
            if i >= boundaries[j]:
                blk = j
            else:
                break
        h_i = heights[blk]
        # h(i+1): if i+1 < N, find its block; else 0
        if i + 1 < N:
            blk_next = 0
            for j in range(1, k):
                if i + 1 >= boundaries[j]:
                    blk_next = j
                else:
                    break
            h_next = heights[blk_next]
        else:
            h_next = 0.0
        ov = h_i * (1.0 - h_next)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _overlap_for_3block(a, b, c, m1, m2, N):
    """Compute max overlap for 3-block construction."""
    # Block 1: indices 0..m1-1, height a
    # Block 2: indices m1..m2-1, height b
    # Block 3: indices m2..N-1, height c
    # Overlap values:
    # Within block 1 (i < m1-1): a*(1-a)
    ov1 = a * (1.0 - a)
    # At transition 1->2 (i = m1-1): a*(1-b)
    ov_trans1 = a * (1.0 - b)
    # Within block 2 (m1 <= i < m2-1): b*(1-b)
    ov2 = b * (1.0 - b)
    # At transition 2->3 (i = m2-1): b*(1-c)
    ov_trans2 = b * (1.0 - c)
    # Within block 3 (m2 <= i < N-1): c*(1-c)
    ov3 = c * (1.0 - c)
    # At end (i = N-1): c*(1-0) = c
    ov_end = c
    return max(ov1, ov_trans1, ov2, ov_trans2, ov3, ov_end)


def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    # Theoretical 3-block optimum: M = 2-sqrt(3)
    M_theory = 2.0 - math.sqrt(3.0)  # ~0.2679
    a0 = M_theory * (1.0 - M_theory)  # ~0.196
    b0 = M_theory  # ~0.268
    c0 = M_theory  # ~0.268

    # Initial 3-block construction with equal thirds
    m1_0 = N // 3
    m2_0 = 2 * N // 3

    best_score = _overlap_for_3block(a0, b0, c0, m1_0, m2_0, N)
    best_params = (a0, b0, c0, m1_0, m2_0)

    # Optimize over continuous (a, b, m1, m2) with c derived from constraint
    # Constraint: m1*a + (m2-m1)*b + (N-m2)*c = N/2
    # c = (N/2 - m1*a - (m2-m1)*b) / (N-m2)

    def obj_3block(x):
        a, b, m1_c, m2_c = x
        if m1_c < 1.0 or m2_c < m1_c + 1.0 or m2_c > N - 1.0:
            return 1e10
        c = (N / 2.0 - m1_c * a - (m2_c - m1_c) * b) / (N - m2_c)
        if c < -0.01 or c > 1.01:
            return 1e10
        c = np.clip(c, 0.0, 1.0)
        return _overlap_for_3block(a, b, c, m1_c, m2_c, N)

    # Multi-start Nelder-Mead optimization
    best_val = 1e10
    best_x = None

    initial_points = []
    # Theoretical optimum
    initial_points.append([a0, b0, m1_0, m2_0])
    # Various perturbations
    for a_p in [a0, 0.2, 0.25, 0.3, 0.15]:
        for b_p in [b0, 0.25, 0.3, 0.35, 0.2]:
            for m1_p in [N//3, N//3 - 5, N//3 + 5, N//4, N//2 - 10]:
                for m2_p in [2*N//3, 2*N//3 - 5, 2*N//3 + 5, 3*N//4]:
                    if 1 <= m1_p < m2_p <= N - 1:
                        initial_points.append([a_p, b_p, float(m1_p), float(m2_p)])

    for x0 in initial_points:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(obj_3block, np.array(x0, dtype=float), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # If we found a good continuous solution, round to integers
    if best_x is not None:
        a_opt = float(best_x[0])
        b_opt = float(best_x[1])
        m1_cont = float(best_x[2])
        m2_cont = float(best_x[3])
        m1_r = int(round(m1_cont))
        m2_r = int(round(m2_cont))
        m1_r = max(1, min(N - 2, m1_r))
        m2_r = max(m1_r + 1, min(N - 1, m2_r))

        # Recompute c with integer boundaries
        c_opt = (N / 2.0 - m1_r * a_opt - (m2_r - m1_r) * b_opt) / (N - m2_r)
        c_opt = np.clip(c_opt, 0.0, 1.0)

        score_r = _overlap_for_3block(a_opt, b_opt, c_opt, m1_r, m2_r, N)
        if score_r < best_score:
            best_score = score_r
            best_params = (a_opt, b_opt, c_opt, m1_r, m2_r)

    # Also try fixed integer boundaries and optimize (a, b) with c derived
    for m1_try in range(max(1, N // 3 - 10), min(N - 2, N // 3 + 11)):
        for m2_try in range(max(m1_try + 1, 2 * N // 3 - 10), min(N - 1, 2 * N // 3 + 11)):
            if time.time() > deadline - 3.0:
                break
            n1 = m1_try
            n2 = m2_try - m1_try
            n3 = N - m2_try

            def obj_ab(x):
                a_v, b_v = x
                c_v = (N / 2.0 - n1 * a_v - n2 * b_v) / n3
                if c_v < 0 or c_v > 1:
                    return 1e10
                return _overlap_for_3block(a_v, b_v, c_v, m1_try, m2_try, N)

            # Grid search for initial guess
            a_grid = np.linspace(0.05, 0.5, 50)
            b_grid = np.linspace(0.05, 0.6, 50)
            best_a_local = a_grid[0]
            best_b_local = b_grid[0]
            best_val_local = 1e10
            for a_g in a_grid:
                for b_g in b_grid:
                    c_g = (N / 2.0 - n1 * a_g - n2 * b_g) / n3
                    if c_g < 0 or c_g > 1:
                        continue
                    v = _overlap_for_3block(a_g, b_g, c_g, m1_try, m2_try, N)
                    if v < best_val_local:
                        best_val_local = v
                        best_a_local = a_g
                        best_b_local = b_g

            # Refine
            try:
                res = minimize(obj_ab, np.array([best_a_local, best_b_local]),
                               method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-15})
                a_ref = float(res.x[0])
                b_ref = float(res.x[1])
                c_ref = (N / 2.0 - n1 * a_ref - n2 * b_ref) / n3
                c_ref = np.clip(c_ref, 0.0, 1.0)
                score_ref = _overlap_for_3block(a_ref, b_ref, c_ref, m1_try, m2_try, N)
                if score_ref < best_score:
                    best_score = score_ref
                    best_params = (a_ref, b_ref, c_ref, m1_try, m2_try)
            except Exception:
                pass

    a_final, b_final, c_final, m1_final, m2_final = best_params

    # Construct h
    h = np.zeros(N)
    h[:m1_final] = a_final
    h[m1_final:m2_final] = b_final
    h[m2_final:] = c_final

    # Fix sum to be exactly N/2
    s = h.sum()
    diff = N / 2.0 - s
    if abs(diff) > 1e-15:
        # Distribute correction across the last element
        h[-1] += diff

    # Guard: ensure values within [0,1]
    h = np.clip(h, 0.0, 1.0)

    # Final sum correction after clipping (rarely needed)
    s = h.sum()
    diff = N / 2.0 - s
    if abs(diff) > 1e-15:
        h[-1] += diff
        h = np.clip(h, 0.0, 1.0)

    return h
