import numpy as np
import time
import sys
from scipy.optimize import minimize

def compute_1d_cost(positions):
    """
    Computes the integral from 0 to 1 of the union length of the intervals
    [x_i + c*i/8, x_i + 1/8 + c*i/8] for i=0..7.
    """
    N = 8
    breakpoints = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / 8.0
            if abs(denom) > 1e-12:
                c_val = (positions[j] - positions[i]) / denom
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

                c_val = (positions[j] + 1.0/8.0 - positions[i]) / denom
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

                c_val = (positions[j] - positions[i] - 1.0/8.0) / denom
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

    breakpoints = sorted(set(breakpoints))

    total_cost = 0.0
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        c_mid = (c_start + c_end) / 2.0

        intervals = []
        for i in range(N):
            l = positions[i] + (i / 8.0) * c_mid
            r = l + 1.0 / 8.0
            intervals.append((l, r))

        intervals.sort(key=lambda x: x[0])
        merged = []
        cur_l, cur_r = intervals[0]
        for l, r in intervals[1:]:
            if l <= cur_r:
                if r > cur_r:
                    cur_r = r
            else:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r
        merged.append((cur_l, cur_r))

        length_at_mid = sum(r - l for l, r in merged)
        total_cost += length_at_mid * (c_end - c_start)

    return total_cost


def compute_2d_volume(positions_2d):
    """
    Computes the volume of the union of 64 sheared tubes in 3D.
    positions_2d: array of shape (64, 2) with (x_k, y_k) for tube k.
    Tube k has i = k // 8, j = k % 8.
    At height c, tube k is [x_k + c*i/8, x_k + c*i/8 + 1/8] x [y_k + c*j/8, y_k + c*j/8 + 1/8].

    We compute V = integral_0^1 Area(c) dc.
    Area(c) is the area of the union of 64 axis-aligned squares at height c.
    We use a 1D breakpoint approach: the union area of axis-aligned rectangles
    is piecewise quadratic in c (since edges are linear in c). We find all
    breakpoints where the combinatorial structure changes, then integrate
    exactly over each piece.
    """
    N = 8
    # Extract base positions
    base_x = np.zeros(64)
    base_y = np.zeros(64)
    for k in range(64):
        base_x[k] = positions_2d[k, 0]
        base_y[k] = positions_2d[k, 1]

    # For each tube k, at height c:
    #   x_min(k,c) = base_x[k] + c * (k//8) / 8
    #   x_max(k,c) = x_min(k,c) + 1/8
    #   y_min(k,c) = base_y[k] + c * (k%8) / 8
    #   y_max(k,c) = y_min(k,c) + 1/8

    # Breakpoints in c where any two vertical edges coincide or any two horizontal edges coincide
    breakpoints = [0.0, 1.0]

    for k1 in range(64):
        i1 = k1 // N
        j1 = k1 % N
        for k2 in range(k1 + 1, 64):
            i2 = k2 // N
            j2 = k2 % N

            # Vertical edges: x_min(k1,c) = x_min(k2,c), x_min(k1,c) = x_max(k2,c), etc.
            # x_min(k1,c) = base_x[k1] + c*i1/8
            # x_max(k1,c) = base_x[k1] + 1/8 + c*i1/8
            # Similarly for k2.

            # Pair: x_min(k1) = x_min(k2)
            denom = (i1 - i2) / 8.0
            if abs(denom) > 1e-12:
                c_val = (base_x[k2] - base_x[k1]) / denom
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

            # x_min(k1) = x_max(k2)
            if abs(denom) > 1e-12:
                c_val = (base_x[k2] + 1.0/8.0 - base_x[k1]) / denom
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

            # x_max(k1) = x_min(k2)
            if abs(denom) > 1e-12:
                c_val = (base_x[k2] - base_x[k1] - 1.0/8.0) / denom
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

            # x_max(k1) = x_max(k2)
            if abs(denom) > 1e-12:
                c_val = (base_x[k2] + 1.0/8.0 - base_x[k1] - 1.0/8.0) / denom
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

            # Horizontal edges
            denom_y = (j1 - j2) / 8.0
            if abs(denom_y) > 1e-12:
                c_val = (base_y[k2] - base_y[k1]) / denom_y
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

                c_val = (base_y[k2] + 1.0/8.0 - base_y[k1]) / denom_y
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

                c_val = (base_y[k2] - base_y[k1] - 1.0/8.0) / denom_y
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

                c_val = (base_y[k2] + 1.0/8.0 - base_y[k1] - 1.0/8.0) / denom_y
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)

    breakpoints = sorted(set(breakpoints))

    total_volume = 0.0
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        c_mid = (c_start + c_end) / 2.0

        # Compute union area at c_mid using sweep line
        x_edges = []
        y_edges = []
        for kk in range(64):
            ik = kk // N
            jk = kk % N
            x_min = base_x[kk] + c_mid * ik / 8.0
            x_max = x_min + 1.0 / 8.0
            y_min = base_y[kk] + c_mid * jk / 8.0
            y_max = y_min + 1.0 / 8.0
            x_edges.append((x_min, 1, y_min, y_max))
            x_edges.append((x_max, -1, y_min, y_max))

        x_edges.sort(key=lambda e: e[0])

        # Sweep line over x
        area = 0.0
        cur_x = x_edges[0][0]
        active_y = []  # list of (y_min, y_max) for active intervals

        for x, delta, y_min, y_max in x_edges:
            dx = x - cur_x
            if dx > 0:
                # Compute union length of active_y
                if active_y:
                    active_y.sort()
                    merged_y = []
                    cy_min, cy_max = active_y[0]
                    for ym, yM in active_y[1:]:
                        if ym <= cy_max:
                            if yM > cy_max:
                                cy_max = yM
                        else:
                            merged_y.append((cy_min, cy_max))
                            cy_min, cy_max = ym, yM
                    merged_y.append((cy_min, cy_max))
                    y_len = sum(yM - ym for ym, yM in merged_y)
                else:
                    y_len = 0.0
                area += y_len * dx
            cur_x = x

            if delta == 1:
                active_y.append((y_min, y_max))
            else:
                # Remove (y_min, y_max)
                for idx, (ym, yM) in enumerate(active_y):
                    if ym == y_min and yM == y_max:
                        active_y.pop(idx)
                        break

        total_volume += area * (c_end - c_start)

    return total_volume


def optimize_1d(positions, time_limit=10.0):
    """
    Simulated Annealing to minimize 1D cost.
    """
    start_time = time.time()
    current_cost = compute_1d_cost(positions)
    best_positions = positions.copy()
    best_cost = current_cost

    T = 0.01
    T_min = 1e-6
    alpha = 0.99
    step_size = 0.05

    while time.time() - start_time < time_limit and T > T_min:
        new_positions = positions + np.random.normal(0, step_size, len(positions))

        new_cost = compute_1d_cost(new_positions)

        if new_cost < current_cost or np.random.rand() < np.exp((current_cost - new_cost) / T):
            current_cost = new_cost
            positions = new_positions
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = positions.copy()

        T *= alpha

    return best_positions, best_cost


def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0

    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    best_1d_positions = keich.copy()
    best_1d_cost = compute_1d_cost(best_1d_positions)

    # Polish the initial Keich configuration with Nelder-Mead
    try:
        res = minimize(compute_1d_cost, best_1d_positions, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success and res.fun < best_1d_cost:
            best_1d_positions = res.x
            best_1d_cost = res.fun
    except:
        pass

    # Search for better 1D configuration
    rng = np.random.default_rng(42)

    while time.time() - start_time < deadline - 30.0:
        init_pos = rng.uniform(-0.5, 1.5, N)

        remaining = deadline - 30.0 - (time.time() - start_time)
        if remaining < 5.0:
            break

        t_limit = min(10.0, remaining)

        cand_pos, cand_cost = optimize_1d(init_pos, time_limit=t_limit)

        if cand_cost < best_1d_cost:
            best_1d_cost = cand_cost
            best_1d_positions = cand_pos.copy()

            # Polish the new best with Nelder-Mead
            if time.time() - start_time < deadline - 30.0:
                try:
                    res = minimize(compute_1d_cost, best_1d_positions, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.success and res.fun < best_1d_cost:
                        best_1d_positions = res.x
                        best_1d_cost = res.fun
                except:
                    pass

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = best_1d_positions[i_flat]
    base_y = best_1d_positions[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions


if __name__ == "__main__":
    # Quick test
    res = kakeya_needle_3d8()
    print(res.shape)
