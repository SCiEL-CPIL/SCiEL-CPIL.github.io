# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_volume_fast(base_x, base_y, N=8):
    """
    Compute the exact union volume exploiting product structure.
    """
    def critical_cs(base_pos, N):
        cs = set()
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = i - j
                if denom == 0:
                    continue
                c_val = N * (base_pos[j] - base_pos[i] + 1.0 / N) / denom
                if 0 < c_val < 1:
                    cs.add(c_val)
        return sorted(cs)

    cx = critical_cs(base_x, N)
    cy = critical_cs(base_y, N)
    all_cs = sorted(set([0.0, 1.0] + cx + cy))

    def union_length_at_c(base_pos, c, N):
        intervals = []
        for i in range(N):
            left = base_pos[i] + c * i / N
            right = left + 1.0 / N
            intervals.append((left, right))
        intervals.sort()
        if not intervals:
            return 0.0
        merged = [intervals[0]]
        for left, right in intervals[1:]:
            if left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        return sum(r - l for l, r in merged)

    total_vol = 0.0
    for k in range(len(all_cs) - 1):
        c_lo = all_cs[k]
        c_hi = all_cs[k + 1]
        if c_hi - c_lo < 1e-15:
            continue
        
        # Use Simpson's rule for exact integration of quadratic product
        c_mid = 0.5 * (c_lo + c_hi)
        
        lx_lo = union_length_at_c(base_x, c_lo, N)
        ly_lo = union_length_at_c(base_y, c_lo, N)
        prod_lo = lx_lo * ly_lo
        
        lx_mid = union_length_at_c(base_x, c_mid, N)
        ly_mid = union_length_at_c(base_y, c_mid, N)
        prod_mid = lx_mid * ly_mid
        
        lx_hi = union_length_at_c(base_x, c_hi, N)
        ly_hi = union_length_at_c(base_y, c_hi, N)
        prod_hi = lx_hi * ly_hi
        
        # Simpson's rule: (h/3) * (f(lo) + 4*f(mid) + f(hi)) where h = (hi-lo)/2
        h = (c_hi - c_lo) / 2.0
        total_vol += (h / 3.0) * (prod_lo + 4.0 * prod_mid + prod_hi)

    X0 = min(base_x[i] for i in range(N))
    X1 = max(base_x[i] + (i + 1) / N for i in range(N))
    Y0 = min(base_y[j] for j in range(N))
    Y1 = max(base_y[j] + (j + 1) / N for j in range(N))

    box_vol = (X1 - X0) * (Y1 - Y0)
    if box_vol <= 1e-9:
        return None, X0, X1, Y0, Y1

    return total_vol, X0, X1, Y0, Y1


def _monte_carlo_volume(positions, N=8, n_mc=500000):
    """Compute MC volume estimate as specified in the task."""
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()

    base_x = positions[:, 0]
    base_y = positions[:, 1]

    X0 = min(base_x[i] for i in range(N))
    X1 = max(base_x[i] + (i + 1) / N for i in range(N))
    Y0 = min(base_y[j] for j in range(N))
    Y1 = max(base_y[j] + (j + 1) / N for j in range(N))

    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return None

    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n_mc)
    b = rng.uniform(Y0, Y1, n_mc)
    c = rng.uniform(0, 1, n_mc)

    inside = np.zeros(n_mc, dtype=bool)
    for k in range(N * N):
        i_k = k // N
        j_k = k % N
        x_k = base_x[k]
        y_k = base_y[k]
        x_lo = x_k + c * i_k / N
        x_hi = x_lo + 1.0 / N
        y_lo = y_k + c * j_k / N
        y_hi = y_lo + 1.0 / N
        mask = (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
        inside |= mask

    frac = np.mean(inside)
    V = frac * box_area
    if not np.isfinite(V) or V <= 0:
        return None
    return V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    """
    N = 8
    deadline = time.time() + 620.0

    def make_positions(base_x, base_y):
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        i_flat = i_grid.ravel()
        j_flat = j_grid.ravel()
        pos_x = base_x[i_flat]
        pos_y = base_y[j_flat]
        return np.stack([pos_x, pos_y], axis=1)

    def evaluate(base_x, base_y):
        V, _, _, _, _ = _compute_volume_fast(base_x, base_y, N)
        if V is None or not np.isfinite(V) or V <= 0:
            return float('inf')
        return V

    # Initialize with Keich construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    best_x = keich.copy()
    best_y = keich.copy()
    best_V = evaluate(best_x, best_y)

    def local_search(base_x, base_y, max_iters=5000):
        curr_x = base_x.copy()
        curr_y = base_y.copy()
        curr_V = evaluate(curr_x, curr_y)

        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
        no_improve = 0
        max_no_improve = 200

        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            if no_improve > max_no_improve:
                break

            dim = np.random.randint(2)
            idx = np.random.randint(N)
            step = np.random.choice(step_sizes) * np.random.choice([-1, 1])

            if dim == 0:
                new_x = curr_x.copy()
                new_x[idx] += step
                new_y = curr_y
            else:
                new_x = curr_x
                new_y = curr_y.copy()
                new_y[idx] += step

            new_V = evaluate(new_x, new_y)
            if new_V < curr_V:
                curr_x, curr_y, curr_V = new_x, new_y, new_V
                no_improve = 0
            else:
                no_improve += 1

        return curr_x, curr_y, curr_V

    n_starts = 20
    for start in range(n_starts):
        if time.time() > deadline - 60:
            break

        if start == 0:
            init_x = best_x.copy()
            init_y = best_y.copy()
        else:
            rng_local = np.random.default_rng(start * 1000 + 42)
            init_x = best_x + rng_local.normal(0, 0.1, N)
            init_y = best_y + rng_local.normal(0, 0.1, N)

        result_x, result_y, result_V = local_search(init_x, init_y, max_iters=3000)

        if result_V < best_V:
            best_x, best_y, best_V = result_x, result_y, result_V

    positions = make_positions(best_x, best_y)
    mc_V = _monte_carlo_volume(positions, N, n_mc=500000)

    if mc_V is not None and np.isfinite(mc_V) and mc_V > 0:
        final_V = mc_V
    else:
        final_V = best_V

    if not np.isfinite(final_V) or final_V <= 0:
        return make_positions(keich, keich)

    return positions


# EVOLVE-BLOCK-END
