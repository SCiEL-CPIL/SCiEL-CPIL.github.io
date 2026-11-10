# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_area_2d(xs, ys, w):
    """
    Compute the union area of axis-aligned squares [x_i, x_i+w] x [y_i, y_i+w].
    Uses sweep line: sort by x, track active y-intervals.
    """
    n = len(xs)
    if n == 0:
        return 0.0
    # Create events: for each square, x_start and x_end
    events = []
    for i in range(n):
        events.append((xs[i], 0, i))      # start
        events.append((xs[i] + w, 1, i))  # end
    events.sort(key=lambda e: (e[0], e[1]))
    
    active = set()
    area = 0.0
    prev_x = events[0][0]
    
    def _active_y_union():
        """Return total length of union of y-intervals for active squares."""
        if not active:
            return 0.0
        intervals = sorted((ys[i], ys[i] + w) for i in active)
        total = 0.0
        cur_start, cur_end = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                total += cur_end - cur_start
                cur_start, cur_end = s, e
        total += cur_end - cur_start
        return total
    
    for x, typ, i in events:
        area += _active_y_union() * (x - prev_x)
        prev_x = x
        if typ == 0:
            active.add(i)
        else:
            active.discard(i)
    
    return area


def _union_volume_exact(positions, N=8):
    """
    Compute the exact union volume by integrating the 2D union area over c in [0,1].
    The cross-section at height c is a set of squares with:
      x_i(c) = base_x[i] + c * i_idx / N
      y_i(c) = base_y[i] + c * j_idx / N
    where i_idx = k // N, j_idx = k % N for tube k.
    
    The union area is piecewise linear in c (changes only when squares' x or y
    boundaries align). We find all critical c values and integrate exactly.
    """
    n = N * N
    w = 1.0 / N
    i_idx = np.arange(n) // N
    j_idx = np.arange(n) % N
    
    base_x = positions[:, 0]
    base_y = positions[:, 1]
    
    # x_i(c) = base_x[i] + c * i_idx[i] / N
    # y_i(c) = base_y[i] + c * j_idx[i] / N
    # Square i at height c: [x_i(c), x_i(c)+w] x [y_i(c), y_i(c)+w]
    
    # Critical c values: where any two x-boundaries align or any two y-boundaries align
    # x_i(c) = x_j(c)  =>  base_x[i] + c*i_idx[i]/N = base_x[j] + c*i_idx[j]/N
    #   => c = (base_x[j] - base_x[i]) / ((i_idx[i] - i_idx[j])/N)  if i_idx[i] != i_idx[j]
    # x_i(c) + w = x_j(c)  =>  c = (base_x[j] - base_x[i] - w) / ((i_idx[i] - i_idx[j])/N)
    # Similarly for y.
    
    crits = [0.0, 1.0]
    for i in range(n):
        for j in range(i + 1, n):
            di = i_idx[i] - i_idx[j]
            dj = j_idx[i] - j_idx[j]
            if di != 0:
                # x_i(c) = x_j(c)
                c1 = (base_x[j] - base_x[i]) / (di / N)
                if 0 <= c1 <= 1:
                    crits.append(c1)
                # x_i(c) + w = x_j(c)
                c2 = (base_x[j] - base_x[i] - w) / (di / N)
                if 0 <= c2 <= 1:
                    crits.append(c2)
                # x_i(c) = x_j(c) + w
                c3 = (base_x[j] - base_x[i] + w) / (di / N)
                if 0 <= c3 <= 1:
                    crits.append(c3)
            if dj != 0:
                # y_i(c) = y_j(c)
                c4 = (base_y[j] - base_y[i]) / (dj / N)
                if 0 <= c4 <= 1:
                    crits.append(c4)
                # y_i(c) + w = y_j(c)
                c5 = (base_y[j] - base_y[i] - w) / (dj / N)
                if 0 <= c5 <= 1:
                    crits.append(c5)
                # y_i(c) = y_j(c) + w
                c6 = (base_y[j] - base_y[i] + w) / (dj / N)
                if 0 <= c6 <= 1:
                    crits.append(c6)
    
    crits = sorted(set(crits))
    
    # Integrate: on each interval [c_a, c_b], the union area is linear in c
    # (since each square's x and y positions are linear in c, and the union
    #  area of axis-aligned rectangles is piecewise linear when no boundaries cross)
    # Actually, the union area is NOT necessarily linear - it's piecewise linear
    # with breakpoints at critical c values. So we evaluate at midpoints.
    
    # For accuracy, evaluate union area at several points in each interval
    # and use trapezoidal rule
    total_vol = 0.0
    for idx in range(len(crits) - 1):
        c_a = crits[idx]
        c_b = crits[idx + 1]
        if c_b - c_a < 1e-15:
            continue
        # Evaluate at c_a, midpoint, c_b
        for c_eval in [c_a, (c_a + c_b) / 2, c_b]:
            if c_eval < 0 or c_eval > 1:
                continue
            xs = base_x + c_eval * i_idx / N
            ys = base_y + c_eval * j_idx / N
            area = _union_area_2d(xs, ys, w)
            if c_eval == c_a:
                a1 = area
            elif c_eval == (c_a + c_b) / 2:
                a2 = area
            else:
                a3 = area
        total_vol += (a1 + 4 * a2 + a3) / 6 * (c_b - c_a)  # Simpson's rule
    
    return total_vol


def _compute_box_and_volume(positions, N=8, n_mc=500000):
    """Compute the Monte Carlo estimate of union volume as specified."""
    n = N * N
    w = 1.0 / N
    i_idx = np.arange(n) // N
    j_idx = np.arange(n) % N
    
    base_x = positions[:, 0]
    base_y = positions[:, 1]
    
    X0 = np.min(base_x)
    X1 = np.max(base_x + (i_idx + 1) / N)
    Y0 = np.min(base_y)
    Y1 = np.max(base_y + (j_idx + 1) / N)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0, box_area
    
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n_mc)
    b = rng.uniform(Y0, Y1, n_mc)
    c = rng.uniform(0, 1, n_mc)
    
    # Check if point (a, b, c) is inside any tube
    # Tube k: [x_k + c*i_k/N, x_k + c*i_k/N + w] x [y_k + c*j_k/N, y_k + c*j_k/N + w]
    inside = np.zeros(n_mc, dtype=bool)
    for k in range(n):
        x_lo = base_x[k] + c * i_idx[k] / N
        y_lo = base_y[k] + c * j_idx[k] / N
        mask = (a >= x_lo) & (a <= x_lo + w) & (b >= y_lo) & (b <= y_lo + w)
        inside |= mask
    
    fraction = np.mean(inside)
    V = fraction * box_area
    return V, box_area


def _keich_construction(N=8):
    """Keich binary construction for initial positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    return np.stack([base_x, base_y], axis=1)


def _objective(params, N=8):
    """
    Objective function for optimization: returns the union volume.
    Uses the exact analytic computation for smoothness.
    """
    positions = params.reshape(-1, 2)
    V = _union_volume_exact(positions, N)
    if V <= 0 or not np.isfinite(V):
        return 1e10
    return V


def _mc_volume(positions, N=8):
    """Monte Carlo volume estimate for final evaluation."""
    V, _ = _compute_box_and_volume(positions, N, n_mc=500000)
    return V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimize the 3D discrete Kakeya set to minimize union volume.
    Uses L-BFGS-B with multiple restarts from different initial points.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    N = 8
    n = N * N
    w = 1.0 / N
    
    # Get initial position from Keich construction
    init_pos = _keich_construction(N)
    
    # Compute initial volume
    V_init = _mc_volume(init_pos, N)
    best_pos = init_pos.copy()
    best_V = V_init
    if best_V <= 0 or not np.isfinite(best_V):
        # Fallback to a simple configuration
        best_pos = np.zeros((n, 2))
        best_V = 1.0
    
    # Generate multiple restart points
    rng = np.random.default_rng(42)
    restarts = []
    
    # Restart 1: Keich construction
    restarts.append(init_pos.copy())
    
    # Add perturbed versions
    for trial in range(5):
        noise_scale = 0.1 / (trial + 1)
        perturbed = init_pos + rng.normal(0, noise_scale, size=(n, 2))
        restarts.append(perturbed)
    
    # Add some structured alternatives
    # Try shifting the Keich positions
    for shift in [0.05, -0.05, 0.1, -0.1]:
        shifted = init_pos.copy()
        shifted[:, 0] += shift
        restarts.append(shifted)
    
    # Optimize each restart
    for r_idx, init in enumerate(restarts):
        if time.time() > deadline - 60:  # Safety margin
            break
        
        params0 = init.flatten()
        
        try:
            result = minimize(
                _objective,
                params0,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_V:
                new_pos = result.x.reshape(-1, 2)
                V_new = _mc_volume(new_pos, N)
                if V_new > 0 and np.isfinite(V_new) and V_new < best_V:
                    best_pos = new_pos.copy()
                    best_V = V_new
        except Exception:
            continue
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 30:
        try:
            params0 = best_pos.flatten()
            result = minimize(
                _objective,
                params0,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-10}
            )
            if result.fun < best_V:
                new_pos = result.x.reshape(-1, 2)
                V_new = _mc_volume(new_pos, N)
                if V_new > 0 and np.isfinite(V_new) and V_new < best_V:
                    best_pos = new_pos.copy()
                    best_V = V_new
        except Exception:
            pass
    
    # Validate result
    V_final = _mc_volume(best_pos, N)
    if V_final <= 0 or not np.isfinite(V_final):
        # Return the Keich construction as fallback
        return _keich_construction(N)
    
    return best_pos


# EVOLVE-BLOCK-END
