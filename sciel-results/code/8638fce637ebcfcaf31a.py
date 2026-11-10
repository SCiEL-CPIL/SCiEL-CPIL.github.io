# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _union_1d_volume(base_positions, N=8, num_c_samples=200):
    """Compute the 1D union volume (integral over c of union length) for N intervals.
    
    Each interval i at height c occupies [base_i + c*i/N, base_i + c*i/N + 1/N].
    """
    N = int(N)
    n = len(base_positions)
    c_samples = np.linspace(0, 1, num_c_samples)
    
    total_length = 0.0
    for c in c_samples:
        # Compute all interval endpoints
        starts = base_positions + c * np.arange(n) / N
        ends = starts + 1.0 / N
        
        # Sort intervals by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Compute union length by sweeping
        if len(s_sorted) == 0:
            continue
        current_end = e_sorted[0]
        union_len = 0.0
        for k in range(1, len(s_sorted)):
            if s_sorted[k] > current_end:
                union_len += current_end - e_sorted[k-1] if False else current_end - s_sorted[k-1]
                # Actually need to track properly
                break
        # Simpler: compute union length directly
        current_end = e_sorted[0]
        union_len = 0.0
        for k in range(1, len(s_sorted)):
            if s_sorted[k] <= current_end:
                current_end = max(current_end, e_sorted[k])
            else:
                union_len += current_end - s_sorted[k-1]
                current_end = e_sorted[k]
        union_len += current_end - s_sorted[-1]
        total_length += union_len
    
    # Trapezoidal integration
    return total_length / num_c_samples


def _union_1d_volume_fast(base_positions, N=8):
    """Fast exact computation of 1D union volume using event-based method.
    
    For each c, the union of intervals [p_i + c*i/N, p_i + c*i/N + 1/N] can be
    computed exactly. The union length as a function of c is piecewise linear,
    so we can find all breakpoints and integrate exactly.
    """
    N = int(N)
    n = len(base_positions)
    
    # The union length changes when the relative ordering of endpoints changes.
    # Endpoint of interval i: start_i(c) = p_i + c*i/N, end_i(c) = p_i + c*i/N + 1/N
    # Two starts cross when p_i + c*i/N = p_j + c*j/N => c = (p_j - p_i) * N / (i - j)
    # Two ends cross similarly.
    # A start crosses an end when p_i + c*i/N = p_j + c*j/N + 1/N => c = (p_j - p_i + 1/N) * N / (i - j)
    
    # Collect all critical c values in [0, 1]
    critical_cs = set([0.0, 1.0])
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Start i crosses start j
            denom = i - j
            if denom != 0:
                c_val = (base_positions[j] - base_positions[i]) * N / denom
                if 0 <= c_val <= 1:
                    critical_cs.add(c_val)
            # End i crosses end j
            if denom != 0:
                c_val = (base_positions[j] - base_positions[i]) * N / denom
                if 0 <= c_val <= 1:
                    critical_cs.add(c_val)
            # Start i crosses end j
            if denom != 0:
                c_val = (base_positions[j] - base_positions[i] + 1.0/N) * N / denom
                if 0 <= c_val <= 1:
                    critical_cs.add(c_val)
            # End i crosses start j
            if denom != 0:
                c_val = (base_positions[j] - base_positions[i] - 1.0/N) * N / denom
                if 0 <= c_val <= 1:
                    critical_cs.add(c_val)
    
    critical_cs = sorted(critical_cs)
    
    # For each sub-interval, the ordering is constant, so union length is linear in c.
    # We can compute union length at two points in each sub-interval and integrate.
    total_volume = 0.0
    for idx in range(len(critical_cs) - 1):
        c0 = critical_cs[idx]
        c1 = critical_cs[idx + 1]
        if c1 - c0 < 1e-15:
            continue
        
        # Sample at midpoint to get the ordering
        c_mid = (c0 + c1) / 2
        starts = base_positions + c_mid * np.arange(n) / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Compute union length at c0 and c1 using this ordering
        def union_length_at(c):
            starts_c = base_positions + c * np.arange(n) / N
            ends_c = starts_c + 1.0 / N
            # Use the ordering from c_mid (valid in this interval)
            s_s = starts_c[order]
            e_s = ends_c[order]
            
            if len(s_s) == 0:
                return 0.0
            
            current_end = e_s[0]
            ul = 0.0
            for k in range(1, len(s_s)):
                if s_s[k] <= current_end:
                    if e_s[k] > current_end:
                        current_end = e_s[k]
                else:
                    ul += current_end - s_s[k-1]
                    current_end = e_s[k]
            ul += current_end - s_s[-1]
            return max(0, ul)
        
        ul0 = union_length_at(c0)
        ul1 = union_length_at(c1)
        
        # Integrate linear function
        total_volume += (ul0 + ul1) / 2 * (c1 - c0)
    
    return total_volume


def _compute_union_volume_3d(positions, N=8):
    """Compute the 3D union volume as product of 1D union volumes."""
    N = int(N)
    base_x = positions[:, 0]
    base_y = positions[:, 1]
    
    # Get unique base positions for x and y
    # Actually, we need to compute 1D union for x-direction and y-direction separately
    # But the 1D union depends on which indices have which base positions
    
    # For x: tube (i,j) has x-interval [x_{i,j} + c*i/N, x_{i,j} + c*i/N + 1/N]
    # The union in x at height c is the union over all (i,j) of these intervals
    # But since x_{i,j} depends on both i and j, we can't separate simply unless x only depends on i
    
    # For the product structure to work, we need x_{i,j} = f(i) and y_{i,j} = g(j)
    # Then the 3D union volume = (1D union volume in x) * (1D union volume in y)
    
    # Let's assume the product structure and compute accordingly
    # Extract unique base positions per index
    n = int(np.sqrt(len(positions)))
    
    # Reshape to (n, n) grid
    pos_grid = positions.reshape(n, n)
    
    # Check if x depends only on i (row) and y only on j (col)
    x_rows = pos_grid[:, 0, 0]  # x for each i, j=0
    y_cols = pos_grid[0, :, 1]  # y for each j, i=0
    
    # Verify consistency
    x_all = pos_grid[:, :, 0]
    y_all = pos_grid[:, :, 1]
    
    x_consistent = np.allclose(x_all, x_all[:, :1], atol=1e-10)
    y_consistent = np.allclose(y_all, y_all[0:1, :], atol=1e-10)
    
    if x_consistent and y_consistent:
        vol_x = _union_1d_volume_fast(x_rows, N)
        vol_y = _union_1d_volume_fast(y_cols, N)
        return vol_x * vol_y
    else:
        # Fallback: Monte Carlo estimate
        rng = np.random.default_rng(12345)
        n_points = 100000
        a = rng.uniform(0, 1, n_points)
        b = rng.uniform(0, 1, n_points)
        c = rng.uniform(0, 1, n_points)
        
        inside = np.zeros(n_points, dtype=bool)
        for k in range(n * n):
            i = k // n
            j = k % n
            x_k = positions[k, 0]
            y_k = positions[k, 1]
            x_lo = x_k + c * i / N
            x_hi = x_lo + 1.0 / N
            y_lo = y_k + c * j / N
            y_hi = y_lo + 1.0 / N
            inside |= (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
        
        return inside.sum() / n_points


def _objective(params, N=8):
    """Objective function: 3D union volume."""
    n = int(np.sqrt(len(params)))
    # params is flattened positions (n*n*2)
    positions = params.reshape(n * n, 2)
    vol = _compute_union_volume_3d(positions, N)
    return vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses optimization to find base positions that minimize the 3D union volume.
    Exploits the product structure: x depends only on i, y only on j.
    """
    N = 8
    n = N  # 8
    deadline = time.time() + 600  # 10 minute deadline
    
    # Initial guess: linear star construction
    # Place base positions to maximize overlap
    # For 1D: intervals i at [p_i + c*i/N, p_i + c*i/N + 1/N]
    # Optimal: p_i = -i/(2N) + offset, so all intervals center around same point at c=0.5
    
    # Better initial guess: use the Keich construction or a simple symmetric placement
    # Symmetric: p_i = -i/(2N) so at c=0.5, all intervals are centered at 0
    
    base_x_init = -np.arange(n) / (2 * N)
    base_y_init = -np.arange(n) / (2 * N)
    
    # Create initial positions with product structure
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    positions_init = np.stack([base_x_init[i_flat], base_y_init[j_flat]], axis=1)
    
    # Compute initial volume
    vol_init = _compute_union_volume_3d(positions_init, N)
    
    # Optimize x and y separately since they're independent
    def objective_x(x_params, N=N, n=n):
        """Optimize x base positions."""
        x_pos = x_params
        y_pos = base_y_init  # Fixed y
        # Build positions
        pos = np.empty((n * n, 2))
        pos[:, 0] = x_pos[i_flat]
        pos[:, 1] = y_pos[j_flat]
        return _compute_union_volume_3d(pos, N)
    
    def objective_y(y_params, N=N, n=n):
        """Optimize y base positions."""
        x_pos = base_x_init  # Fixed x
        y_pos = y_params
        pos = np.empty((n * n, 2))
        pos[:, 0] = x_pos[i_flat]
        pos[:, 1] = y_pos[j_flat]
        return _compute_union_volume_3d(pos, N)
    
    best_x = base_x_init.copy()
    best_y = base_y_init.copy()
    best_vol = vol_init
    
    # Optimize x
    try:
        result_x = minimize(objective_x, base_x_init.copy(), method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if result_x.fun < best_vol:
            best_x = result_x.x
            best_vol = result_x.fun
    except Exception:
        pass
    
    if time.time() > deadline - 120:
        # Return current best
        pos = np.empty((n * n, 2))
        pos[:, 0] = best_x[i_flat]
        pos[:, 1] = best_y[j_flat]
        return pos
    
    # Optimize y with updated x
    try:
        def objective_y_updated(y_params, N=N, n=n, best_x=best_x):
            x_pos = best_x
            y_pos = y_params
            pos = np.empty((n * n, 2))
            pos[:, 0] = x_pos[i_flat]
            pos[:, 1] = y_pos[j_flat]
            return _compute_union_volume_3d(pos, N)
        
        result_y = minimize(objective_y_updated, base_y_init.copy(), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if result_y.fun < best_vol:
            best_y = result_y.x
            best_vol = result_y.fun
    except Exception:
        pass
    
    # Iterate a few more times
    for iteration in range(5):
        if time.time() > deadline - 60:
            break
        
        # Optimize x
        try:
            def objective_x_iter(x_params, N=N, n=n, best_y=best_y):
                x_pos = x_params
                y_pos = best_y
                pos = np.empty((n * n, 2))
                pos[:, 0] = x_pos[i_flat]
                pos[:, 1] = y_pos[j_flat]
                return _compute_union_volume_3d(pos, N)
            
            result_x = minimize(objective_x_iter, best_x.copy(), method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if result_x.fun < best_vol:
                best_x = result_x.x
                best_vol = result_x.fun
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
        
        # Optimize y
        try:
            def objective_y_iter(y_params, N=N, n=n, best_x=best_x):
                x_pos = best_x
                y_pos = y_params
                pos = np.empty((n * n, 2))
                pos[:, 0] = x_pos[i_flat]
                pos[:, 1] = y_pos[j_flat]
                return _compute_union_volume_3d(pos, N)
            
            result_y = minimize(objective_y_iter, best_y.copy(), method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if result_y.fun < best_vol:
                best_y = result_y.x
                best_vol = result_y.fun
        except Exception:
            pass
    
    # Final positions
    pos = np.empty((n * n, 2))
    pos[:, 0] = best_x[i_flat]
    pos[:, 1] = best_y[j_flat]
    
    return pos


# EVOLVE-BLOCK-END
