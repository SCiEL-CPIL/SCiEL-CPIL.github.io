import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    N = 8
    w = 1.0 / N
    slopes = np.arange(N, dtype=float) / N
    
    def _compute_3d_volume(x, y):
        """
        Compute exact 3D union volume of 64 sheared tubes.
        """
        bps = {0.0, 1.0}
        
        # Breakpoints from x-dimension
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = slopes[i] - slopes[j]
                if abs(denom) < 1e-15:
                    continue
                for d1 in (0.0, w):
                    for d2 in (0.0, w):
                        c_val = (x[j] - x[i] + d2 - d1) / denom
                        if -1e-12 < c_val < 1.0 + 1e-12:
                            c_val = max(0.0, min(1.0, c_val))
                            bps.add(c_val)
        
        # Breakpoints from y-dimension
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = slopes[i] - slopes[j]
                if abs(denom) < 1e-15:
                    continue
                for d1 in (0.0, w):
                    for d2 in (0.0, w):
                        c_val = (y[j] - y[i] + d2 - d1) / denom
                        if -1e-12 < c_val < 1.0 + 1e-12:
                            c_val = max(0.0, min(1.0, c_val))
                            bps.add(c_val)
        
        bps = sorted(bps)
        
        total_vol = 0.0
        for k in range(len(bps) - 1):
            c0 = bps[k]
            c1 = bps[k + 1]
            h = c1 - c0
            if h < 1e-12:
                continue
            
            c_mid = 0.5 * (c0 + c1)
            
            def union_area(c_val):
                xl = x + c_val * slopes
                xr = xl + w
                yl = y + c_val * slopes
                yr = yl + w
                
                # Events for sweep line
                # (x_coord, type, y_start, y_end)
                # type 1: add, type -1: remove
                events = []
                for i in range(N):
                    for j in range(N):
                        events.append((xl[i], 1, yl[j], yr[j]))
                        events.append((xr[i], -1, yl[j], yr[j]))
                
                # Sort events by x, then by type (remove before add? or add before remove?)
                # For union area, if an interval ends at x and another starts at x, they touch.
                # The area contribution is 0 at that point, so order doesn't matter for the integral measure.
                # However, for the active set, if we remove then add, the active set is empty for an instant.
                # If we add then remove, the active set has both.
                # Since we calculate area between events, the x-width is 0 at the event point.
                # So it doesn't matter.
                events.sort(key=lambda e: (e[0], e[1]))
                
                active_y = []
                area = 0.0
                prev_x = None
                
                for ev in events:
                    ex, typ, yl, yr = ev
                    if prev_x is not None and ex > prev_x:
                        if active_y:
                            # Calculate union length of active y-intervals
                            # Sort active_y by start
                            active_y.sort(key=lambda t: t[0])
                            cur_l, cur_r = active_y[0]
                            total_len = 0.0
                            for idx in range(1, len(active_y)):
                                l, r = active_y[idx]
                                if l <= cur_r + 1e-14:
                                    cur_r = max(cur_r, r)
                                else:
                                    total_len += cur_r - cur_l
                                    cur_l, cur_r = l, r
                            total_len += cur_r - cur_l
                            area += total_len * (ex - prev_x)
                    
                    if typ == 1:
                        active_y.append((yl, yr))
                    else:
                        # Remove the specific interval
                        # Since we might have duplicates, we should remove one instance
                        # But in this geometry, are there duplicate y-intervals?
                        # y[j] + c*s[j] is unique for different j if y[j] are distinct?
                        # If y[j] are not distinct, we might have duplicates.
                        # To be safe, we can use a multiset or just try to remove.
                        # Python list remove removes the first occurrence.
                        try:
                            active_y.remove((yl, yr))
                        except ValueError:
                            pass # Should not happen if logic is correct
                    
                    prev_x = ex
                
                return area
            
            A0 = union_area(c0)
            Am = union_area(c_mid)
            A1 = union_area(c1)
            
            total_vol += h / 6.0 * (A0 + 4 * Am + A1)
        
        return total_vol
    
    def objective(params):
        x = np.zeros(N)
        y = np.zeros(N)
        x[1:] = params[:N-1]
        y[1:] = params[N-1:]
        return _compute_3d_volume(x, y)
    
    best_params = None
    best_vol = float('inf')
    
    # Generate diverse initial guesses
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Symmetric spread
    x_lin = np.linspace(-0.5, 0.5, N)
    x_lin[0] = 0.0
    y_lin = np.linspace(-0.5, 0.5, N)
    y_lin[0] = 0.0
    candidates.append(np.concatenate([x_lin[1:], y_lin[1:]]))
    
    # 2. Clustered
    for scale in [0.1, 0.2, 0.5]:
        x_cl = np.arange(N) * scale - 0.5 * scale * (N-1)
        x_cl[0] = 0.0
        y_cl = np.arange(N) * scale - 0.5 * scale * (N-1)
        y_cl[0] = 0.0
        candidates.append(np.concatenate([x_cl[1:], y_cl[1:]]))
        
    # 3. Random
    for _ in range(20):
        x_r = rng.uniform(-1.0, 1.0, N)
        x_r[0] = 0.0
        y_r = rng.uniform(-1.0, 1.0, N)
        y_r[0] = 0.0
        candidates.append(np.concatenate([x_r[1:], y_r[1:]]))
    
    # Run Nelder-Mead on each candidate
    for p0 in candidates:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                objective, 
                p0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # If no valid result, fallback
    if best_params is None or not np.isfinite(best_vol) or best_vol <= 0:
        x = np.linspace(-0.5, 0.5, N)
        y = np.linspace(-0.5, 0.5, N)
        x[0] = 0.0
        y[0] = 0.0
    else:
        x = np.zeros(N)
        y = np.zeros(N)
        x[1:] = best_params[:N-1]
        y[1:] = best_params[N-1:]
        
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x[i_flat]
    base_y = y[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
