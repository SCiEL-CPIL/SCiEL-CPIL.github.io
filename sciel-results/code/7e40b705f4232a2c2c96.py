import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def _validate_solution(w, xs, ys, rs):
        """Check if solution is valid: no overlaps, within bounds, positive radii."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False
        if np.any(rs <= 0):
            return False
        # Check bounds
        if np.any(xs < rs) or np.any(xs > w - rs):
            return False
        if np.any(ys < rs) or np.any(ys > h - rs):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if d < rs[i] + rs[j] - 1e-9:
                    return False
        return True

    def _resolve_collisions(w, xs, ys, rs, max_iter=200):
        """Iteratively resolve collisions by adjusting positions and shrinking radii."""
        h = 2.0 - w
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            resolved = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs[i] + rs[j]
                    if d < min_d:
                        resolved = False
                        if d > 1e-10:
                            # Push apart
                            overlap = (min_d - d) * 0.5
                            nx = dx / d
                            ny = dy / d
                            xs[i] += nx * overlap
                            ys[i] += ny * overlap
                            xs[j] -= nx * overlap
                            ys[j] -= ny * overlap
                        else:
                            # Random nudge if exactly overlapping
                            xs[i] += 0.01 * np.random.randn()
                            ys[i] += 0.01 * np.random.randn()
                        # Shrink radii slightly
                        factor = d / (min_d * 1.005) if d > 1e-10 else 0.995
                        rs[i] *= max(factor, 0.995)
                        rs[j] *= max(factor, 0.995)
            # Ensure within bounds by adjusting positions
            for i in range(N):
                # Clamp x
                xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                # Clamp y
                ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                # Ensure radius fits
                rs[i] = min(rs[i], xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = max(rs[i], 1e-6)
            if resolved:
                break
        return xs, ys, rs

    def _generate_hex_init(w, h):
        """Generate initial hexagonal cluster layout."""
        # Hexagonal packing: rows offset
        # For 21 circles, use a hexagonal arrangement
        # Row sizes: 4, 5, 4, 5, 3 = 21
        rows = [4, 5, 4, 5, 3]
        xs = []
        ys = []
        rs = []
        
        # Scale to fit in rectangle
        # Hexagonal spacing
        spacing = min(w, h) / 5.5
        offset = spacing * 0.866  # sqrt(3)/2
        
        y_offset = 0
        for row_idx, count in enumerate(rows):
            y = (row_idx + 0.5) * offset
            # Center the row
            x_start = (w - (count - 1) * spacing) / 2
            for col in range(count):
                x = x_start + col * spacing
                # Compute max radius based on distance to walls and neighbors
                r = min(x, w - x, y, h - y) * 0.9
                # Also consider neighbor distances
                r = min(r, spacing * 0.45)
                xs.append(x)
                ys.append(y)
                rs.append(max(r, 0.05))
        
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def _generate_grid_init(w, h):
        """Generate initial grid layout for given aspect ratio."""
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []
        for i in range(side):
            for j in range(side):
                if len(xs) >= N:
                    break
                cx = (i + 0.5) / side * w
                cy = (j + 0.5) / side * h
                r = min(cx, w - cx, cy, h - cy) * 0.85
                xs.append(cx)
                ys.append(cy)
                rs.append(r)
        while len(xs) < N:
            xs.append(w/2)
            ys.append(h/2)
            rs.append(0.05)
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def _slsqp_optimize(w_init, xs_init, ys_init, rs_init, maxiter=500):
        """Optimize using SLSQP with explicit constraints."""
        h_init = 2.0 - w_init
        x0 = np.concatenate(([w_init], xs_init, ys_init, rs_init))
        
        def objective(v):
            w = v[0]
            h = 2.0 - w
            xs = v[1:N+1]
            ys = v[N+1:2*N+1]
            rs = v[2*N+1:3*N+1]
            return -np.sum(rs)
        
        # Constraints: non-overlap
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def make_overlap_constraint(i, j):
                    def constraint(v):
                        w = v[0]
                        h = 2.0 - w
                        xs = v[1:N+1]
                        ys = v[N+1:2*N+1]
                        rs = v[2*N+1:3*N+1]
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        return d - rs[i] - rs[j]
                    return constraint
                cons.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
        
        # Constraints: within bounds
        for i in range(N):
            # x >= r
            def make_x_min(i):
                def constraint(v):
                    xs = v[1:N+1]
                    rs = v[2*N+1:3*N+1]
                    return xs[i] - rs[i]
                return constraint
            # x <= w - r
            def make_x_max(i):
                def constraint(v):
                    w = v[0]
                    xs = v[1:N+1]
                    rs = v[2*N+1:3*N+1]
                    return w - xs[i] - rs[i]
                return constraint
            # y >= r
            def make_y_min(i):
                def constraint(v):
                    ys = v[N+1:2*N+1]
                    rs = v[2*N+1:3*N+1]
                    return ys[i] - rs[i]
                return constraint
            # y <= h - r
            def make_y_max(i):
                def constraint(v):
                    w = v[0]
                    h = 2.0 - w
                    ys = v[N+1:2*N+1]
                    rs = v[2*N+1:3*N+1]
                    return h - ys[i] - rs[i]
                return constraint
            # r > 0
            def make_r_pos(i):
                def constraint(v):
                    rs = v[2*N+1:3*N+1]
                    return rs[i]
                return constraint
            
            cons.append({'type': 'ineq', 'fun': make_x_min(i)})
            cons.append({'type': 'ineq', 'fun': make_x_max(i)})
            cons.append({'type': 'ineq', 'fun': make_y_min(i)})
            cons.append({'type': 'ineq', 'fun': make_y_max(i)})
            cons.append({'type': 'ineq', 'fun': make_r_pos(i)})
        
        # Bounds
        w_min, w_max = 0.1, 1.9
        bounds = [(w_min, w_max)]
        for i in range(N):
            bounds.append((0.01, 1.99))  # x
        for i in range(N):
            bounds.append((0.01, 1.99))  # y
        for i in range(N):
            bounds.append((0.001, 1.0))  # r
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.success or res.fun < -0.5:  # Check if we got a reasonable solution
                w_best = res.x[0]
                xs_best = res.x[1:N+1]
                ys_best = res.x[N+1:2*N+1]
                rs_best = res.x[2*N+1:3*N+1]
                
                # Polish the solution
                xs_best, ys_best, rs_best = _resolve_collisions(w_best, xs_best, ys_best, rs_best)
                
                if _validate_solution(w_best, xs_best, ys_best, rs_best):
                    return w_best, xs_best, ys_best, rs_best
        except Exception:
            pass
        
        return None

    best_val = -1e10
    best_result = None
    
    # Try multiple aspect ratios with both grid and hex init
    for w_init in [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.9, 1.1]:
        if time.time() > deadline:
            break
            
        h_init = 2.0 - w_init
        
        # Try grid init
        xs_init, ys_init, rs_init = _generate_grid_init(w_init, h_init)
        result = _slsqp_optimize(w_init, xs_init, ys_init, rs_init, maxiter=400)
        if result is not None:
            w_best, xs_best, ys_best, rs_best = result
            val = np.sum(rs_best)
            if val > best_val:
                best_val = val
                best_result = (w_best, xs_best, ys_best, rs_best)
        
        if time.time() > deadline:
            break
        
        # Try hex init
        xs_init, ys_init, rs_init = _generate_hex_init(w_init, h_init)
        result = _slsqp_optimize(w_init, xs_init, ys_init, rs_init, maxiter=400)
        if result is not None:
            w_best, xs_best, ys_best, rs_best = result
            val = np.sum(rs_best)
            if val > best_val:
                best_val = val
                best_result = (w_best, xs_best, ys_best, rs_best)
    
    if best_result is None:
        # Fallback: return a simple valid solution
        w = 1.0
        h = 1.0
        side = 5
        xs = np.array([(i+0.5)/side * w for i in range(side) for _ in range(side)][:N])
        ys = np.array([(j+0.5)/side * h for j in range(side) for _ in range(side)][:N])
        rs = np.array([min(x, w-x, y, h-y)*0.8 for x, y in zip(xs, ys)])
        best_result = (w, xs, ys, rs)
    
    w_best, xs_best, ys_best, rs_best = best_result
    
    # Final validation and cleanup
    xs_best, ys_best, rs_best = _resolve_collisions(w_best, xs_best, ys_best, rs_best)
    
    # Construct output array (x, y, r)
    result = np.column_stack((xs_best, ys_best, rs_best))
    
    return result
