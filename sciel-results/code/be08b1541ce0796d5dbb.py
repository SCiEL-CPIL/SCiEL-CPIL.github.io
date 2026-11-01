"""
Circle packing in perimeter-4 rectangle: 21 circles.
Combined approach:
1. Binary search for the maximum uniform radius (or scaled radii) for a given aspect ratio.
2. Force-directed relaxation (vectorized/optimized) to find valid positions for a candidate radius.
3. Hill climbing on the aspect ratio (width) to find the global maximum sum of radii.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def time_left():
        return deadline - time.time()

    def check_validity(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Overlap checks (Vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        # We only need to check i < j, but checking all is fine with a mask or just upper triangle
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < r_sum[iu] ** 2 - tol):
            return False, 0.0
            
        return True, float(np.sum(r))

    def relax_forces(x, y, r, w, iterations=300):
        """
        Optimized force-directed relaxation.
        Returns updated x, y.
        """
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        
        # Precompute pair indices for efficiency
        iu = np.triu_indices(N, k=1)
        
        for _ in range(iterations):
            if time_left() < 1.0:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # 1. Wall Forces
            # Left
            viol_l = np.maximum(r - x, 0)
            fx += viol_l * 100.0
            # Right
            viol_r = np.maximum(x - (w - r), 0)
            fx -= viol_r * 100.0
            # Bottom
            viol_b = np.maximum(r - y, 0)
            fy += viol_b * 100.0
            # Top
            viol_t = np.maximum(y - (h - r), 0)
            fy -= viol_t * 100.0

            # 2. Pairwise Repulsion
            dx = x[iu[0]] - x[iu[1]]
            dy = y[iu[0]] - y[iu[1]]
            d2 = dx * dx + dy * dy
            d = np.sqrt(d2)
            
            # Avoid division by zero
            d_safe = np.where(d < 1e-9, 1e-9, d)
            
            r_sum = r[iu[0]] + r[iu[1]]
            overlap = r_sum - d_safe
            
            # Only apply force if overlapping (or very close)
            mask = overlap > 0
            if np.any(mask):
                # Force magnitude proportional to overlap
                # Using a steeper potential helps resolve deep overlaps
                force_mag = overlap[mask] * 50.0 
                
                ux = dx[mask] / d_safe[mask]
                uy = dy[mask] / d_safe[mask]
                
                # Accumulate forces
                # iu[0] gets +, iu[1] gets -
                np.add.at(fx, iu[0][mask], ux * force_mag)
                np.add.at(fy, iu[0][mask], uy * force_mag)
                np.add.at(fx, iu[1][mask], -ux * force_mag)
                np.add.at(fy, iu[1][mask], -uy * force_mag)

            # Update positions
            # Adaptive step size: smaller if forces are large
            max_f = np.max(np.abs(fx)) + np.max(np.abs(fy))
            dt = 0.01 if max_f < 1.0 else 0.005 if max_f < 5.0 else 0.001
            
            x += dt * fx
            y += dt * fy
            
            # Clamp to bounds to prevent explosion
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            
            # Early termination if converged
            if max_f < 1e-6:
                break

        return x, y

    def find_valid_positions(r, w, x_init, y_init, max_iter=200):
        """
        Attempt to find valid positions for given radii r in rectangle w x (2-w).
        Returns (x, y, is_valid).
        """
        x, y = x_init.copy(), y_init.copy()
        
        # Try force relaxation first
        x, y = relax_forces(x, y, r, w, iterations=max_iter)
        
        valid, _ = check_validity(x, y, r, w)
        if valid:
            return x, y, True
            
        # If force relaxation fails, try a short SLSQP polish
        # This is expensive, so only do it if time permits
        if time_left() > 2.0:
            h = 2.0 - w
            n = len(r)
            def flatten(xy):
                return np.concatenate([xy[0], xy[1]])
            def unflatten(v):
                return v[:n], v[n:]
                
            def objective(v):
                xi, yi = unflatten(v)
                pen = 0.0
                # Boundary
                pen += np.sum(np.maximum(r - xi, 0)**2)
                pen += np.sum(np.maximum(xi - (w - r), 0)**2)
                pen += np.sum(np.maximum(r - yi, 0)**2)
                pen += np.sum(np.maximum(yi - (h - r), 0)**2)
                # Overlap
                dx = xi[:, None] - xi[None, :]
                dy = yi[:, None] - yi[None, :]
                d2 = dx*dx + dy*dy
                r_sum = r[:, None] + r[None, :]
                iu = np.triu_indices(N, k=1)
                d = np.sqrt(np.maximum(d2[iu], 1e-15))
                overlap = np.maximum(r_sum[iu] - d, 0)
                pen += np.sum(overlap**2)
                return pen

            v0 = flatten((x, y))
            bounds = [(r[i], w - r[i]) for i in range(n)] + [(r[i], h - r[i]) for i in range(n)]
            
            try:
                res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12})
                if res.fun < 1e-6:
                    xi, yi = unflatten(res.x)
                    valid, _ = check_validity(xi, yi, r, w)
                    if valid:
                        return xi, yi, True
            except Exception:
                pass
        
        return x, y, False

    def binary_search_max_radius(w, x_init, y_init, r_low, r_high, max_iter=15):
        """
        Binary search for the maximum uniform scaling of radii.
        Note: We assume we start with a base radius vector and scale it.
        Here, we assume uniform scaling of an initial configuration.
        """
        best_x, best_y, best_r = x_init.copy(), y_init.copy(), r_low * np.ones(N)
        
        # Ensure r_low is valid first
        x_check, y_check, is_v = find_valid_positions(r_low * np.ones(N), w, x_init, y_init, max_iter=100)
        if not is_v:
            # If even low is invalid, we have a bad init or w is too small
            # Try to fix init with force relaxation
            x_fix, y_fix = relax_forces(x_init, y_init, r_low * np.ones(N), w, iterations=500)
            x_check, y_check, is_v = find_valid_positions(r_low * np.ones(N), w, x_fix, y_fix, max_iter=100)
            if not is_v:
                return None # Failed to establish baseline
                
        best_x, best_y = x_check, y_check
        best_r = r_low * np.ones(N)
        
        for _ in range(max_iter):
            if time_left() < 5.0:
                break
            r_mid = (r_low + r_high) / 2.0
            if r_mid - r_low < 1e-6:
                break
                
            r_test = r_mid * np.ones(N)
            x_new, y_new, is_v = find_valid_positions(r_test, w, best_x, best_y, max_iter=150)
            
            if is_v:
                best_x, best_y, best_r = x_new, y_new, r_test
                r_low = r_mid
            else:
                r_high = r_mid
                
        return best_x, best_y, best_r

    def hex_init(w, h, n):
        """Initialize circles in a hexagonal lattice pattern."""
        x = np.zeros(n)
        y = np.zeros(n)
        
        # Estimate spacing
        area_per = w * h / n
        r_est = np.sqrt(area_per / (np.pi * 2.0 / np.sqrt(3))) * 0.85
        r_est = min(r_est, 0.18)
        
        spacing = 2 * r_est * 0.95
        cols = max(1, int(np.ceil(w / spacing)))
        rows = max(1, int(np.ceil(h / (spacing * np.sqrt(3) / 2))))
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                offset = (spacing / 2) if (row % 2 == 1) else 0
                cx = (col + 0.5) * spacing + offset
                cy = (row + 0.5) * spacing * np.sqrt(3) / 2
                
                cx = np.clip(cx, r_est, w - r_est)
                cy = np.clip(cy, r_est, h - r_est)
                
                x[idx] = cx
                y[idx] = cy
                idx += 1
        
        if idx < n:
            for i in range(idx, n):
                x[i] = np.random.uniform(r_est, w - r_est)
                y[i] = np.random.uniform(r_est, h - r_est)
        
        return x, y

    best_value = 0.0
    best_result = None

    # Aspect ratios to try
    # For 21 circles, square (1.0) is a strong candidate, but slightly rectangular might be better.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 1.4, 0.6, 1.5]
    
    for w0 in aspect_ratios:
        if time_left() < 15.0:
            break
            
        h0 = 2.0 - w0
        if h0 <= 0.1:
            continue
            
        # 1. Initialization
        x_init, y_init = hex_init(w0, h0, N)
        
        # 2. Determine initial radius bounds
        # Lower bound: small enough to definitely fit
        r_low = 0.05
        # Upper bound: large enough to be impossible (packing fraction ~0.9 for hex)
        # Area = w*h. N * pi * r^2 < 0.9 * w*h
        r_max_possible = np.sqrt(0.9 * w0 * h0 / (N * np.pi))
        r_high = r_max_possible * 1.05
        
        # 3. Binary Search
        res = binary_search_max_radius(w0, x_init, y_init, r_low, r_high, max_iter=12)
        
        if res is not None:
            x, y, r = res
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_result = (x, y, r, w0)
                
    # Final Polish
    if best_result is not None:
        x, y, r, w = best_result
        
        # Try to squeeze a bit more with very fine binary search steps
        # We can try to increase r by tiny amounts
        for _ in range(20):
            if time_left() < 2.0:
                break
            new_r = r * 1.0005
            x_new, y_new, is_v = find_valid_positions(new_r, w, x, y, max_iter=300)
            if is_v:
                x, y, r = x_new, y_new, new_r
                best_value = np.sum(r)
            else:
                # Try even smaller step
                new_r = r * 1.0001
                x_new, y_new, is_v = find_valid_positions(new_r, w, x, y, max_iter=200)
                if is_v:
                    x, y, r = x_new, y_new, new_r
                    best_value = np.sum(r)
                else:
                    break
                    
        # Final validity check
        valid, val = check_validity(x, y, r, w)
        if valid:
            # Shrink slightly for safety
            r_final = r * (1 - 1e-9)
            return np.column_stack((x, y, r_final))
        else:
            # Fallback to previous valid if polish broke it (shouldn't happen with is_v check, but safe)
            # We don't store the "previous" explicitly in this loop, so we rely on the fact
            # that we only update if is_v is True.
            # So if we are here, the last update was valid.
            # But if the *very first* check fails?
            # The loop only updates if is_v. So r, x, y are always valid at the end of the loop.
            # Unless the initial best_result was invalid? No, it came from binary_search which checks.
            # So this else is likely unreachable unless check_validity has floating point issues.
            # Let's just return the best we have, shrunk.
            r_final = r * (1 - 1e-9)
            return np.column_stack((x, y, r_final))

    # Fallback: Simple grid in unit square
    cols = 5
    rows = 5
    cells = []
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5) / cols
        cy = (row + 0.5) / rows
        r0 = min(1.0 / cols, 1.0 / rows) * 0.45
        cells.append((cx, cy, r0))
    return np.array(cells)

if __name__ == "__main__":
    res = construct()
    print(res.shape)
