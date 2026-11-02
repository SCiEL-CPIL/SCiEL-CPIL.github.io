"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

    def is_valid(x, w, h, tol=1e-6):
        """Check if configuration is strictly valid with tolerance."""
        n = len(x) // 3
        # Check bounds
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9:
                return False
            if xi - ri < -tol or xi + ri > w + tol or yi - ri < -tol or yi + ri > h + tol:
                return False
        
        # Check disjointness
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                if dist_sq < (min_dist - tol)**2:
                    return False
        return True

    def check_and_update(x, w, h):
        nonlocal best_value, best_result, best_w, best_h
        if time.time() > deadline:
            return
        
        if is_valid(x, w, h, tol=1e-5):
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()
                best_w = w
                best_h = h

    def softplus(x, beta=1.0):
        """Numerically stable softplus for smooth penalty."""
        return np.log1p(np.exp(-beta * x)) / beta

    def smooth_objective(params, w, h):
        """Smooth objective using softplus penalties for better gradients."""
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        # Ensure positive radii
        rs = np.maximum(rs, 1e-6)
        
        penalty = 0.0
        
        # Boundary penalties using softplus
        for i in range(N):
            # Left boundary: xi - ri >= 0
            penalty += softplus(-(xs[i] - rs[i]), beta=50.0)
            # Right boundary: xi + ri <= w
            penalty += softplus(xs[i] + rs[i] - w, beta=50.0)
            # Bottom boundary: yi - ri >= 0
            penalty += softplus(-(ys[i] - rs[i]), beta=50.0)
            # Top boundary: yi + ri <= h
            penalty += softplus(ys[i] + rs[i] - h, beta=50.0)
        
        # Overlap penalties using softplus
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = rs[i] + rs[j]
                # Penalize when dist < min_dist
                penalty += softplus(min_dist - dist, beta=50.0)
        
        # Maximize sum of radii
        return penalty - np.sum(rs)

    def generate_hexagonal_lattice(w, h, n_circles):
        """Generate initial positions using hexagonal lattice pattern."""
        # For hexagonal packing, rows are offset
        # Try different row counts to fit n_circles
        best_centers = None
        best_score = -np.inf
        
        for n_rows in range(2, 10):
            if time.time() > deadline:
                break
            
            # Calculate row positions
            row_heights = np.linspace(h/(n_rows+1), h - h/(n_rows+1), n_rows) if n_rows > 1 else [h/2]
            
            # Distribute circles across rows
            circles_per_row = [n_circles // n_rows] * n_rows
            for i in range(n_circles % n_rows):
                circles_per_row[i] += 1
            
            centers = []
            for row_idx, row_h in enumerate(row_heights):
                n_in_row = circles_per_row[row_idx]
                if n_in_row == 0:
                    continue
                
                # Hexagonal offset: alternate rows shifted by half spacing
                row_width = w
                spacing = row_width / n_in_row if n_in_row > 0 else row_width
                
                # Offset for hexagonal pattern
                offset = 0.0
                if n_rows > 1:
                    # Shift odd rows by half the horizontal spacing
                    # This creates the hexagonal pattern
                    if row_idx % 2 == 1:
                        offset = spacing / 2.0
                
                for col in range(n_in_row):
                    x = (col + 0.5) * spacing + offset
                    y = row_h
                    # Keep within bounds
                    x = np.clip(x, 0, w)
                    y = np.clip(y, 0, h)
                    centers.append([x, y])
            
            if len(centers) >= n_circles:
                centers = centers[:n_circles]
            
            # Score this configuration by checking minimum distance
            if len(centers) == n_circles:
                min_dist = np.inf
                for i in range(n_circles):
                    for j in range(i+1, n_circles):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        d = np.sqrt(dx*dx + dy*dy)
                        min_dist = min(min_dist, d)
                
                # Also consider distance to boundaries
                for c in centers:
                    min_dist = min(min_dist, c[0], w - c[0], c[1], h - c[1])
                
                if min_dist > best_score:
                    best_score = min_dist
                    best_centers = centers
        
        if best_centers is None:
            # Fallback to simple grid
            a = int(np.ceil(np.sqrt(n_circles * w / h)))
            b = int(np.ceil(n_circles / a))
            best_centers = []
            for i in range(n_circles):
                col = i % a
                row = i // a
                best_centers.append([(col + 0.5) * w / a, (row + 0.5) * h / b])
        
        return np.array(best_centers)

    def polish(x, w, h):
        """Polish solution using smooth objective."""
        if time.time() > deadline:
            return x
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((1e-4, w - 1e-4))
            bounds.append((1e-4, h - 1e-4))
            bounds.append((1e-6, r_max))
            
        try:
            res = minimize(smooth_objective, x, args=(w, h), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if is_valid(res.x, w, h, tol=1e-6):
                if np.sum(res.x[2::3]) >= np.sum(x[2::3]) - 1e-6:
                    return res.x
        except Exception:
            pass
        return x

    # Try multiple aspect ratios with hexagonal initialization
    widths = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Generate hexagonal lattice initialization
        centers = generate_hexagonal_lattice(w, h, N)
        
        # Estimate initial radius from minimum spacing
        min_spacing = np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = np.sqrt(dx*dx + dy*dy)
                min_spacing = min(min_spacing, d)
        for c in centers:
            min_spacing = min(min_spacing, c[0], w - c[0], c[1], h - c[1])
        
        r_est = min_spacing * 0.45  # Conservative initial radius
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = centers[i][0]
            x0[3*i+1] = centers[i][1]
            x0[3*i+2] = r_est
        
        # Optimize with smooth objective
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((1e-4, w - 1e-4))
            bounds.append((1e-4, h - 1e-4))
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(smooth_objective, x0, args=(w, h), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() > deadline:
                break
            
            # Polish and check
            polished = polish(res.x, w, h)
            check_and_update(polished, w, h)
            
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.09
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
        best_w = w
        best_h = h

    # Final validation
    if not is_valid(best_result, best_w, best_h, tol=1e-7):
        for i in range(N):
            best_result[3*i+2] *= 0.99
        
        if not is_valid(best_result, best_w, best_h, tol=1e-7):
            a, b = 7, 3
            w, h = 1.4, 0.6
            r = 0.05
            best_result = np.zeros(3*N)
            for i in range(a):
                for j in range(b):
                    idx = i*b + j
                    best_result[3*idx] = (i + 0.5) * w / a
                    best_result[3*idx+1] = (j + 0.5) * h / b
                    best_result[3*idx+2] = r
            best_w = w
            best_h = h

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
