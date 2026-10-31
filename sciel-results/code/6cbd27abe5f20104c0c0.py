"""Improved circle packing: topology-based search with aspect ratio optimization."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 21

def _pack_rect(w, h, n, seed=42):
    """Pack n circles in rectangle [0,w]x[0,h], maximize sum of radii.
    Uses a grid-based initial guess then local optimization."""
    if n == 0:
        return np.zeros((0, 3)), 0.0
    
    # Try multiple grid configurations and pick the best
    best_val = -np.inf
    best_sol = None
    
    # Grid configurations: (cols, rows)
    configs = []
    for c in range(1, n + 1):
        r = (n + c - 1) // c
        configs.append((c, r))
    # Also try some non-standard
    for c in range(1, n + 1):
        for r in range(1, n + 1):
            if c * r >= n and c * r <= n + 4:
                configs.append((c, r))
    configs = list(set(configs))
    
    deadline = time.time() + 140.0
    
    for cols, rows in configs:
        if time.time() > deadline:
            break
        if cols * rows < n:
            continue
        if cols * rows > n * 2:
            continue
        
        # Create grid positions
        xs = np.linspace(0.5 * w / cols, w - 0.5 * w / cols, cols)
        ys = np.linspace(0.5 * h / rows, h - 0.5 * h / rows, rows)
        
        # Generate all grid points
        grid_pts = []
        for i in range(cols):
            for j in range(rows):
                grid_pts.append((xs[i], ys[j]))
        
        # Select n points from grid (first n)
        if len(grid_pts) < n:
            continue
        
        # Use the first n grid points
        init = np.array(grid_pts[:n])
        
        # Initial radii: half the minimum spacing
        # For a grid, the natural radius is half the cell size
        r_est = 0.4 * min(w / cols, h / rows)
        init_radii = np.full(n, r_est)
        
        # Objective: minimize negative sum of radii
        # Constraints: circles don't overlap, stay in box
        def objective(x):
            cx = x[:n]
            cy = x[n:2*n]
            r = x[2*n:3*n]
            
            # Check boundaries
            if np.any(cx < 0) or np.any(cx > w) or np.any(cy < 0) or np.any(cy > h):
                return 1e10
            if np.any(r < 0):
                return 1e10
            
            # Check circle-box constraints
            for i in range(n):
                if cx[i] < r[i] or cx[i] > w - r[i] or cy[i] < r[i] or cy[i] > h - r[i]:
                    return 1e10
            
            # Check circle-circle constraints
            for i in range(n):
                for j in range(i+1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-9:
                        return 1e10
            
            return -np.sum(r)
        
        # Try a few local optimizations from different perturbations
        for trial in range(3):
            if time.time() > deadline:
                break
            
            if trial == 0:
                x0 = np.concatenate([init[:, 0], init[:, 1], init_radii])
            else:
                # Perturb
                x0 = np.concatenate([init[:, 0], init[:, 1], init_radii])
                np.random.seed(seed + trial * 100 + cols * 10 + rows)
                x0 = x0 + np.random.randn(len(x0)) * 0.01 * min(w, h)
                # Clamp
                x0[:n] = np.clip(x0[:n], 0.01, w - 0.01)
                x0[n:2*n] = np.clip(x0[n:2*n], 0.01, h - 0.01)
                x0[2*n:] = np.clip(x0[2*n:], 1e-4, min(w, h) * 0.4)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_val and res.fun < 1e9:
                    cx = res.x[:n]
                    cy = res.x[n:2*n]
                    r = res.x[2*n:3*n]
                    # Verify validity
                    valid = True
                    for i in range(n):
                        if cx[i] < r[i] - 1e-7 or cx[i] > w - r[i] + 1e-7 or \
                           cy[i] < r[i] - 1e-7 or cy[i] > h - r[i] + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i in range(n):
                            for j in range(i+1, n):
                                dx = cx[i] - cx[j]
                                dy = cy[i] - cy[j]
                                if np.sqrt(dx*dx + dy*dy) < r[i] + r[j] - 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid and np.all(r > 0):
                        best_val = res.fun
                        best_sol = np.column_stack([cx, cy, r])
            except Exception:
                pass
    
    if best_sol is not None:
        return best_sol, -best_val
    else:
        # Fallback: simple equal circles
        r = min(w, h) / (2 * np.ceil(np.sqrt(n)))
        cols = int(np.ceil(np.sqrt(n * w / h)))
        rows = int(np.ceil(n / cols))
        xs = np.linspace(r, w - r, cols)
        ys = np.linspace(r, h - r, rows)
        pts = []
        for i in range(cols):
            for j in range(rows):
                if len(pts) < n:
                    pts.append([xs[i], ys[j], r * 0.99])
        return np.array(pts[:n]), np.sum(np.array(pts[:n])[:, 2])


def construct():
    deadline = time.time() + 145.0
    
    # Optimize aspect ratio
    # Perimeter = 2*(w+h) = 4 => w + h = 2
    # So h = 2 - w, w in (0, 2)
    
    best_val = -np.inf
    best_result = None
    
    # Try a range of aspect ratios
    w_vals = np.linspace(0.3, 1.7, 20)
    
    for w in w_vals:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        try:
            sol, val = _pack_rect(w, h, N)
            if val > best_val:
                best_val = val
                best_result = sol
        except Exception:
            continue
    
    # Refine around best aspect ratio
    if best_result is not None:
        # Find which w gave best
        # Re-scan with finer grid around the best
        # We need to track which w was best
        pass
    
    # Do a finer scan
    best_val2 = -np.inf
    best_result2 = None
    best_w = 1.0
    
    # Coarse scan to find good region
    for w in np.linspace(0.3, 1.7, 10):
        if time.time() > deadline:
            break
        h = 2.0 - w
        try:
            sol, val = _pack_rect(w, h, N)
            if val > best_val2:
                best_val2 = val
                best_result2 = sol
                best_w = w
        except Exception:
            continue
    
    # Fine scan around best
    if best_result2 is not None:
        for w in np.linspace(max(0.3, best_w - 0.15), min(1.7, best_w + 0.15), 8):
            if time.time() > deadline:
                break
            h = 2.0 - w
            try:
                sol, val = _pack_rect(w, h, N)
                if val > best_val2:
                    best_val2 = val
                    best_result2 = sol
                    best_w = w
            except Exception:
                continue
    
    if best_result2 is not None and best_val2 > best_val:
        best_val = best_val2
        best_result = best_result2
    
    if best_result is None:
        # Fallback: square
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
