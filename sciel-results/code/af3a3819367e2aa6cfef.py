"""Circle packing in perimeter-4 rectangle: maximize sum of 21 radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s limit


def _objective(params, w):
    """Negative of sum of radii plus smooth penalties."""
    h = 2.0 - w
    n = N
    # params: [x_1..x_n, y_1..y_n, r_1..r_n]
    xs = params[:n]
    ys = params[n:2*n]
    rs = params[2*n:3*n]
    
    # Clamp radii to positive
    rs = np.maximum(rs, 1e-8)
    
    # Sum of radii (to maximize)
    sum_r = np.sum(rs)
    
    # Boundary penalties (smooth): circle must be inside [0,w]x[0,h]
    pen_bound = 0.0
    for i in range(n):
        # x boundaries
        pen_bound += max(0.0, rs[i] - xs[i])**2
        pen_bound += max(0.0, rs[i] - (w - xs[i]))**2
        # y boundaries
        pen_bound += max(0.0, rs[i] - ys[i])**2
        pen_bound += max(0.0, rs[i] - (h - ys[i]))**2
    
    # Overlap penalties (smooth): for each pair, penalize if r_i + r_j > dist
    pen_overlap = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                pen_overlap += overlap**2
    
    return -sum_r + 10.0 * pen_bound + 10.0 * pen_overlap


def _objective_vec(params, w):
    """Vectorized version for speed."""
    h = 2.0 - w
    n = N
    xs = params[:n]
    ys = params[n:2*n]
    rs = np.maximum(params[2*n:3*n], 1e-8)
    
    sum_r = np.sum(rs)
    
    # Boundary penalties
    pen_bound = 0.0
    for i in range(n):
        pen_bound += max(0.0, rs[i] - xs[i])**2
        pen_bound += max(0.0, rs[i] - (w - xs[i]))**2
        pen_bound += max(0.0, rs[i] - ys[i])**2
        pen_bound += max(0.0, rs[i] - (h - ys[i]))**2
    
    # Overlap penalties - vectorized
    pen_overlap = 0.0
    for i in range(n):
        dx = xs[i] - xs[i+1:]
        dy = ys[i] - ys[i+1:]
        dists = np.sqrt(dx*dx + dy*dy)
        overlaps = rs[i] + rs[i+1:] - dists
        pos = np.maximum(overlaps, 0.0)
        pen_overlap += np.sum(pos**2)
    
    return -sum_r + 10.0 * pen_bound + 10.0 * pen_overlap


def _optimize_fixed_width(w, x0, time_budget):
    """Optimize positions and radii for fixed width w."""
    h = 2.0 - w
    n = N
    res = minimize(_objective_vec, x0, args=(w,), method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000})
    return res


def _make_initial_grid(w, h, cols, rows):
    """Create grid-based initial positions."""
    n = N
    xs = np.zeros(n)
    ys = np.zeros(n)
    rs = np.zeros(n)
    count = 0
    for j in range(rows):
        for i in range(cols):
            if count >= n:
                break
            xs[count] = (i + 0.5) * w / cols
            ys[count] = (j + 0.5) * h / rows
            rs[count] = 0.5 * min(w / cols, h / rows) * 0.9
            count += 1
    return np.concatenate([xs, ys, rs])


def _make_random_init(w, h, rng):
    """Random initial positions."""
    n = N
    xs = rng.uniform(0.05*w, 0.95*w, n)
    ys = rng.uniform(0.05*h, 0.95*h, n)
    rs = rng.uniform(0.03, 0.08, n)
    return np.concatenate([xs, ys, rs])


def _make_hex_init(w, h):
    """Hexagonal packing initial guess."""
    n = N
    xs = np.zeros(n)
    ys = np.zeros(n)
    rs = np.zeros(n)
    # Try to fit hexagonal pattern
    # Estimate radius from area
    area = w * h
    # 21 circles, hex packing efficiency ~0.9069
    r_est = np.sqrt(area / (21 * np.pi / 0.9069)) * 0.9
    r_est = min(r_est, 0.15)
    
    count = 0
    spacing = 2 * r_est * 1.05  # slight gap
    dy = spacing * np.sqrt(3) / 2
    row = 0
    while count < n:
        x_offset = 0 if row % 2 == 0 else spacing / 2
        col = 0
        while count < n:
            x = x_offset + col * spacing
            y = row * dy + r_est
            if x + r_est <= w and y + r_est <= h:
                xs[count] = x
                ys[count] = y
                rs[count] = r_est
                count += 1
            col += 1
        row += 1
    return np.concatenate([xs, ys, rs])


def construct():
    t0 = time.time()
    best_score = -np.inf
    best_result = None
    
    # Width values to try (w + h = 2)
    # Square is w=1, h=1. Try various aspect ratios.
    widths = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    
    rng = np.random.default_rng(42)
    
    for w in widths:
        if time.time() - t0 > DEADLINE:
            break
        h = 2.0 - w
        
        # Multiple initializations
        inits = []
        
        # Grid 5x5 (take 21)
        inits.append(_make_initial_grid(w, h, 5, 5))
        # Grid 4x6
        inits.append(_make_initial_grid(w, h, 4, 6))
        # Grid 6x4
        inits.append(_make_initial_grid(w, h, 6, 4))
        # Hex
        inits.append(_make_hex_init(w, h))
        # Random
        for _ in range(3):
            inits.append(_make_random_init(w, h, rng))
        
        for x0 in inits:
            if time.time() - t0 > DEADLINE:
                break
            try:
                res = _optimize_fixed_width(w, x0, 10)
                if res.success or res.fun < 0:
                    params = res.x
                    # Extract and validate
                    n = N
                    xs = params[:n]
                    ys = params[n:2*n]
                    rs = np.maximum(params[2*n:3*n], 1e-6)
                    
                    # Check validity
                    valid = True
                    for i in range(n):
                        if xs[i] < -1e-6 or xs[i] > w + 1e-6:
                            valid = False; break
                        if ys[i] < -1e-6 or ys[i] > h + 1e-6:
                            valid = False; break
                        if rs[i] < 1e-6:
                            valid = False; break
                    if valid:
                        for i in range(n):
                            for j in range(i+1, n):
                                dx = xs[i] - xs[j]
                                dy = ys[i] - ys[j]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < rs[i] + rs[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        score = np.sum(rs)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([xs, ys, rs])
            except Exception:
                pass
    
    # Polish pass on best result with tighter tolerances
    if best_result is not None and time.time() - t0 < DEADLINE - 10:
        # Find the width that gave best result
        # Re-optimize with tighter settings
        for w in [0.95, 1.0, 1.05, 1.1]:
            if time.time() - t0 > DEADLINE - 5:
                break
            h = 2.0 - w
            x0 = best_result.flatten().copy()
            try:
                res = minimize(_objective_vec, x0, args=(w,), method='L-BFGS-B',
                               options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
                params = res.x
                n = N
                xs = params[:n]
                ys = params[n:2*n]
                rs = np.maximum(params[2*n:3*n], 1e-6)
                
                valid = True
                for i in range(n):
                    if xs[i] < -1e-6 or xs[i] > w + 1e-6 or ys[i] < -1e-6 or ys[i] > h + 1e-6:
                        valid = False; break
                    if rs[i] < 1e-6:
                        valid = False; break
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            if dist < rs[i] + rs[j] - 1e-6:
                                valid = False; break
                        if not valid: break
                
                if valid:
                    score = np.sum(rs)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([xs, ys, rs])
            except Exception:
                pass
    
    # Final safety: if no valid result, use seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
