"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# Contact graph for 21 circles in hexagonal-like arrangement
# Rows: 5, 4, 5, 4, 3 = 21 circles
# Each circle touches its neighbors in the hexagonal lattice
def _get_contact_pairs():
    """Return list of (i, j) pairs that should be touching in the hexagonal arrangement."""
    rows = [5, 4, 5, 4, 3]
    pairs = []
    idx = 0
    for row_i, count in enumerate(rows):
        for col_j in range(count):
            # Touch right neighbor in same row
            if col_j < count - 1:
                pairs.append((idx, idx + 1))
            # Touch neighbors in row below
            if row_i + 1 < len(rows):
                below_count = rows[row_i + 1]
                # In hexagonal packing, each circle touches up to 2 in the row below
                if col_j < below_count:
                    pairs.append((idx, idx + count + col_j))
                if col_j + 1 < below_count:
                    pairs.append((idx, idx + count + col_j + 1))
            idx += count
    return pairs

def _get_boundary_circles(rows):
    """Return indices of circles on the boundary of the arrangement."""
    indices = set()
    idx = 0
    for row_i, count in enumerate(rows):
        for col_j in range(count):
            if col_j == 0 or col_j == count - 1:
                indices.add(idx)
            idx += count
    # Top and bottom rows
    top_count = rows[0]
    indices.update(range(top_count))
    bottom_start = sum(rows[:-1])
    indices.update(range(bottom_start, bottom_start + rows[-1]))
    return sorted(indices)

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    # Vectorized overlap check
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _smooth_overlap_penalty(circles, w, h, beta=50.0):
    """Compute smooth penalty for overlaps and boundary violations."""
    n = len(circles)
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Boundary penalty
    pen = 0.0
    for i in range(n):
        if cx[i] - r[i] < 0:
            pen += beta * (cx[i] - r[i])**2
        if cx[i] + r[i] > w:
            pen += beta * (cx[i] + r[i] - w)**2
        if cy[i] - r[i] < 0:
            pen += beta * (cy[i] - r[i])**2
        if cy[i] + r[i] > h:
            pen += beta * (cy[i] + r[i] - h)**2
    
    # Overlap penalty (vectorized)
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[i] + r[i+1:]
        overlap = np.maximum(0, min_dist - dist)
        pen += beta * np.sum(overlap**2)
    
    return pen

def _relax_circles(circles, w, h, iterations=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Uses iterative pairwise separation with boundary clamping.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                overlap = min_dist - dist
                
                if overlap > tol:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = overlap / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = overlap / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    return circles

def _make_hexagonal_grid(n, w, h):
    """Create initial hexagonal-like arrangement for 21 circles with better spacing."""
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    # Use the actual row counts to determine horizontal spacing
    max_cols = max(rows)
    row_height = h / total_rows
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * row_height
        # Center the row horizontally
        row_width = w / max_cols
        # Offset: alternate rows shift by half a cell for hexagonal packing
        if row_i % 2 == 1:
            offset = 0.5 * row_width
        else:
            offset = 0.0
        # Center the row: start position accounts for the number of circles
        start_x = (w - count * row_width) / 2.0 + 0.5 * row_width
        for col_j in range(count):
            x = start_x + col_j * row_width
            # Radius based on minimum spacing
            r = 0.45 * min(row_width, row_height)
            circles.append([x, y, r])
    
    return np.array(circles[:n])

def _make_grid(n, w, h):
    """Fallback grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=1000.0, n_beta_steps=10):
    """
    Penalty-based optimization using L-BFGS-B.
    Faster than constraint-based methods due to vectorized penalty evaluation.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2)) # r

    def objective_and_grad(x, beta, eps):
        """Compute objective and gradient with smooth penalty using soft-min epsilon."""
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        # Objective: -sum(r) + penalty
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty (smooth)
        for i in range(n):
            for (val, gpos, grpos) in [
                (cx[i] - r[i], i, -(2*n+i)),
                (w - cx[i] - r[i], i, -(2*n+i)),
                (cy[i] - r[i], n+i, -(2*n+i)),
                (h - cy[i] - r[i], n+i, -(2*n+i))
            ]:
                if val < eps:
                    # Smooth approximation: use log-sum-exp style
                    t = val / eps
                    pen = beta * eps * eps * np.logaddexp(0, -t)
                    # d/dval of logaddexp(0, -t) = -exp(-t)/(1+exp(-t)) = -sigmoid(t)
                    sig = 1.0 / (1.0 + np.exp(t))
                    dpen_dval = -beta * eps * sig
                    # dpen/dx or dpen/dy
                    if gpos < n:  # x
                        grad[gpos] += dpen_dval
                        grad[2*n+i] += -dpen_dval
                    else:  # y
                        grad[gpos] += dpen_dval
                        grad[2*n+i] += -dpen_dval
                    obj += pen
        
        # Overlap penalty (smooth with soft-min)
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[i+1:]
            
            # Only compute for pairs that might be close
            mask = dist_sq < (min_dist + eps)**2
            if np.any(mask):
                dist = np.sqrt(dist_sq[mask] + 1e-12)
                gap = dist - min_dist[mask]  # positive if separated
                pos = np.where(mask)[0]
                
                for k in range(len(gap)):
                    idx = pos[k]
                    j = i + 1 + idx
                    if gap[k] < eps:
                        # Smooth penalty: soft-max of (0, -gap)
                        t = -gap[k] / eps
                        pen = beta * eps * eps * np.logaddexp(0, t)
                        sig = 1.0 / (1.0 + np.exp(-t))  # sigmoid(t)
                        dpen_dgap = -beta * eps * sig
                        # dpen/dx_i, dpen/dy_i, dpen/dr_i, dpen/dx_j, dpen/dy_j, dpen/dr_j
                        ux = dx[idx] / dist[k]
                        uy = dy[idx] / dist[k]
                        # gap = dist - r_i - r_j
                        # dgap/dx_i = -ux, dgap/dx_j = ux
                        # dgap/dr_i = -1, dgap/dr_j = -1
                        grad[i] += dpen_dgap * (-ux)
                        grad[j] += dpen_dgap * ux
                        grad[n+i] += dpen_dgap * (-uy)
                        grad[n+j] += dpen_dgap * uy
                        grad[2*n+i] += dpen_dgap * (-1)
                        grad[2*n+j] += dpen_dgap * (-1)
                        obj += pen
        
        return obj, grad

    # Progressive beta and eps schedule
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    epsilons = np.geomspace(0.01, 1e-6, n_beta_steps)
    x = x0.copy()
    
    for step in range(n_beta_steps):
        if time.time() > deadline - 3.0:
            break
        
        beta = betas[step]
        eps = epsilons[step]
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta, eps)[0],
                x,
                jac=lambda x: objective_and_grad(x, beta, eps)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < 0:
                x = res.x
        except Exception:
            pass
    
    new_circles = np.column_stack((x[:n], x[n:2*n], x[2*n:]))
    
    # Relax to ensure validity
    new_circles = _relax_circles(new_circles, w, h, iterations=200)
    
    if _validate(new_circles, w, h, tol=1e-5):
        return new_circles
    
    return None

def _high_precision_polish(circles, w, h, deadline):
    """
    Final high-precision polish with progressive smooth-max epsilon.
    Uses very tight tolerances and small eps to resolve near-tangent contacts.
    """
    if time.time() > deadline - 8.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Tighter bounds
    bounds = []
    for i in range(n):
        bounds.append((circles[i, 2] - 1e-8, w - circles[i, 2] + 1e-8)) # x
        bounds.append((circles[i, 2] - 1e-8, h - circles[i, 2] + 1e-8)) # y
    for i in range(n):
        bounds.append((circles[i, 2] * 0.99, min(w, h) / 2)) # r

    def objective_and_grad(x, beta, eps):
        """High-precision objective with smooth penalties."""
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty
        for i in range(n):
            for (val, gpos) in [
                (cx[i] - r[i], i),
                (w - cx[i] - r[i], i),
                (cy[i] - r[i], n+i),
                (h - cy[i] - r[i], n+i)
            ]:
                if val < eps:
                    t = val / eps
                    pen = beta * eps * eps * np.logaddexp(0, -t)
                    sig = 1.0 / (1.0 + np.exp(t))
                    dpen_dval = -beta * eps * sig
                    grad[gpos] += dpen_dval
                    grad[2*n+i] += -dpen_dval
                    obj += pen
        
        # Overlap penalty
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[i+1:]
            
            mask = dist_sq < (min_dist + eps)**2
            if np.any(mask):
                dist = np.sqrt(dist_sq[mask] + 1e-12)
                gap = dist - min_dist[mask]
                pos = np.where(mask)[0]
                
                for k in range(len(gap)):
                    idx = pos[k]
                    j = i + 1 + idx
                    if gap[k] < eps:
                        t = -gap[k] / eps
                        pen = beta * eps * eps * np.logaddexp(0, t)
                        sig = 1.0 / (1.0 + np.exp(-t))
                        dpen_dgap = -beta * eps * sig
                        ux = dx[idx] / dist[k]
                        uy = dy[idx] / dist[k]
                        grad[i] += dpen_dgap * (-ux)
                        grad[j] += dpen_dgap * ux
                        grad[n+i] += dpen_dgap * (-uy)
                        grad[n+j] += dpen_dgap * uy
                        grad[2*n+i] += dpen_dgap * (-1)
                        grad[2*n+j] += dpen_dgap * (-1)
                        obj += pen
        
        return obj, grad

    # Progressive schedule: start with moderate eps, tighten to very small
    epsilons = [0.001, 1e-4, 1e-5, 1e-6, 1e-7]
    betas = [100.0, 500.0, 1000.0, 2000.0, 5000.0]
    x = x0.copy()
    
    for step in range(len(epsilons)):
        if time.time() > deadline - 3.0:
            break
        
        eps = epsilons[step]
        beta = betas[step]
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta, eps)[0],
                x,
                jac=lambda x: objective_and_grad(x, beta, eps)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.success or res.fun < 0:
                x = res.x
        except Exception:
            pass
    
    new_circles = np.column_stack((x[:n], x[n:2*n], x[2*n:]))
    
    # Final relaxation
    new_circles = _relax_circles(new_circles, w, h, iterations=300)
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    
    return None

def _final_micro_polish(circles, w, h, deadline):
    """
    Ultra-tight final polish with very small eps to extract maximum precision.
    Only runs if there's enough time remaining.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Very tight bounds
    bounds = []
    for i in range(n):
        bounds.append((circles[i, 2] - 1e-9, w - circles[i, 2] + 1e-9))
        bounds.append((circles[i, 2] - 1e-9, h - circles[i, 2] + 1e-9))
    for i in range(n):
        bounds.append((circles[i, 2] * 0.999, min(w, h) / 2))

    def objective_and_grad(x, beta, eps):
        """Ultra-tight objective with smooth penalties."""
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty
        for i in range(n):
            for (val, gpos) in [
                (cx[i] - r[i], i),
                (w - cx[i] - r[i], i),
                (cy[i] - r[i], n+i),
                (h - cy[i] - r[i], n+i)
            ]:
                if val < eps:
                    t = val / eps
                    pen = beta * eps * eps * np.logaddexp(0, -t)
                    sig = 1.0 / (1.0 + np.exp(t))
                    dpen_dval = -beta * eps * sig
                    grad[gpos] += dpen_dval
                    grad[2*n+i] += -dpen_dval
                    obj += pen
        
        # Overlap penalty
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[i+1:]
            
            mask = dist_sq < (min_dist + eps)**2
            if np.any(mask):
                dist = np.sqrt(dist_sq[mask] + 1e-14)
                gap = dist - min_dist[mask]
                pos = np.where(mask)[0]
                
                for k in range(len(gap)):
                    idx = pos[k]
                    j = i + 1 + idx
                    if gap[k] < eps:
                        t = -gap[k] / eps
                        pen = beta * eps * eps * np.logaddexp(0, t)
                        sig = 1.0 / (1.0 + np.exp(-t))
                        dpen_dgap = -beta * eps * sig
                        ux = dx[idx] / dist[k]
                        uy = dy[idx] / dist[k]
                        grad[i] += dpen_dgap * (-ux)
                        grad[j] += dpen_dgap * ux
                        grad[n+i] += dpen_dgap * (-uy)
                        grad[n+j] += dpen_dgap * uy
                        grad[2*n+i] += dpen_dgap * (-1)
                        grad[2*n+j] += dpen_dgap * (-1)
                        obj += pen
        
        return obj, grad

    # Ultra-tight schedule
    epsilons = [1e-6, 1e-7, 1e-8]
    betas = [5000.0, 10000.0, 20000.0]
    x = x0.copy()
    
    for step in range(len(epsilons)):
        if time.time() > deadline - 2.0:
            break
        
        eps = epsilons[step]
        beta = betas[step]
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta, eps)[0],
                x,
                jac=lambda x: objective_and_grad(x, beta, eps)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.success or res.fun < 0:
                x = res.x
        except Exception:
            pass
    
    new_circles = np.column_stack((x[:n], x[n:2*n], x[2*n:]))
    
    # Minimal relaxation to ensure validity
    new_circles = _relax_circles(new_circles, w, h, iterations=100)
    
    if _validate(new_circles, w, h, tol=1e-8):
        return new_circles
    
    return None

def _slsqp_polish(circles, w, h, deadline):
    """
    Final polish using SLSQP with soft constraints (penalty-based) for robustness.
    Uses a large penalty weight to enforce constraints while maintaining differentiability.
    """
    if time.time() > deadline - 8.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])

    def objective(x):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        # Primary objective: maximize sum of radii
        obj = -np.sum(r)
        
        # Soft boundary penalties (quadratic)
        pen = 0.0
        for i in range(n):
            pen += max(0, r[i] - cx[i])**2
            pen += max(0, cx[i] + r[i] - w)**2
            pen += max(0, r[i] - cy[i])**2
            pen += max(0, cy[i] + r[i] - h)**2
        
        # Soft overlap penalties (quadratic)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                overlap = max(0, r[i] + r[j] - dist)
                pen += overlap**2
        
        return obj + 1e6 * pen

    bounds = []
    for i in range(n):
        bounds.append((1e-8, w - 1e-8))
        bounds.append((1e-8, h - 1e-8))
    for i in range(n):
        bounds.append((1e-8, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12}
        )

        if res.success:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            # Relax to clean up any numerical issues
            new_circles = _relax_circles(new_circles, w, h, iterations=100)
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass

    return None

def _quick_pack(circles, w, h, deadline):
    """Quick optimization pipeline for a given aspect ratio."""
    if time.time() > deadline - 15.0:
        return None
    
    try:
        # Initialize with hexagonal grid
        circles = _make_hexagonal_grid(N, w, h)
        
        # Relax to ensure validity
        circles = _relax_circles(circles, w, h, iterations=200)
        
        if not _validate(circles, w, h):
            return None
        
        current_value = _total_radius(circles)
        
        # Penalty-based optimization (fast)
        if time.time() < deadline - 25.0:
            optimized = _penalty_optimize(circles, w, h, deadline)
            if optimized is not None:
                circles = optimized
                current_value = _total_radius(circles)
        
        # High-precision polish
        if time.time() < deadline - 15.0:
            polished = _high_precision_polish(circles, w, h, deadline)
            if polished is not None:
                circles = polished
                current_value = _total_radius(circles)
        
        return circles, current_value
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None
    best_ar = None

    # Phase 1: Coarse aspect ratio search with quick optimization
    coarse_ratios = [1.0, 0.8, 1.25, 0.65, 1.5, 0.5, 1.8, 0.4, 2.2]
    
    for ar in coarse_ratios:
        if time.time() > deadline - 40.0:
            break
        
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 40.0:
                break
            
            result = _quick_pack(None, ww, hh, deadline)
            if result is not None:
                circles, val = result
                if val > best_value:
                    best_value = val
                    best_result = circles.copy()
                    best_ar = (ww, hh)
    
    # Phase 2: Nelder-Mead refinement around best aspect ratio
    if best_ar is not None and time.time() < deadline - 30.0:
        from scipy.optimize import minimize as scipy_minimize
        
        def _aspect_objective(params):
            """Objective for Nelder-Mead: negative total radius for given aspect ratio."""
            if time.time() > deadline - 25.0:
                return 1e10  # timeout penalty
            
            ww, hh = params
            if ww < 0.2 or ww > 5.0 or hh < 0.2 or hh > 5.0:
                return 1e10
            
            # Check perimeter constraint
            if 2*(ww + hh) > 4.0 + 1e-6:
                return 1e10
            
            result = _quick_pack(None, ww, hh, deadline)
            if result is None:
                return 1e10
            
            circles, val = result
            return -val
        
        # Start from best found aspect ratio
        x0 = np.array([best_ar[0], best_ar[1]])
        
        try:
            nm_result = scipy_minimize(
                _aspect_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 15, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if nm_result.fun < -best_value + 1e-6:  # improved
                ww, hh = nm_result.x
                if 0.2 < ww < 5.0 and 0.2 < hh < 5.0 and 2*(ww+hh) <= 4.0 + 1e-6:
                    result = _quick_pack(None, ww, hh, deadline)
                    if result is not None:
                        circles, val = result
                        if val > best_value:
                            best_value = val
                            best_result = circles.copy()
                            best_ar = (ww, hh)
        except Exception:
            pass
    
    # Phase 3: Final deep polish on the best result
    if best_result is not None and time.time() < deadline - 15.0:
        ww, hh = best_ar
        
        # SLSQP polish
        if time.time() < deadline - 12.0:
            polished = _slsqp_polish(best_result, ww, hh, deadline)
            if polished is not None:
                best_result = polished
                best_value = _total_radius(best_result)
        
        # Ultra-tight micro polish
        if time.time() < deadline - 8.0:
            polished = _final_micro_polish(best_result, ww, hh, deadline)
            if polished is not None:
                best_result = polished
                best_value = _total_radius(best_result)
    
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
