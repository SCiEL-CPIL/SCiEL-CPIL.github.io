<<<<<<< SEARCH
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
    Vectorized for speed.
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

        # 2. Resolve overlaps (vectorized per circle)
        for i in range(n):
            dx = pos[i+1:, 0] - pos[i, 0]
            dy = pos[i+1:, 1] - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rad[i] + rad[i+1:]
            overlap = min_dist - dist
            
            for k in range(len(overlap)):
                if overlap[k] > 0:
                    if dist[k] < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = overlap[k] / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[i+1+k, 0] += np.cos(angle) * push
                        pos[i+1+k, 1] += np.sin(angle) * push
                    else:
                        push = overlap[k] / 2.0
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[i+1+k, 0] += ux * push
                        pos[i+1+k, 1] += uy * push
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
    """Create initial hexagonal-like arrangement for 21 circles."""
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            # Offset alternating rows for hexagonal packing
            offset = 0.5 * (w / 5) if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * w / 5 + offset
            r = 0.35 * min(w/5, h/total_rows)
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

    def objective_and_grad(x, beta):
        """Compute objective and gradient with penalty."""
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        # Objective: -sum(r) + penalty
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty
        for i in range(n):
            if cx[i] - r[i] < 0:
                pen = beta * (cx[i] - r[i])**2
                obj += pen
                grad[i] += 2 * beta * (cx[i] - r[i])
                grad[2*n+i] += -2 * beta * (cx[i] - r[i])
            if cx[i] + r[i] > w:
                pen = beta * (cx[i] + r[i] - w)**2
                obj += pen
                grad[i] += 2 * beta * (cx[i] + r[i] - w)
                grad[2*n+i] += 2 * beta * (cx[i] + r[i] - w)
            if cy[i] - r[i] < 0:
                pen = beta * (cy[i] - r[i])**2
                obj += pen
                grad[n+i] += 2 * beta * (cy[i] - r[i])
                grad[2*n+i] += -2 * beta * (cy[i] - r[i])
            if cy[i] + r[i] > h:
                pen = beta * (cy[i] + r[i] - h)**2
                obj += pen
                grad[n+i] += 2 * beta * (cy[i] + r[i] - h)
                grad[2*n+i] += 2 * beta * (cy[i] + r[i] - h)
        
        # Overlap penalty (vectorized)
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[i+1:]
            
            # Only compute for pairs that might overlap
            mask = dist_sq < min_dist**2
            if np.any(mask):
                dist = np.sqrt(dist_sq[mask] + 1e-12)
                overlap = min_dist[mask] - dist
                pos = np.where(mask)[0]
                
                for k in range(len(overlap)):
                    idx = pos[k]
                    j = i + 1 + idx
                    if overlap[k] > 0:
                        pen = beta * overlap[k]**2
                        obj += pen
                        # Gradient w.r.t. positions
                        ux = dx[idx] / dist[k]
                        uy = dy[idx] / dist[k]
                        grad[i] += -2 * beta * overlap[k] * ux
                        grad[j] += 2 * beta * overlap[k] * ux
                        grad[n+i] += -2 * beta * overlap[k] * uy
                        grad[n+j] += 2 * beta * overlap[k] * uy
                        # Gradient w.r.t. radii
                        grad[2*n+i] += 2 * beta * overlap[k]
                        grad[2*n+j] += 2 * beta * overlap[k]
        
        return obj, grad

    # Progressive beta schedule
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 3.0:
            break
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
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

def _slsqp_polish(circles, w, h, deadline):
    """
    Final polish using SLSQP with hard constraints on the best result.
    Only run if there's enough time remaining.
    """
    if time.time() > deadline - 8.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])

    def objective(x):
        return -np.sum(x[2*n:])

    constraints = []

    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[n+i] - x[2*n+i]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i]-x[j])**2 + (x[n+i]-x[n+j])**2) - x[2*n+i] - x[2*n+j]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-10}
        )

        if res.success:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try a range of aspect ratios - broader range for elongated rectangles
    ratios = [1.0, 0.85, 1.18, 0.75, 1.35, 0.65, 1.55, 0.55, 1.8, 0.48, 2.0, 0.42, 2.2]
    
    for ar in ratios:
        if time.time() > deadline - 12.0:
            break
        
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Try both orientations
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 12.0:
                break
            
            try:
                # Initialize with hexagonal grid
                circles = _make_hexagonal_grid(N, ww, hh)
                
                # Relax to ensure validity
                circles = _relax_circles(circles, ww, hh, iterations=300)
                
                if not _validate(circles, ww, hh):
                    continue
                
                current_value = _total_radius(circles)
                
                # Penalty-based optimization (fast)
                if time.time() < deadline - 20.0:
                    optimized = _penalty_optimize(circles, ww, hh, deadline)
                    if optimized is not None:
                        circles = optimized
                        current_value = _total_radius(circles)
                
                # SLSQP polish (slower but more precise)
                if time.time() < deadline - 10.0:
                    polished = _slsqp_polish(circles, ww, hh, deadline)
                    if polished is not None:
                        circles = polished
                        current_value = _total_radius(circles)
                
                if current_value > best_value:
                    best_value = current_value
                    best_result = circles.copy()
                    
            except Exception:
                continue

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
=======
"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
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

def _relax_circles(circles, w, h, iterations=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Vectorized for speed.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

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

        for i in range(n):
            dx = pos[i+1:, 0] - pos[i, 0]
            dy = pos[i+1:, 1] - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rad[i] + rad[i+1:]
            overlap = min_dist - dist
            
            for k in range(len(overlap)):
                if overlap[k] > 0:
                    if dist[k] < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = overlap[k] / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[i+1+k, 0] += np.cos(angle) * push
                        pos[i+1+k, 1] += np.sin(angle) * push
                    else:
                        push = overlap[k] / 2.0
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[i+1+k, 0] += ux * push
                        pos[i+1+k, 1] += uy * push
                    moved = True

        if not moved:
            break

    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    return circles

def _make_hexagonal_grid(n, w, h):
    """Create initial hexagonal-like arrangement for 21 circles."""
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            offset = 0.5 * (w / 5) if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * w / 5 + offset
            r = 0.35 * min(w/5, h/total_rows)
            circles.append([x, y, r])
    
    return np.array(circles[:n])

def _smooth_max_penalty(x, w, h, n, logit_scale=50.0):
    """
    Smooth penalty using log-sum-exp for both boundary and overlap violations.
    Returns (penalty, gradient) as a smooth surrogate.
    """
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    grad = np.zeros_like(x)
    pen = 0.0
    
    # Boundary violations as smooth max
    boundary_violations = []
    boundary_grads = []
    
    for i in range(n):
        v1 = r[i] - cx[i]          # left
        v2 = cx[i] + r[i] - w     # right
        v3 = r[i] - cy[i]          # bottom
        v4 = cy[i] + r[i] - h     # top
        
        vals = np.array([v1, v2, v3, v4])
        max_val = np.max(vals)
        
        if max_val > 0:
            # log-sum-exp for smooth max
            shifted = vals - max_val
            log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)))
            pen += log_sum_exp
            
            # Gradient via softmax weights
            weights = np.exp(shifted) / np.sum(np.exp(shifted))
            
            # d(v1)/d(cx) = -1, d(v1)/d(r) = 1
            grad[i] += -weights[0]
            grad[2*n+i] += weights[0]
            
            # d(v2)/d(cx) = 1, d(v2)/d(r) = 1
            grad[i] += weights[1]
            grad[2*n+i] += weights[1]
            
            # d(v3)/d(cy) = -1, d(v3)/d(r) = 1
            grad[n+i] += -weights[2]
            grad[2*n+i] += weights[2]
            
            # d(v4)/d(cy) = 1, d(v4)/d(r) = 1
            grad[n+i] += weights[3]
            grad[2*n+i] += weights[3]
    
    # Overlap violations as smooth max per pair
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist_sq = dx*dx + dy*dy
        min_dist = r[i] + r[i+1:]
        
        # Only consider pairs that are close
        mask = dist_sq < (min_dist * 1.1)**2
        if not np.any(mask):
            continue
        
        idxs = np.where(mask)[0]
        for idx in idxs:
            j = i + 1 + idx
            dist = np.sqrt(dist_sq[idx] + 1e-15)
            violation = min_dist[idx] - dist
            
            if violation > 0:
                # Smooth: use violation directly (it's already smooth when dist > 0)
                pen += violation
                
                ux = dx[idx] / dist
                uy = dy[idx] / dist
                
                # d(violation)/d(cx_i) = -ux, d(violation)/d(cx_j) = ux
                grad[i] += -ux
                grad[j] += ux
                grad[n+i] += -uy
                grad[n+j] += uy
                
                # d(violation)/d(r_i) = 1, d(violation)/d(r_j) = 1
                grad[2*n+i] += 1.0
                grad[2*n+j] += 1.0
    
    return pen, grad

def _penalty_optimize(circles, w, h, deadline, beta_start=1.0, beta_end=50.0, n_beta_steps=8):
    """
    Penalty-based optimization using smooth max penalty and L-BFGS-B.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))

    def objective_and_grad(x, beta):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        pen, pen_grad = _smooth_max_penalty(x, w, h, n)
        obj += beta * pen
        grad += beta * pen_grad
        
        return obj, grad

    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 3.0:
            break
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < 0:
                x = res.x
        except Exception:
            pass
    
    new_circles = np.column_stack((x[:n], x[n:2*n], x[2*n:]))
    new_circles = _relax_circles(new_circles, w, h, iterations=300)
    
    if _validate(new_circles, w, h, tol=1e-5):
        return new_circles
    
    return None

def _slsqp_polish(circles, w, h, deadline):
    """
    Final polish using SLSQP with hard constraints.
    """
    if time.time() > deadline - 8.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])

    def objective(x):
        return -np.sum(x[2*n:])

    constraints = []

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[n+i] - x[2*n+i]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i]-x[j])**2 + (x[n+i]-x[n+j])**2) - x[2*n+i] - x[2*n+j]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )

        if res.success:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass

    return None

def _high_precision_polish(circles, w, h, deadline):
    """
    Very tight final polish using SLSQP with more iterations and tighter tolerances.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])

    def objective(x):
        return -np.sum(x[2*n:])

    constraints = []

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[n+i] - x[2*n+i]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i]-x[j])**2 + (x[n+i]-x[n+j])**2) - x[2*n+i] - x[2*n+j]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-15}
        )

        if res.success:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            if _validate(new_circles, w, h, tol=1e-6):
                return new_circles
    except Exception:
        pass

    return None

def _optimize_with_variable_w(circles, ar_init, deadline):
    """
    Jointly optimize circle positions/radii and aspect ratio.
    Parameterization: w = ar/(1+ar)*2, h = 1/(1+ar)*2 where ar is the variable.
    """
    if time.time() > deadline - 10.0:
        return None

    n = len(circles)
    # x = [cx_0..cx_{n-1}, cy_0..cy_{n-1}, r_0..r_{n-1}, ar]
    ar0 = ar_init
    w0 = ar0 / (1 + ar0) * 2
    h0 = 1 / (1 + ar0) * 2
    
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2], [ar0]])
    
    def objective(x):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:3*n]
        ar = x[3*n]
        
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Check validity quickly
        for i in range(n):
            if cx[i] - r[i] < -1e-6 or cx[i] + r[i] > w + 1e-6:
                return 1e6
            if cy[i] - r[i] < -1e-6 or cy[i] + r[i] > h + 1e-6:
                return 1e6
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                if d < r[i] + r[j] - 1e-6:
                    return 1e6
        
        return -np.sum(r)
    
    # Use a simpler approach: just try a few nearby aspect ratios with the fixed-w optimizer
    best_circles = circles
    best_val = _total_radius(circles)
    
    for delta_ar in [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]:
        if time.time() > deadline - 5.0:
            break
        new_ar = ar_init + delta_ar
        if new_ar < 0.3 or new_ar > 3.0:
            continue
        new_w = new_ar / (1 + new_ar) * 2
        new_h = 1 / (1 + new_ar) * 2
        
        try:
            # Scale circles to fit new rectangle
            scale_w = new_w / w0
            scale_h = new_h / h0
            scaled = circles.copy()
            scaled[:, 0] *= scale_w
            scaled[:, 1] *= scale_h
            scaled[:, 2] *= min(scale_w, scale_h)
            
            scaled = _relax_circles(scaled, new_w, new_h, iterations=200)
            if not _validate(scaled, new_w, new_h):
                continue
            
            val = _total_radius(scaled)
            if val > best_val:
                best_val = val
                best_circles = scaled.copy()
        except Exception:
            continue
    
    return best_circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try a range of aspect ratios
    ratios = [1.0, 0.85, 1.18, 0.75, 1.35, 0.65, 1.55, 0.55, 1.8, 0.48, 2.0, 0.42, 2.2, 0.9, 1.1, 0.8]
    
    for ar in ratios:
        if time.time() > deadline - 15.0:
            break
        
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 15.0:
                break
            
            try:
                circles = _make_hexagonal_grid(N, ww, hh)
                circles = _relax_circles(circles, ww, hh, iterations=300)
                
                if not _validate(circles, ww, hh):
                    continue
                
                current_value = _total_radius(circles)
                
                # Penalty-based optimization with smooth max
                if time.time() < deadline - 25.0:
                    optimized = _penalty_optimize(circles, ww, hh, deadline)
                    if optimized is not None:
                        circles = optimized
                        current_value = _total_radius(circles)
                
                # SLSQP polish
                if time.time() < deadline - 12.0:
                    polished = _slsqp_polish(circles, ww, hh, deadline)
                    if polished is not None:
                        circles = polished
                        current_value = _total_radius(circles)
                
                # High-precision final polish
                if time.time() < deadline - 6.0:
                    hp_polished = _high_precision_polish(circles, ww, hh, deadline)
                    if hp_polished is not None:
                        circles = hp_polished
                        current_value = _total_radius(circles)
                
                if current_value > best_value:
                    best_value = current_value
                    best_result = circles.copy()
                    
            except Exception:
                continue

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
>>>>>>> REPLACE
