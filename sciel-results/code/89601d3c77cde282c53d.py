"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, minimize as _minimize_nm

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

def _relax_circles(circles, w, h, iterations=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Uses iterative pairwise separation with boundary clamping.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    # Ensure radii fit in the rectangle
    max_r = min(w, h) / 2.0
    rad = np.minimum(rad, max_r * 0.99)

    for it in range(iterations):
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

        # 2. Resolve overlaps - multiple passes for convergence
        for _pass in range(3):
            any_overlap = False
            for i in range(n):
                dx = pos[i+1:, 0] - pos[i, 0]
                dy = pos[i+1:, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[i+1:]
                overlap = min_dist - dist
                
                for k in range(len(overlap)):
                    if overlap[k] > 1e-10:
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
                        any_overlap = True
            
            if not any_overlap:
                break

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

def _polish(circles, w, h, deadline, max_iters=100):
    """
    High-precision polish pass with aggressive penalty and tight tolerances.
    """
    if time.time() > deadline - 3.0:
        return circles
    
    n = len(circles)
    
    # Run penalty optimization with very high beta for tight convergence
    result = _penalty_optimize(circles, w, h, deadline, beta_start=500.0, beta_end=50000.0, n_beta_steps=8, maxiter=200)
    
    if result is not None:
        if _validate(result, w, h):
            return result
    
    return circles

def _make_hexagonal_grid(w, h, rows=None):
    """Create initial hexagonal-like arrangement for 21 circles."""
    if rows is None:
        # Choose rows based on aspect ratio
        ar = w / h
        if ar > 1.2:
            rows = [5, 4, 5, 4, 3]
        elif ar < 0.8:
            rows = [4, 5, 4, 5, 3]
        else:
            rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            # Offset alternating rows for hexagonal packing
            avg_w = w / 5.0 
            offset = 0.5 * avg_w if row_i % 2 == 1 else 0
            # Distribute columns evenly across width, with offset
            x = (col_j + 0.5) * w / 5.0 + offset
            # Initial radius estimate
            r = 0.35 * min(w/5.0, h/total_rows)
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _make_grid(w, h, nc, nr, stagger=False):
    """Generate initial circle positions in a grid pattern."""
    positions = []
    for j in range(nr):
        for i in range(nc):
            if len(positions) >= N:
                break
            if stagger and j % 2 == 1:
                x = (i + 0.5) * w / nc + 0.3 * w / nc
            else:
                x = (i + 0.5) * w / nc
            y = (j + 0.5) * h / nr
            positions.append([x, y])
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx - N / 2) * w / N * 0.5
        y = h / 2 + (idx - N / 2) * h / N * 0.5
        positions.append([x, y])
    
    pos = np.array(positions[:N])
    r = 0.35 * min(w/nc, h/nr)
    rad = np.full(N, r)
    return np.column_stack((pos, rad))

def _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=1000.0, n_beta_steps=10, maxiter=100):
    """
    Penalty-based optimization using L-BFGS-B.
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
        
        # Boundary penalty - use smooth max for better gradients
        for i in range(n):
            # Left: violation = r[i] - cx[i] > 0
            viol = r[i] - cx[i]
            if viol > 0:
                pen = beta * viol**2
                obj += pen
                grad[i] -= 2 * beta * viol
                grad[2*n+i] += 2 * beta * viol
            # Right: violation = cx[i] + r[i] - w > 0
            viol = cx[i] + r[i] - w
            if viol > 0:
                pen = beta * viol**2
                obj += pen
                grad[i] += 2 * beta * viol
                grad[2*n+i] += 2 * beta * viol
            # Bottom: violation = r[i] - cy[i] > 0
            viol = r[i] - cy[i]
            if viol > 0:
                pen = beta * viol**2
                obj += pen
                grad[n+i] -= 2 * beta * viol
                grad[2*n+i] += 2 * beta * viol
            # Top: violation = cy[i] + r[i] - h > 0
            viol = cy[i] + r[i] - h
            if viol > 0:
                pen = beta * viol**2
                obj += pen
                grad[n+i] += 2 * beta * viol
                grad[2*n+i] += 2 * beta * viol
        
        # Overlap penalty (vectorized)
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[i+1:]
            overlap = np.maximum(0, min_dist - dist)
            
            # Only compute gradients where overlap > 0
            mask = overlap > 1e-12
            if np.any(mask):
                obj += beta * np.sum(overlap[mask]**2)
                
                d_dist_dx = dx[mask] / dist[mask]
                d_dist_dy = dy[mask] / dist[mask]
                
                # f = overlap^2 = (min_dist - dist)^2
                # df/d(cx[i]) = 2*overlap * d(min_dist)/d(cx[i]) - 2*overlap * d(dist)/d(cx[i])
                # d(min_dist)/d(cx[i]) = 0, d(dist)/d(cx[i]) = dx/dist
                # df/d(cx[i]) = -2*overlap * dx/dist
                # But we minimize, so gradient contribution is -2*beta*overlap*dx/dist
                # Wait: we're adding beta * overlap^2 to objective, so d(obj)/d(cx[i]) = 2*beta*overlap * d(overlap)/d(cx[i])
                # d(overlap)/d(cx[i]) = -d(dist)/d(cx[i]) = -dx/dist
                # So d(obj)/d(cx[i]) = -2*beta*overlap*dx/dist
                # But this is the gradient of the penalty term. The total gradient includes this.
                
                # Actually let me re-derive:
                # obj = -sum(r) + beta * sum_j (overlap_ij)^2
                # d(obj)/d(cx[i]) = sum_j 2*beta*overlap_ij * d(overlap_ij)/d(cx[i])
                # d(overlap_ij)/d(cx[i]) = -d(dist_ij)/d(cx[i]) = -(cx[i]-cx[j])/dist = -dx/dist
                # So d(obj)/d(cx[i]) = -2*beta*sum(overlap*dx/dist)
                
                grad[i] += -2 * beta * np.sum(overlap[mask] * d_dist_dx)
                grad[n+i] += -2 * beta * np.sum(overlap[mask] * d_dist_dy)
                
                # d(overlap_ij)/d(r[i]) = 1
                # d(obj)/d(r[i]) = -1 + 2*beta*sum(overlap)
                grad[2*n+i] += 2 * beta * np.sum(overlap[mask])
                
                # For j: d(overlap_ij)/d(cx[j]) = d(dist)/d(cx[j]) = (cx[j]-cx[i])/dist = -dx/dist
                # d(obj)/d(cx[j]) = 2*beta*overlap * (-dx/dist) ... wait
                # d(overlap)/d(cx[j]) = -d(dist)/d(cx[j]) = -(cx[j]-cx[i])/dist = -(-dx/dist) = dx/dist
                # So d(obj)/d(cx[j]) = 2*beta*overlap*dx/dist
                
                idxs = np.arange(i+1, n)[mask]
                np.add.at(grad, idxs, 2 * beta * overlap[mask] * d_dist_dx)
                np.add.at(grad, n + idxs, 2 * beta * overlap[mask] * d_dist_dy)
                
                # d(overlap_ij)/d(r[j]) = 1
                np.add.at(grad, 2*n + idxs, 2 * beta * overlap[mask])

        return obj, grad

    best_result = None
    best_val = np.inf
    
    for step in range(n_beta_steps):
        if time.time() > deadline - 2.0:
            break
        beta = beta_start * (beta_end / beta_start) ** (step / max(1, n_beta_steps - 1))
        
        try:
            result = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x0,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_result = result
            x0 = result.x
        except Exception:
            pass

    if best_result is not None:
        pos = best_result.x[:2*n].reshape(n, 2)
        rad = best_result.x[2*n:]
        # Clamp to ensure validity
        for i in range(n):
            rad[i] = max(1e-6, min(rad[i], pos[i,0], w - pos[i,0], pos[i,1], h - pos[i,1]))
            pos[i,0] = np.clip(pos[i,0], rad[i], w - rad[i])
            pos[i,1] = np.clip(pos[i,1], rad[i], h - rad[i])
        return np.column_stack((pos, rad))
    return None

def _optimize_with_variable_w(x0, deadline):
    """
    Optimize with variable width w (and h = 2 - w).
    x0 = [w, x1..xN, y1..yN, r1..rN]
    """
    def objective(v):
        w = v[0]
        if w <= 0.01 or w >= 1.99:
            return 1e10
        h = 2.0 - w
        
        xs = v[1:N+1]
        ys = v[N+1:2*N+1]
        rs = v[2*N+1:3*N+1]
        
        if np.any(rs < 0):
            return 1e10
            
        # Boundary penalties
        pen = 0.0
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, rs - (w - xs))**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, rs - (h - ys))**2)
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if d < min_d:
                    pen += (min_d - d)**2
        
        return -np.sum(rs) + 100.0 * pen

    bounds = [(0.01, 1.99)]
    for i in range(N):
        bounds.append((0.0, 2.0))
    for i in range(N):
        bounds.append((0.0, 2.0))
    for i in range(N):
        bounds.append((1e-4, 1.0))
        
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6,
                                  'maxls': 50, 'maxcor': 10})
        return result
    except Exception:
        return None

def _evaluate_aspect(w, deadline, time_budget=5.0):
    """Evaluate a given width w by running the full pipeline."""
    if time.time() > deadline - time_budget:
        return -1.0
    h = 2.0 - w
    if w < 0.3 or h < 0.3:
        return -1.0
    
    circles = _make_hexagonal_grid(w, h)
    circles = _relax_circles(circles, w, h, iterations=200)
    result = _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=5000.0, n_beta_steps=8, maxiter=150)
    if result is not None:
        if _validate(result, w, h):
            return _total_radius(result)
    return -1.0

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None
    best_w = None
    best_h = None

    # Strategy 1: Fixed aspect ratios with hexagonal and grid initializations
    aspect_ratios = [1.0, 1.5, 2.0/3, 0.75, 1.25, 1.333, 0.667, 0.5, 2.0, 1.1, 0.9, 0.85, 1.15]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try hexagonal init
        if time.time() < deadline - 5.0:
            circles = _make_hexagonal_grid(w, h)
            circles = _relax_circles(circles, w, h, iterations=150)
            result = _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=5000.0, n_beta_steps=8, maxiter=150)
            if result is not None:
                if _validate(result, w, h):
                    value = _total_radius(result)
                    if value > best_value:
                        best_value = value
                        best_result = result
                        best_w = w
                        best_h = h

        # Try grid inits
        layouts = [
            (5, 4, True),
            (4, 5, True),
            (5, 4, False),
            (4, 5, False),
            (6, 4, False),
            (4, 6, False),
        ]
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 5.0:
                break
            circles = _make_grid(w, h, nc, nr, stagger)
            circles = _relax_circles(circles, w, h, iterations=150)
            result = _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=5000.0, n_beta_steps=8, maxiter=150)
            if result is not None:
                if _validate(result, w, h):
                    value = _total_radius(result)
                    if value > best_value:
                        best_value = value
                        best_result = result
                        best_w = w
                        best_h = h

    # Strategy 2: Nelder-Mead aspect ratio search around best found
    if best_w is not None and time.time() < deadline - 20.0:
        # Local search around best aspect ratio using Nelder-Mead
        w_best = best_w
        def nm_obj(w_vec):
            w = w_vec[0]
            if w < 0.3 or w > 1.7:
                return 1e10
            val = _evaluate_aspect(w, deadline, time_budget=3.0)
            return -val if val > 0 else 1e10
        
        w0 = np.array([w_best])
        try:
            nm_res = minimize(nm_obj, w0, method='Nelder-Mead',
                            options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6})
            if nm_res.fun < 1e9:
                w_nm = nm_res.x[0]
                h_nm = 2.0 - w_nm
                circles = _make_hexagonal_grid(w_nm, h_nm)
                circles = _relax_circles(circles, w_nm, h_nm, iterations=200)
                result = _penalty_optimize(circles, w_nm, h_nm, deadline, beta_start=50.0, beta_end=20000.0, n_beta_steps=10, maxiter=200)
                if result is not None:
                    if _validate(result, w_nm, h_nm):
                        value = _total_radius(result)
                        if value > best_value:
                            best_value = value
                            best_result = result
                            best_w = w_nm
                            best_h = h_nm
        except Exception:
            pass

    # Strategy 3: Polish the best result found so far
    if best_result is not None and time.time() < deadline - 10.0:
        polished = _polish(best_result, best_w, best_h, deadline)
        if polished is not None:
            if _validate(polished, best_w, best_h):
                value = _total_radius(polished)
                if value > best_value:
                    best_value = value
                    best_result = polished

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
