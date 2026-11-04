"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n == 0:
        return False
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

def _make_hexagonal_grid(n, w, h, offset_factor=0.5):
    """
    Create initial hexagonal-like arrangement for 21 circles.
    Rows: 5, 4, 5, 4, 3
    """
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    # Estimate radius based on area or spacing
    # Area of rect = w*h. 21 circles.
    # Approx radius ~ sqrt(w*h / 21 / pi) * 0.9
    r_est = 0.35 * min(w/5, h/total_rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        # Offset alternating rows
        # In a hex pack, row i+1 is shifted by half the horizontal spacing
        # Horizontal spacing is roughly w/5
        spacing = w / 5.0
        offset = 0.5 * spacing * offset_factor if row_i % 2 == 1 else 0
        for col_j in range(count):
            # Center the row horizontally
            # If count < 5, we might want to center it
            # Simple approach: distribute evenly across width, with offset
            x = (col_j + 0.5) * spacing + offset
            # Clamp x to be within bounds roughly
            # But let relaxation handle it
            circles.append([x, y, r_est])
    
    return np.array(circles[:n])

def _relax_circles(circles, w, h, iterations=300, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
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

def _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=100.0, n_beta_steps=5):
    """
    Fast penalty-based optimization using L-BFGS-B.
    """
    if time.time() > deadline - 10.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))

    def objective_and_grad(x, beta, eps):
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

    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    epsilons = np.geomspace(0.01, 1e-4, n_beta_steps)
    x = x0.copy()
    
    for step in range(n_beta_steps):
        if time.time() > deadline - 5.0:
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
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < 0:
                x = res.x
        except Exception:
            pass
    
    new_circles = np.column_stack((x[:n], x[n:2*n], x[2*n:]))
    new_circles = _relax_circles(new_circles, w, h, iterations=100)
    
    if _validate(new_circles, w, h, tol=1e-5):
        return new_circles
    
    return None

def _slsqp_polish(circles, w, h, deadline):
    """
    Final polish using SLSQP with hard constraints.
    This is the structural change: rigorous constrained optimization.
    """
    if time.time() > deadline - 15.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])

    def objective(x):
        return -np.sum(x[2*n:])

    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        return grad

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
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
        )

        if res.success or res.fun < 0:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            # SLSQP might violate constraints slightly due to numerical errors
            # So we must validate and potentially relax
            if _validate(new_circles, w, h, tol=1e-6):
                return new_circles
            else:
                # Try to repair with a small relaxation
                repaired = _relax_circles(new_circles, w, h, iterations=50)
                if _validate(repaired, w, h, tol=1e-6):
                    return repaired
    except Exception:
        pass

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    best_value = 0.0
    best_result = None
    candidates = []

    # Aspect ratios to explore
    # Perimeter 4 => w + h = 2
    # w = 2 * ar / (1 + ar), h = 2 / (1 + ar)
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 2.5]
    
    for ar in ratios:
        if time.time() > deadline - 20.0:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        # Try both orientations (w,h) and (h,w) if ar != 1
        orientations = [(w, h)]
        if abs(w - h) > 1e-6:
            orientations.append((h, w))
            
        for (ww, hh) in orientations:
            if time.time() > deadline - 15.0:
                break
            
            try:
                # 1. Initialize
                circles = _make_hexagonal_grid(N, ww, hh)
                
                # 2. Relax
                circles = _relax_circles(circles, ww, hh, iterations=200)
                
                if not _validate(circles, ww, hh):
                    continue
                
                # 3. Fast Penalty Optimization
                if time.time() < deadline - 25.0:
                    optimized = _penalty_optimize(circles, ww, hh, deadline)
                    if optimized is not None:
                        circles = optimized
                
                current_value = _total_radius(circles)
                
                # Store candidate for deep polish
                if current_value > best_value:
                    best_value = current_value
                    best_result = circles.copy()
                
                candidates.append((current_value, circles, ww, hh))
                
            except Exception:
                continue

    # 4. Deep Polish on top candidates
    # Sort candidates by value descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 3 distinct candidates to avoid redundant work
    polished_candidates = []
    seen_keys = set()
    for val, circ, ww, hh in candidates:
        # Simple key: rounded dimensions and value
        key = (round(ww, 2), round(hh, 2), round(val, 4))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        polished_candidates.append((val, circ, ww, hh))
        if len(polished_candidates) >= 3:
            break

    for val, circ, ww, hh in polished_candidates:
        if time.time() > deadline - 10.0:
            break
        
        try:
            polished = _slsqp_polish(circ, ww, hh, deadline)
            if polished is not None:
                new_val = _total_radius(polished)
                if new_val > best_value:
                    best_value = new_val
                    best_result = polished.copy()
        except Exception:
            pass

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
