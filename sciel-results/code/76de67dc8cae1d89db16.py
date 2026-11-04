"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
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

def _relax_circles(circles, w, h, iterations=1000, tol=1e-9):
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
                if overlap[k] > tol:
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

def _make_hexagonal_grid(w, h, jitter=0.0):
    """Create initial hexagonal-like arrangement for 21 circles."""
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    # Calculate optimal radius estimate
    # Area of rect = w*h. Sum of areas of circles <= w*h
    # Approx radius r ~ sqrt(w*h / (n * pi))
    r_est = np.sqrt(w*h / (N * np.pi)) * 0.9
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            # Offset alternating rows for hexagonal packing
            offset = 0.5 * (w / 5.0) if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * w / 5.0 + offset
            
            # Add jitter if requested
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
                
            # Use a slightly smaller radius to ensure initial validity
            r = 0.8 * r_est
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _make_grid(w, h, nc, nr, stagger=False, jitter=0.0):
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
            
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
                
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

def _smooth_penalty_objective(x, w, h, beta, eps):
    """
    Compute objective and gradient using smooth log-sum-exp penalties.
    x = [x1..xN, y1..yN, r1..rN]
    """
    n = N
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    obj = -np.sum(r)
    grad = np.zeros_like(x)
    grad[2*n:] = -1.0
    
    # Boundary penalties (smooth)
    # We want g(x) = cx - r >= 0, etc.
    # Penalty for g < 0 is beta * eps * log(1 + exp(-g/eps))
    # d(pen)/d(g) = -beta * sigmoid(-g/eps)
    
    def add_boundary_penalty(val, idx_pos, idx_r, sign_pos, sign_r):
        # val is the constraint value (should be >= 0)
        if val < 2*eps: # Only compute if close to boundary to save time/numerical stability
            t = -val / eps
            # log(1 + exp(t))
            pen = beta * eps * np.logaddexp(0, t)
            # d(pen)/d(val) = beta * eps * (exp(t) / (1+exp(t))) * (-1/eps) = -beta * sigmoid(-t)
            # sigmoid(-t) = 1 / (1 + exp(t))
            sig = 1.0 / (1.0 + np.exp(t))
            dpen_dval = -beta * sig
            
            obj_contrib = pen
            grad_contrib_pos = dpen_dval * sign_pos
            grad_contrib_r = dpen_dval * sign_r
            
            return obj_contrib, grad_contrib_pos, grad_contrib_r
        return 0.0, 0.0, 0.0

    # Left: cx - r >= 0
    val = cx - r
    for i in range(n):
        o, gp, gr = add_boundary_penalty(val[i], i, 2*n+i, 1.0, -1.0)
        obj += o
        grad[i] += gp
        grad[2*n+i] += gr
        
    # Right: w - cx - r >= 0
    val = w - cx - r
    for i in range(n):
        o, gp, gr = add_boundary_penalty(val[i], i, 2*n+i, -1.0, -1.0)
        obj += o
        grad[i] += gp
        grad[2*n+i] += gr
        
    # Bottom: cy - r >= 0
    val = cy - r
    for i in range(n):
        o, gp, gr = add_boundary_penalty(val[i], n+i, 2*n+i, 1.0, -1.0)
        obj += o
        grad[n+i] += gp
        grad[2*n+i] += gr
        
    # Top: h - cy - r >= 0
    val = h - cy - r
    for i in range(n):
        o, gp, gr = add_boundary_penalty(val[i], n+i, 2*n+i, -1.0, -1.0)
        obj += o
        grad[n+i] += gp
        grad[2*n+i] += gr

    # Overlap penalties (smooth)
    # dist(i,j) - r_i - r_j >= 0
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq + 1e-12)
        min_dist = r[i] + r[i+1:]
        
        val = dist - min_dist
        for k in range(len(val)):
            if val[k] < 2*eps:
                t = -val[k] / eps
                pen = beta * eps * np.logaddexp(0, t)
                sig = 1.0 / (1.0 + np.exp(t))
                dpen_dval = -beta * sig
                
                # d(val)/d(cx_i) = dx[k] / dist[k]
                # d(val)/d(cx_j) = -dx[k] / dist[k]
                # d(val)/d(r_i) = -1
                # d(val)/d(r_j) = -1
                
                ux = dx[k] / dist[k]
                uy = dy[k] / dist[k]
                
                obj += pen
                
                j = i + 1 + k
                grad[i] += dpen_dval * ux
                grad[j] += dpen_dval * (-ux)
                grad[n+i] += dpen_dval * uy
                grad[n+j] += dpen_dval * (-uy)
                grad[2*n+i] += dpen_dval * (-1)
                grad[2*n+j] += dpen_dval * (-1)

    return obj, grad

def _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=10000.0, n_beta_steps=15):
    """
    Penalty-based optimization using L-BFGS-B with smooth log-sum-exp penalties.
    """
    if time.time() > deadline - 5.0:
        return None

    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2)) # r

    def objective(x):
        return _smooth_penalty_objective(x, w, h, beta, eps)[0]
    
    def gradient(x):
        return _smooth_penalty_objective(x, w, h, beta, eps)[1]

    best_result = None
    best_val = np.inf
    
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    epsilons = np.geomspace(0.05, 1e-4, n_beta_steps)
    x = x0.copy()
    
    for step in range(n_beta_steps):
        if time.time() > deadline - 3.0:
            break
        
        beta = betas[step]
        eps = epsilons[step]
        
        try:
            result = minimize(
                objective,
                x,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_result = result
            x = result.x
        except Exception:
            pass

    if best_result is not None:
        pos = best_result.x[:2*n].reshape(n, 2)
        rad = best_result.x[2*n:]
        return np.column_stack((pos, rad))
    return None

def _high_precision_polish(circles, w, h, deadline):
    """
    High-precision polish using SLSQP with explicit constraints.
    """
    if time.time() > deadline - 10.0:
        return None
    
    n = N
    x0 = circles.flatten()
    
    # Bounds
    bounds = []
    for i in range(n):
        r_i = circles[i, 2]
        bounds.append((max(1e-6, r_i * 0.9), w - max(1e-6, r_i * 0.9)))
        bounds.append((max(1e-6, r_i * 0.9), h - max(1e-6, r_i * 0.9)))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))

    def boundary_constraints(x):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        cons = []
        for i in range(n):
            cons.append(cx[i] - r[i])
            cons.append(w - cx[i] - r[i])
            cons.append(cy[i] - r[i])
            cons.append(h - cy[i] - r[i])
        return np.array(cons)

    def overlap_constraints(x):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                cons.append(dist - r[i] - r[j])
        return np.array(cons)

    def all_constraints(x):
        return np.concatenate([boundary_constraints(x), overlap_constraints(x)])

    def objective(x):
        return -np.sum(x[2*n:])

    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        return grad

    try:
        result = minimize(
            objective,
            x0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': all_constraints}],
            options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.success and result.fun < 0:
            new_circles = np.column_stack((result.x[:n], result.x[n:2*n], result.x[2*n:]))
            if _validate(new_circles, w, h, tol=1e-8):
                return new_circles
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None
    best_w = None
    best_h = None

    # Strategy 1: Fixed aspect ratios with diverse initializations
    aspect_ratios = [1.0, 1.5, 2.0/3, 0.75, 1.25, 1.333, 0.667, 0.5, 2.0, 1.1, 0.9, 1.4, 0.8, 1.2, 0.7]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try hexagonal init with and without jitter
        for jitter in [0.0, 0.01, 0.02]:
            if time.time() > deadline - 5.0:
                break
            circles = _make_hexagonal_grid(w, h, jitter=jitter)
            circles = _relax_circles(circles, w, h, iterations=100)
            result = _penalty_optimize(circles, w, h, deadline)
            if result is not None:
                if _validate(result, w, h):
                    value = _total_radius(result)
                    if value > best_value:
                        best_value = value
                        best_result = result.copy()
                        best_w = w
                        best_h = h

        # Try grid inits
        layouts = [
            (5, 4, True),
            (4, 5, True),
            (5, 4, False),
            (4, 5, False),
            (7, 3, True),
            (3, 7, True),
        ]
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 5.0:
                break
            circles = _make_grid(w, h, nc, nr, stagger, jitter=0.01)
            circles = _relax_circles(circles, w, h, iterations=100)
            result = _penalty_optimize(circles, w, h, deadline)
            if result is not None:
                if _validate(result, w, h):
                    value = _total_radius(result)
                    if value > best_value:
                        best_value = value
                        best_result = result.copy()
                        best_w = w
                        best_h = h

    # Strategy 2: Variable aspect ratio optimization (if time permits)
    if best_result is not None and time.time() < deadline - 30.0:
        # Try to optimize width w around the current best
        for w_init in [best_w * 0.9, best_w * 1.1, best_w]:
            if time.time() > deadline - 20.0:
                break
            
            h_init = 2.0 - w_init
            if h_init <= 0.1 or w_init <= 0.1: continue
            
            circles = _make_hexagonal_grid(w_init, h_init, jitter=0.01)
            circles = _relax_circles(circles, w_init, h_init, iterations=100)
            
            # We need a version of penalty optimize that includes w in the variable space
            # For simplicity, we just run the standard penalty optimize on the fixed w_init
            # but we can also try to manually adjust w based on the result
            result = _penalty_optimize(circles, w_init, h_init, deadline)
            if result is not None:
                if _validate(result, w_init, h_init):
                    value = _total_radius(result)
                    if value > best_value:
                        best_value = value
                        best_result = result.copy()
                        best_w = w_init
                        best_h = h_init

    # Strategy 3: Polish the best result found so far
    if best_result is not None and time.time() < deadline - 20.0:
        polished = _high_precision_polish(best_result, best_w, best_h, deadline)
        if polished is not None:
            value = _total_radius(polished)
            if value > best_value:
                best_value = value
                best_result = polished

    # Strategy 4: Additional penalty-based polish with higher beta
    if best_result is not None and time.time() < deadline - 5.0:
        result = _penalty_optimize(best_result, best_w, best_h, deadline, 
                                   beta_start=100.0, beta_end=50000.0, n_beta_steps=10)
        if result is not None:
            if _validate(result, best_w, best_h):
                value = _total_radius(result)
                if value > best_value:
                    best_value = value
                    best_result = result

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
