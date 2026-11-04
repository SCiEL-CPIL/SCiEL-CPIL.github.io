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

def _relax_circles(circles, w, h, iterations=500):
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

def _make_hexagonal_grid(n, w, h):
    """Create initial hexagonal-like arrangement for 21 circles."""
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            # Offset alternating rows for hexagonal packing
            # Use width/5 as base unit for consistency
            offset = 0.5 * (w / 5) if row_i % 2 == 1 else 0
            # Center the row horizontally
            row_width = count * (w / 5)
            start_x = (w - row_width) / 2
            x = start_x + (col_j + 0.5) * (w / 5)
            
            # Initial radius estimate
            # Hexagonal packing density ~0.9069
            # Area per circle ~ pi*r^2
            # Rectangle area = w*h
            # n*pi*r^2 < 0.9 * w*h  => r < sqrt(0.9*w*h/(n*pi))
            r = 0.9 * min(w/5, h/total_rows) * 0.8
            circles.append([x, y, r])
    
    return np.array(circles[:n])

def _make_grid(n, w, h):
    """Fallback grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=1000.0, n_beta_steps=10, eps_start=1e-2, eps_end=1e-6):
    """
    Penalty-based optimization using L-BFGS-B.
    Uses smooth max (softplus) for constraints to allow gradient-based optimization.
    Progressive smoothing and penalty increase.
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

    def softplus(x, eps):
        return eps * np.log(1 + np.exp(x / eps))

    def objective_and_grad(x, beta, eps):
        """Compute objective and gradient with penalty."""
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        # Objective: -sum(r) + penalty
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty
        # Constraints:
        # x_i >= r_i  => x_i - r_i >= 0  => violation = r_i - x_i
        # w - x_i - r_i >= 0 => violation = x_i + r_i - w
        # y_i >= r_i
        # h - y_i - r_i >= 0
        
        for i in range(n):
            # Left
            viol = r[i] - cx[i]
            if viol > -5*eps: # Only compute if close or violating
                sp = softplus(viol / eps, eps)
                obj += beta * sp
                # d/dx_i (softplus((r_i - x_i)/eps, eps)) = - (1/(1+exp(-(r_i-x_i)/eps)))
                # d/dx_i softplus = - sigmoid(viol/eps)
                sig = 1.0 / (1.0 + np.exp(-viol/eps))
                grad[i] += beta * (-sig)
                # d/dr_i softplus((r_i - x_i)/eps, eps) = + sigmoid(viol/eps)
                grad[2*n + i] += beta * sig
            
            # Right
            viol = cx[i] + r[i] - w
            if viol > -5*eps:
                sp = softplus(viol / eps, eps)
                obj += beta * sp
                sig = 1.0 / (1.0 + np.exp(-viol/eps))
                grad[i] += beta * sig
                grad[2*n + i] += beta * sig

            # Bottom
            viol = r[i] - cy[i]
            if viol > -5*eps:
                sp = softplus(viol / eps, eps)
                obj += beta * sp
                sig = 1.0 / (1.0 + np.exp(-viol/eps))
                grad[n + i] += beta * (-sig)
                grad[2*n + i] += beta * sig
            
            # Top
            viol = cy[i] + r[i] - h
            if viol > -5*eps:
                sp = softplus(viol / eps, eps)
                obj += beta * sp
                sig = 1.0 / (1.0 + np.exp(-viol/eps))
                grad[n + i] += beta * sig
                grad[2*n + i] += beta * sig

        # Overlap penalty
        # Constraint: dist_ij >= r_i + r_j => violation = r_i + r_j - dist_ij
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                viol = r[i] + r[j] - dist
                
                if viol > -5*eps:
                    sp = softplus(viol / eps, eps)
                    obj += beta * sp
                    sig = 1.0 / (1.0 + np.exp(-viol/eps))
                    
                    # Gradients of viol = r_i + r_j - dist
                    # d(viol)/dx_i = -dx/dist
                    # d(viol)/dy_i = -dy/dist
                    # d(viol)/dr_i = 1
                    # d(viol)/dx_j = +dx/dist
                    # d(viol)/dy_j = +dy/dist
                    # d(viol)/dr_j = 1
                    
                    # Gradient of penalty = beta * sig * grad(viol)
                    grad[i] += beta * sig * (-dx / dist)
                    grad[n + i] += beta * sig * (-dy / dist)
                    grad[2*n + i] += beta * sig * 1.0
                    
                    grad[j] += beta * sig * (dx / dist)
                    grad[n + j] += beta * sig * (dy / dist)
                    grad[2*n + j] += beta * sig * 1.0

        return obj, grad

    current_x = x0.copy()
    
    # Schedule for beta and eps
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    epsilons = np.geomspace(eps_start, eps_end, n_beta_steps)

    for step in range(n_beta_steps):
        if time.time() > deadline - 5.0:
            break
        
        beta = betas[step]
        eps = epsilons[step]
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta, eps)[0],
                current_x,
                jac=lambda x: objective_and_grad(x, beta, eps)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                new_x = res.x
                # Check validity
                new_circles = np.array([(new_x[i], new_x[n+i], new_x[2*n+i]) for i in range(n)])
                if _validate(new_circles, w, h, tol=1e-5):
                    current_x = new_x
                else:
                    # If invalid, keep previous valid state
                    pass
            else:
                # If optimization fails, keep current
                pass
                
        except Exception:
            pass

    # Final relaxation and validation
    final_circles = np.array([(current_x[i], current_x[n+i], current_x[2*n+i]) for i in range(n)])
    final_circles = _relax_circles(final_circles, w, h, iterations=200)
    
    if _validate(final_circles, w, h, tol=1e-6):
        return final_circles
    
    return None

def _polish_circles(circles, w, h, deadline):
    """
    High-precision polish pass using sequential pairwise optimization.
    For each circle, try to increase its radius by optimizing position and radius
    while keeping other circles fixed.
    """
    if time.time() > deadline - 3.0:
        return circles

    n = len(circles)
    result = circles.copy()
    improved = True
    max_rounds = 2

    for _round in range(max_rounds):
        if not improved:
            break
        if time.time() > deadline - 3.0:
            break
        improved = False

        for i in range(n):
            if time.time() > deadline - 3.0:
                break

            cx, cy, r = result[i]
            
            def objective(x):
                return -x[2]

            def jac(x):
                return np.array([0.0, 0.0, -1.0])

            constraints = []

            # Boundary constraints
            constraints.append({'type': 'ineq', 'fun': lambda x: x[0] - x[2]})
            constraints.append({'type': 'ineq', 'fun': lambda x: w - x[0] - x[2]})
            constraints.append({'type': 'ineq', 'fun': lambda x: x[1] - x[2]})
            constraints.append({'type': 'ineq', 'fun': lambda x: h - x[1] - x[2]})

            # Non-overlap with all other circles
            for j in range(n):
                if j == i:
                    continue
                ox, oy, or_ = result[j]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, ox=ox, oy=oy, or_=or_: 
                        (x[0] - ox)**2 + (x[1] - oy)**2 - (x[2] + or_)**2
                })

            # Minimum radius
            constraints.append({'type': 'ineq', 'fun': lambda x: x[2] - 1e-8})

            bounds = [
                (1e-6, w - 1e-6),
                (1e-6, h - 1e-6),
                (1e-8, min(w, h)/2)
            ]

            x0 = np.array([cx, cy, r])

            try:
                res = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    jac=jac,
                    options={'maxiter': 30, 'ftol': 1e-12}
                )

                if res.success and res.x[2] > r + 1e-10:
                    new_cx, new_cy, new_r = res.x
                    new_result = result.copy()
                    new_result[i] = [new_cx, new_cy, new_r]

                    if _validate(new_result, w, h, tol=1e-7):
                        result = new_result
                        improved = True

            except Exception:
                pass

    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try a range of aspect ratios
    # Focus on aspect ratios that allow hexagonal packing to fit well
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 1.3, 0.75, 1.5, 0.65]
    
    for ar in ratios:
        if time.time() > deadline - 15.0:
            break
        
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Also try swapped orientation
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 15.0:
                break
            
            try:
                # Initialize with hexagonal grid
                circles = _make_hexagonal_grid(N, ww, hh)
                
                # Relax to ensure validity
                circles = _relax_circles(circles, ww, hh, iterations=500)
                
                if not _validate(circles, ww, hh):
                    continue
                
                current_value = _total_radius(circles)
                
                # Try penalty optimization
                if time.time() < deadline - 20.0:
                    optimized = _penalty_optimize(circles, ww, hh, deadline)
                    if optimized is not None:
                        circles = optimized
                        current_value = _total_radius(circles)

                # High-precision polish pass
                if time.time() < deadline - 5.0:
                    circles = _polish_circles(circles, ww, hh, deadline)
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

