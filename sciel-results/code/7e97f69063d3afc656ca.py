import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(circles):
    """Check if circles are valid: within a perimeter-4 rectangle, non-overlapping."""
    if circles.shape[0] != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check radii positive
    if np.any(rs <= 0):
        return False
    
    # Find bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    # Perimeter constraint
    if 2 * (w + h) > 4.0 + 1e-9:
        return False
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    
    return True

def _total_radius(circles):
    return np.sum(circles[:, 2])

def _separate_circles(circles, iterations=50):
    """Iteratively separate overlapping circles."""
    circles = circles.copy()
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist - 1e-10:
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    overlap = min_dist - dist
                    # Push apart
                    ux, uy = dx / dist, dy / dist
                    push = overlap / 2 + 1e-6
                    circles[i, 0] += ux * push
                    circles[i, 1] += uy * push
                    circles[j, 0] -= ux * push
                    circles[j, 1] -= uy * push
                    moved = True
        if not moved:
            break
    return circles

def _smooth_overlap_penalty(circles, w=100.0):
    """Smooth penalty for overlaps."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = circles[i, 2] + circles[j, 2]
            if dist < min_dist:
                penalty += w * (min_dist - dist) ** 2
    return penalty

def _objective_and_grad(x, w_overlap=200.0, w_boundary=200.0):
    """
    x: [x1...x21, y1...y21, r]
    Returns (value, grad) where value is negative sum of radii + penalties.
    """
    xs = x[:N]
    ys = x[N:2*N]
    r = x[2*N]
    
    val = -r * N  # maximize sum of radii (all same r)
    grad = np.zeros_like(x)
    grad[2*N] = -N
    
    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = 2 * r
            if dist < min_dist and dist > 1e-12:
                diff = min_dist - dist
                penalty = w_overlap * diff ** 2
                val += penalty
                
                # Gradient
                factor = 2 * w_overlap * diff / dist
                grad[i] += factor * dx
                grad[N + i] += factor * dy
                grad[j] -= factor * dx
                grad[N + j] -= factor * dy
                grad[2*N] += 2 * w_overlap * diff * 2  # d(min_dist)/dr = 2, d(diff)/dr = -2, so d(penalty)/dr = -2*w*diff*(-2) = 4*w*diff
    
    # Boundary penalties: keep circles inside a unit square (we'll scale later)
    # Actually, let's work in the rectangle directly.
    # We'll normalize to perimeter 4 at the end.
    
    return val, grad

def _local_optimize(x0, max_iter=100, tol=1e-8, deadline=None):
    """Local optimization using gradient descent with line search."""
    x = x0.copy()
    best_x = x.copy()
    best_val = _objective_and_grad(x)[0]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        val, grad = _objective_and_grad(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        if np.max(np.abs(grad)) < tol:
            break
        
        # Gradient descent with adaptive step size
        step = 0.01
        for _ in range(20):
            x_new = x - step * grad
            val_new, _ = _objective_and_grad(x_new)
            if val_new < val:
                x = x_new
                break
            step *= 0.5
        
        if step < 1e-10:
            break
    
    return best_x, best_val

def _grid_initialization():
    """Create a grid-based initial configuration."""
    # Try a 5x5 grid minus 4 corners = 21 circles
    positions = []
    for i in range(5):
        for j in range(5):
            if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                continue
            positions.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    
    positions = np.array(positions)
    r = 0.1
    x = np.concatenate([positions[:, 0], positions[:, 1], [r]])
    return x

def _random_initialization(rng):
    """Create a random initial configuration."""
    xs = rng.uniform(0.1, 0.9, size=N)
    ys = rng.uniform(0.1, 0.9, size=N)
    r = 0.08
    x = np.concatenate([xs, ys, [r]])
    return x

def _successive_convex_relaxation(x0, max_sweeps=20, deadline=None):
    """
    Successive convex relaxation: linearize non-overlap constraints iteratively.
    At each iteration, solve a QP-like problem with linearized constraints.
    """
    x = x0.copy()
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Linearize overlap constraints at current x
        # For each pair (i,j), the constraint is:
        # (xi-xj)^2 + (yi-yj)^2 >= (2r)^2
        # Linearize around current position:
        # 2*(xi-xj)*(xi-xj_cur) + 2*(yi-yj)*(yi-yj_cur) >= (2r)^2 - ((xi-xj_cur)^2 + (yi-yj_cur)^2)
        
        cons = []
        bounds = [(0.01, 0.99) for _ in range(2*N)] + [(0.01, 0.5)]
        
        # Boundary constraints (keep in unit square for now)
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*N]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[N+i] - x[2*N]})
        
        # Linearized overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx_cur = x[i] - x[j]
                dy_cur = x[N+i] - x[N+j]
                rhs = (2 * x[2*N]) ** 2 - (dx_cur ** 2 + dy_cur ** 2)
                
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j, dx_cur=dx_cur, dy_cur=dy_cur, rhs=rhs: \
                        2*dx_cur*(x[i]-x[j]) + 2*dy_cur*(x[N+i]-x[N+j]) - rhs
                })
        
        try:
            res = minimize(
                lambda x: -x[2*N],
                x,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            if res.success:
                x = res.x
        except Exception:
            pass
    
    return x

def _polish(circles, iterations=100, deadline=None):
    """Polish the solution by optimizing positions and radius."""
    xs = circles[:, 0].copy()
    ys = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    for _ in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to increase r slightly and re-separate
        r_new = r[0] * 1.001
        if r_new > 0.25:
            break
        
        circles[:, 2] = r_new
        circles = _separate_circles(circles, iterations=30)
        
        # Check if still valid in terms of perimeter
        min_x = np.min(circles[:, 0] - circles[:, 2])
        max_x = np.max(circles[:, 0] + circles[:, 2])
        min_y = np.min(circles[:, 1] - circles[:, 2])
        max_y = np.max(circles[:, 1] + circles[:, 2])
        w = max_x - min_x
        h = max_y - min_y
        
        if 2 * (w + h) > 4.0:
            # Scale down to fit
            scale = 4.0 / (2 * (w + h))
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            circles[:, 0] = cx + (circles[:, 0] - cx) * scale
            circles[:, 1] = cy + (circles[:, 1] - cy) * scale
            circles[:, 2] *= scale
        
        # Check overlaps after scaling
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            circles[:, 2] = r  # revert
            break
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_circles = None
    best_score = 0
    
    # Generate multiple initial configurations
    inits = []
    
    # Grid initialization
    x_grid = _grid_initialization()
    circles_grid = np.column_stack([x_grid[:N], x_grid[N:2*N], np.full(N, x_grid[2*N])])
    inits.append(circles_grid)
    
    # Random initializations
    for _ in range(5):
        x_rand = _random_initialization(rng)
        circles_rand = np.column_stack([x_rand[:N], x_rand[N:2*N], np.full(N, x_rand[2*N])])
        inits.append(circles_rand)
    
    # Process each initial configuration
    for init in inits:
        if time.time() > deadline:
            break
        
        # Separate circles
        circles = _separate_circles(init, iterations=50)
        
        # Successive convex relaxation
        x_flat = np.concatenate([circles[:, 0], circles[:, 1], [circles[0, 2]]])
        x_scr = _successive_convex_relaxation(x_flat, max_sweeps=10, deadline=deadline)
        
        # Reconstruct circles
        circles = np.column_stack([x_scr[:N], x_scr[N:2*N], np.full(N, x_scr[2*N])])
        
        # Polish
        circles = _polish(circles, iterations=50, deadline=deadline)
        
        # Check validity and score
        if _validate(circles):
            score = _total_radius(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    # If no valid solution found, create a simple valid one
    if best_circles is None:
        # Simple grid that definitely fits
        side = 5
        cells = []
        for i in range(5):
            for j in range(5):
                if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                    continue
                x = (i + 0.5) / 5
                y = (j + 0.5) / 5
                r = 0.5 / 5 - 0.01
                cells.append([x, y, r])
        best_circles = np.array(cells)
    
    # Final normalization: scale to exactly perimeter 4
    min_x = np.min(best_circles[:, 0] - best_circles[:, 2])
    max_x = np.max(best_circles[:, 0] + best_circles[:, 2])
    min_y = np.min(best_circles[:, 1] - best_circles[:, 2])
    max_y = np.max(best_circles[:, 1] + best_circles[:, 2])
    w = max_x - min_x
    h = max_y - min_y
    perimeter = 2 * (w + h)
    
    if perimeter > 4.0:
        scale = 4.0 / perimeter
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        best_circles[:, 0] = cx + (best_circles[:, 0] - cx) * scale
        best_circles[:, 1] = cy + (best_circles[:, 1] - cy) * scale
        best_circles[:, 2] *= scale
    
    # Final validation
    if not _validate(best_circles):
        # Fallback
        side = 5
        cells = []
        for i in range(5):
            for j in range(5):
                if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                    continue
                x = (i + 0.5) / 5
                y = (j + 0.5) / 5
                r = 0.5 / 5 - 0.02
                cells.append([x, y, r])
        best_circles = np.array(cells)
    
    return best_circles
