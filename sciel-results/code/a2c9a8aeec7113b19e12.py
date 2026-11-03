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
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _relax_circles(circles, w, h, iterations=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    This ensures the output is strictly valid.
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
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]

                if dist < min_dist:
                    if dist < 1e-9:
                        # Random push if coincident
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp to ensure validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    # Final validation check
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    # If relaxation failed (shouldn't happen with small radii), return original
    return circles

def _make_grid(n, w, h):
    # Try different grid layouts and pick the one with largest total radius
    best_cells = None
    best_total = 0.0
    
    # Layout 1: Square grid
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.85
            cells.append([cx, cy, r])
    if len(cells) == n:
        total = sum(c[2] for c in cells)
        if total > best_total:
            best_total = total
            best_cells = cells
    
    # Layout 2: Rows with varying column counts
    # Try to fit circles in rows
    for n_rows in range(3, 6):
        n_cols = int(np.ceil(n / n_rows))
        cells = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(cells) >= n:
                    break
                cx = (j + 0.5) * w / n_cols
                cy = (i + 0.5) * h / n_rows
                r = 0.5 * min(w / n_cols, h / n_rows) * 0.85
                cells.append([cx, cy, r])
        if len(cells) == n:
            total = sum(c[2] for c in cells)
            if total > best_total:
                best_total = total
                best_cells = cells
    
    # Layout 3: Hexagonal packing
    # Approximate hexagonal arrangement
    for offset in [0.0, 0.5]:
        cells = []
        row_height = h / 5.0
        for row in range(5):
            n_in_row = 5 if row % 2 == 0 else 4
            for col in range(n_in_row):
                if len(cells) >= n:
                    break
                cx = (col + 0.5 + offset) * w / (n_in_row + offset)
                cy = (row + 0.5) * row_height
                r = 0.5 * min(w / (n_in_row + offset), row_height) * 0.8
                cells.append([cx, cy, r])
        if len(cells) == n:
            total = sum(c[2] for c in cells)
            if total > best_total:
                best_total = total
                best_cells = cells
    
    if best_cells is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = 0.5 * min(w, h) / side * 0.8
                cells.append([cx, cy, r])
        best_cells = cells[:n]
    
    return np.array(best_cells)

def _polish_circles(circles, w, h, deadline):
    """
    High-precision polish pass using sequential pairwise optimization.
    For each circle, try to increase its radius by optimizing position and radius
    while keeping other circles fixed. This exploits the sparse structure of
    non-overlap constraints.
    """
    if time.time() > deadline - 3.0:
        return circles

    n = len(circles)
    result = circles.copy()
    improved = True
    max_rounds = 3

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
            # Try to increase radius of circle i by optimizing its position
            # Subject to: not overlapping with any other circle, staying in bounds

            # Collect constraints from all other circles
            def objective(x):
                # x = [cx, cy, r_new]
                return -x[2]  # maximize radius

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
                # (cx - ox)^2 + (cy - oy)^2 >= (r + or)^2
                # => (cx - ox)^2 + (cy - oy)^2 - (r + or)^2 >= 0
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
                    options={'maxiter': 50, 'ftol': 1e-12}
                )

                if res.success and res.x[2] > r + 1e-10:
                    # Check validity of the new solution
                    new_cx, new_cy, new_r = res.x
                    new_result = result.copy()
                    new_result[i] = [new_cx, new_cy, new_r]

                    if _validate(new_result, w, h, tol=1e-7):
                        result = new_result
                        improved = True

            except Exception:
                pass

    return result

def _slsqp_optimize(circles, w, h, deadline):
    """
    Optimize radii and positions using SLSQP with hard constraints.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.array([v for c in circles for v in c])

    # Objective: maximize sum of radii => minimize negative sum
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)

    # Constraints
    constraints = []

    # Boundary constraints
    for i in range(n):
        # x >= r  => x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # x + r <= w => w - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        # y >= r  => y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # y + r <= h => h - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        # r > 0 => r >= 1e-6
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    # Overlap constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9}
        )

        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
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

    # Try a range of aspect ratios
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 1.3, 0.75, 1.5, 0.65]
    
    for ar in ratios:
        if time.time() > deadline - 10.0:
            break
        
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Also try swapped orientation
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                # Initialize with grid
                circles = _make_grid(N, ww, hh)
                
                # Relax to ensure validity
                circles = _relax_circles(circles, ww, hh, iterations=500)
                
                if not _validate(circles, ww, hh):
                    continue
                
                current_value = _total_radius(circles)
                
                # Try trust-constr optimization first
                if time.time() < deadline - 20.0:
                    optimized = _trust_constr_optimize(circles, ww, hh, deadline)
                    if optimized is not None:
                        circles = optimized
                        current_value = _total_radius(circles)
                
                # If trust-constr failed or time is running out, try SLSQP
                if time.time() < deadline - 15.0:
                    optimized_slsqp = _slsqp_optimize(circles, ww, hh, deadline)
                    if optimized_slsqp is not None:
                        circles = optimized_slsqp
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
