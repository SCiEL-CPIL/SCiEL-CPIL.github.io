"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _get_contact_pairs():
    """
    Return list of (i, j) pairs that should be touching in the hexagonal arrangement.
    Rows: 5, 4, 5, 4, 3
    Indices are assigned row by row.
    """
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

def _relax_circles(circles, w, h, iterations=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Returns relaxed circles if valid, else original.
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
            # Try to center the rows
            if count == 5:
                x_start = 0.5 * (w / 5)
                x_step = w / 5
            elif count == 4:
                x_start = 0.75 * (w / 5)
                x_step = w / 5
            elif count == 3:
                x_start = 1.0 * (w / 5)
                x_step = w / 5
            else:
                x_start = 0.5
                x_step = 1.0
                
            x = x_start + col_j * x_step
            r = 0.45 * min(w/5, h/total_rows) # Initial radius estimate
            circles.append([x, y, r])
    
    return np.array(circles[:n])

def _optimize_aspect(aspect, deadline):
    """
    Optimize packing for a given aspect ratio using penalty method.
    Returns (circles, score) or (None, 0.0) if failed.
    """
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    # Initial guess
    circles = _make_hexagonal_grid(N, w, h)
    
    # Relax to get a valid starting point
    circles = _relax_circles(circles, w, h, iterations=1000)
    
    if not _validate(circles, w, h):
        # Try a simpler grid if hex fails
        side = 5
        circles = []
        for i in range(5):
            for j in range(5):
                if i == 4 and j >= 3:
                    continue # Only 21 circles
                cx = (j + 0.5) * w / 5
                cy = (i + 0.5) * h / 5
                r = 0.4 * min(w/5, h/5)
                circles.append([cx, cy, r])
        circles = np.array(circles[:N])
        circles = _relax_circles(circles, w, h, iterations=1000)
        if not _validate(circles, w, h):
            return None, 0.0

    # Optimization using penalty method
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r

    def objective_and_grad(x, beta):
        cx = x[:N]
        cy = x[N:2*N]
        r = x[2*N:]
        
        # Objective: -sum(r) + penalty
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*N:] = -1.0
        
        # Boundary penalty
        for i in range(N):
            if cx[i] - r[i] < 0:
                pen = beta * (cx[i] - r[i])**2
                obj += pen
                grad[i] += 2 * beta * (cx[i] - r[i])
                grad[2*N+i] += -2 * beta * (cx[i] - r[i])
            if cx[i] + r[i] > w:
                pen = beta * (cx[i] + r[i] - w)**2
                obj += pen
                grad[i] += 2 * beta * (cx[i] + r[i] - w)
                grad[2*N+i] += 2 * beta * (cx[i] + r[i] - w)
            if cy[i] - r[i] < 0:
                pen = beta * (cy[i] - r[i])**2
                obj += pen
                grad[N+i] += 2 * beta * (cy[i] - r[i])
                grad[2*N+i] += -2 * beta * (cy[i] - r[i])
            if cy[i] + r[i] > h:
                pen = beta * (cy[i] + r[i] - h)**2
                obj += pen
                grad[N+i] += 2 * beta * (cy[i] + r[i] - h)
                grad[2*N+i] += 2 * beta * (cy[i] + r[i] - h)
        
        # Overlap penalty
        for i in range(N):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[i+1:]
            overlap = np.maximum(0, min_dist - dist)
            
            if np.any(overlap > 0):
                pen = beta * np.sum(overlap**2)
                obj += pen
                
                # Gradients for overlaps are complex, approximate or use finite diff if needed
                # For speed, we might skip gradient for overlaps and rely on numerical grad in minimize?
                # No, let's compute it properly.
                # d(overlap)/d(cx_i) = (cx_i - cx_j) / dist * 2 * beta * overlap
                # d(overlap)/d(cx_j) = (cx_j - cx_i) / dist * 2 * beta * overlap
                # d(overlap)/d(r_i) = 2 * beta * overlap
                # d(overlap)/d(r_j) = 2 * beta * overlap
                
                for k in range(len(overlap)):
                    if overlap[k] > 0:
                        j = i + 1 + k
                        if dist[k] < 1e-9:
                            continue
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        factor = 2 * beta * overlap[k]
                        
                        grad[i] += factor * ux
                        grad[N+i] += factor * uy
                        grad[2*N+i] += factor
                        
                        grad[j] -= factor * ux
                        grad[N+j] -= factor * uy
                        grad[2*N+j] += factor

        return obj, grad

    best_circles = circles
    best_score = np.sum(circles[:, 2])
    
    # Annealing beta
    beta = 10.0
    for step in range(20):
        if time.time() > deadline - 10.0:
            break
            
        res = minimize(objective_and_grad, x0, args=(beta,), jac=True, bounds=bounds, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 50})
        
        if res.success or res.fun < objective_and_grad(x0, beta)[0]:
            x_opt = res.x
            new_circles = np.column_stack([x_opt[:N], x_opt[N:2*N], x_opt[2*N:]])
            
            # Relax to ensure validity
            new_circles = _relax_circles(new_circles, w, h, iterations=200)
            
            if _validate(new_circles, w, h):
                score = np.sum(new_circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = new_circles
                    x0 = np.concatenate([best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]])
                else:
                    x0 = np.concatenate([new_circles[:, 0], new_circles[:, 1], new_circles[:, 2]])
        
        beta *= 2.0
        
    return best_circles, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_circles = None
    
    # Try a range of aspect ratios
    aspects = [1.0, 0.8, 1.25, 0.6, 1.6, 0.5, 2.0, 1.5, 0.7]
    
    for aspect in aspects:
        if time.time() > deadline - 15.0:
            break
            
        circles, score = _optimize_aspect(aspect, deadline)
        
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles
            
    # Fallback if nothing worked
    if best_circles is None:
        w, h = 1.0, 1.0
        circles = _make_hexagonal_grid(N, w, h)
        circles = _relax_circles(circles, w, h, iterations=2000)
        if _validate(circles, w, h):
            best_circles = circles
            best_score = np.sum(circles[:, 2])
        else:
            # Tiny circles fallback
            w, h = 1.0, 1.0
            circles = np.zeros((N, 3))
            for i in range(N):
                row = i // 5
                col = i % 5
                circles[i, 0] = (col + 0.5) * w / 5
                circles[i, 1] = (row + 0.5) * h / 5
                circles[i, 2] = 0.01
            best_circles = circles
            best_score = 0.21

    return best_circles

# EVOLVE-BLOCK-END
