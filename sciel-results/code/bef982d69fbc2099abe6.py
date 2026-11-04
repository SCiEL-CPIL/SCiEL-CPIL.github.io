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

def _relax_positions(pos, rad, w, h, iterations=2000, tol=1e-10):
    """
    Iteratively resolve overlaps by pushing circles apart.
    Returns True if no overlaps remain, False otherwise.
    """
    n = len(pos)
    pos = np.array(pos, dtype=float)
    rad = np.array(rad, dtype=float)
    
    for _ in range(iterations):
        moved = False
        # Boundary constraints
        for i in range(n):
            # Left
            if pos[i, 0] < rad[i]:
                pos[i, 0] = rad[i]
                moved = True
            # Right
            if pos[i, 0] > w - rad[i]:
                pos[i, 0] = w - rad[i]
                moved = True
            # Bottom
            if pos[i, 1] < rad[i]:
                pos[i, 1] = rad[i]
                moved = True
            # Top
            if pos[i, 1] > h - rad[i]:
                pos[i, 1] = h - rad[i]
                moved = True
        
        # Overlap constraints
        # We use a simple pair-wise push. For better convergence, we could use a more sophisticated method,
        # but this is often sufficient for the 'growth' phase.
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = rad[i] + rad[j]
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                    if dist < 1e-9:
                        # Random push if exactly overlapping
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    overlap = min_dist - dist
                    if overlap > 0:
                        # Push apart by half the overlap each
                        push = overlap * 0.5
                        pos[i, 0] -= dx * push
                        pos[i, 1] -= dy * push
                        pos[j, 0] += dx * push
                        pos[j, 1] += dy * push
                        moved = True
        
        if not moved:
            break
            
    # Final check for validity
    for i in range(n):
        if pos[i, 0] < rad[i] - 1e-9 or pos[i, 0] > w - rad[i] + 1e-9 or \
           pos[i, 1] < rad[i] - 1e-9 or pos[i, 1] > h - rad[i] + 1e-9:
            return False
            
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rad[i] + rad[j] - 1e-9:
                return False
                
    return True

def _make_hexagonal_init(w, h):
    """Create a hexagonal-like initial layout."""
    # Try to fit 21 circles. 
    # A common dense packing for 21 is rows of 5, 4, 5, 4, 3 or 4, 5, 4, 4, 4 etc.
    # Let's try a generic grid first, then relax.
    rows = [5, 4, 5, 4, 3]
    pos = []
    rad = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            # Offset alternating rows
            offset = 0.5 * (w / 5.0) if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * w / 5.0 + offset
            # Keep within bounds
            x = np.clip(x, 0.1, w - 0.1)
            pos.append([x, y])
            rad.append(0.05) # Start small
    
    # Pad if necessary (should be exactly 21)
    while len(pos) < N:
        pos.append([w/2, h/2])
        rad.append(0.05)
        
    pos = np.array(pos[:N])
    rad = np.array(rad[:N])
    return pos, rad

def _make_grid_init(w, h):
    """Create a grid initial layout."""
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    rad = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N: break
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            pos.append([x, y])
            rad.append(0.05)
    while len(pos) < N:
        pos.append([w/2, h/2])
        rad.append(0.05)
    return np.array(pos[:N]), np.array(rad[:N])

def _grow_circles(pos, rad, w, h, max_iter=1000, delta_start=0.01, delta_decay=0.9):
    """
    Grow circles iteratively while relaxing positions.
    """
    pos = np.array(pos, dtype=float)
    rad = np.array(rad, dtype=float)
    delta = delta_start
    stall_count = 0
    
    for _ in range(max_iter):
        # Try to grow
        new_rad = rad + delta
        
        # Check if growth is possible
        # We don't need to check analytically, we can just try to relax
        # But if delta is too large, relaxation might fail.
        
        if _relax_positions(pos, new_rad, w, h, iterations=1000):
            rad = new_rad
            stall_count = 0
            # Increase growth rate slightly if stable? Or keep constant.
            # Let's keep it constant or slightly increase to speed up.
            # delta *= 1.01 
        else:
            # Growth failed, try smaller delta
            delta *= delta_decay
            stall_count += 1
            if delta < 1e-5:
                break
            if stall_count > 50:
                break
                
    return pos, rad

def _polish(circles, w, h, deadline):
    """
    Use L-BFGS-B to maximize sum of radii with penalty for overlaps.
    """
    if time.time() > deadline - 5.0:
        return circles

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))

    def objective_and_grad(x, beta):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty
        for i in range(n):
            if cx[i] - r[i] < 0:
                val = cx[i] - r[i]
                obj += beta * val**2
                grad[i] += 2 * beta * val
                grad[2*n+i] -= 2 * beta * val
            if cx[i] + r[i] > w:
                val = cx[i] + r[i] - w
                obj += beta * val**2
                grad[i] += 2 * beta * val
                grad[2*n+i] += 2 * beta * val
            if cy[i] - r[i] < 0:
                val = cy[i] - r[i]
                obj += beta * val**2
                grad[n+i] += 2 * beta * val
                grad[2*n+i] -= 2 * beta * val
            if cy[i] + r[i] > h:
                val = cy[i] + r[i] - h
                obj += beta * val**2
                grad[n+i] += 2 * beta * val
                grad[2*n+i] += 2 * beta * val
        
        # Overlap penalty
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[i+1:]
            overlap = np.maximum(0, min_dist - dist)
            
            mask = overlap > 0
            if np.any(mask):
                obj += beta * np.sum(overlap[mask]**2)
                
                d_dist_dx = dx[mask] / dist[mask]
                d_dist_dy = dy[mask] / dist[mask]
                
                grad[i] += 2 * beta * np.sum(overlap[mask] * d_dist_dx)
                grad[n+i] += 2 * beta * np.sum(overlap[mask] * d_dist_dy)
                grad[2*n+i] += 2 * beta * np.sum(overlap[mask])
                
                idxs = np.arange(i+1, n)[mask]
                np.add.at(grad, idxs, 2 * beta * overlap[mask] * d_dist_dx)
                np.add.at(grad, n + idxs, 2 * beta * overlap[mask] * d_dist_dy)
                np.add.at(grad, 2*n + idxs, 2 * beta * overlap[mask])

        return obj, grad

    best_circles = circles
    best_val = np.inf
    x0 = x0.copy()
    
    # Anneal beta
    for beta in [10, 50, 200, 1000]:
        if time.time() > deadline - 2.0:
            break
        try:
            result = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x0,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                x0 = result.x
        except Exception:
            pass
            
    if x0 is not None:
        pos = x0[:2*n].reshape(n, 2)
        rad = x0[2*n:]
        new_circles = np.column_stack((pos, rad))
        if _validate(new_circles, w, h):
            return new_circles
    return best_circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_value = 0.0
    best_result = None

    # Try a range of aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h=2.
    # w in [0.5, 1.5]
    aspect_ratios = np.linspace(0.6, 1.4, 9)
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        w = ar
        h = 2.0 - w
        
        # Initialization
        pos, rad = _make_hexagonal_init(w, h)
        
        # Phase 1: Grow circles
        # We want to maximize sum of radii. 
        # The 'grow' algorithm increases all radii equally. 
        # This assumes equal radii, which is often optimal or near-optimal for dense packings.
        # If the optimal solution has unequal radii, the 'polish' phase can adjust them.
        # However, starting with equal radii is a good heuristic.
        
        pos, rad = _grow_circles(pos, rad, w, h, max_iter=500, delta_start=0.005)
        
        # Phase 2: Polish with L-BFGS
        circles = np.column_stack((pos, rad))
        
        # Ensure validity before polish
        if not _validate(circles, w, h):
             # Fallback to simple relaxation
             if _relax_positions(circles[:, :2], circles[:, 2], w, h, iterations=1000):
                 pass
             else:
                 continue

        polished = _polish(circles, w, h, deadline)
        
        if _validate(polished, w, h):
            val = _total_radius(polished)
            if val > best_value:
                best_value = val
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
