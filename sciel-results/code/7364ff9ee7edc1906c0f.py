import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_circles = None
    best_w, best_h = 1.0, 1.0

    # Candidate aspect ratios (w, h) such that w + h = 2
    aspect_ratios = [
        (1.0, 1.0),
        (0.95, 1.05),
        (1.05, 0.95),
        (0.9, 1.1),
        (1.1, 0.9),
        (0.85, 1.15),
        (1.15, 0.85),
        (0.8, 1.2),
        (1.2, 0.8),
    ]

    def _validate(circles, w, h, tol=1e-6):
        """Check if circles are valid: inside rectangle and non-overlapping."""
        for cx, cy, r in circles:
            if r <= 0:
                return False
            if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
                return False
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < circles[i][2] + circles[j][2] - tol:
                    return False
        return True

    def _total_radius(circles):
        return sum(c[2] for c in circles)

    def _make_staggered_grid(w, h, n_circles):
        """Create a staggered grid of circles."""
        # Determine grid dimensions
        # Approximate aspect ratio of grid to match rectangle
        ar = w / h
        rows = int(np.sqrt(n_circles / ar))
        cols = int(np.sqrt(n_circles * ar))
        
        # Adjust to get exactly n_circles
        # Try combinations
        best_rows, best_cols = 0, 0
        min_diff = float('inf')
        for r in range(1, 10):
            for c in range(1, 10):
                if r * c >= n_circles:
                    diff = abs(r * c - n_circles)
                    if diff < min_diff:
                        min_diff = diff
                        best_rows, best_cols = r, c
        
        rows, cols = best_rows, best_cols
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n_circles:
                    break
                # Staggered offset
                offset = 0.5 if i % 2 == 0 else 0.0
                cx = (j + offset + 0.5) / (cols + 0.5) * w
                cy = (i + 0.5) / rows * h
                # Initial radius: small enough to not overlap
                # Rough estimate of spacing
                dx = w / (cols + 0.5)
                dy = h / rows
                r = 0.4 * min(dx, dy)
                circles.append((cx, cy, r))
        
        # If we have fewer circles than needed, add them randomly or extend grid
        # This simple grid might not be perfect, but relaxation will fix it
        return circles

    def _penalty_objective(x, w, h, lam):
        """
        Objective function with penalty terms.
        x: array of shape (3*N) containing [x1, y1, r1, x2, y2, r2, ...]
        """
        n = N
        cx = x[0::3]
        cy = x[1::3]
        r = x[2::3]
        
        obj = -np.sum(r)
        
        # Boundary penalties
        # x - r >= 0, x + r <= w, y - r >= 0, y + r <= h
        pen_x_min = np.maximum(0, r - cx)
        pen_x_max = np.maximum(0, cx + r - w)
        pen_y_min = np.maximum(0, r - cy)
        pen_y_max = np.maximum(0, cy + r - h)
        
        boundary_pen = np.sum(pen_x_min**2 + pen_x_max**2 + pen_y_min**2 + pen_y_max**2)
        
        # Overlap penalties
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2
        
        return obj + lam * (boundary_pen + overlap_pen)

    def _polish_slsqp(circles, w, h, deadline):
        """Final polish with SLSQP to ensure strict validity and maximize radius."""
        if time.time() > deadline - 5.0:
            return None
        
        n = len(circles)
        x0 = np.array([v for c in circles for v in c])
        
        def objective(x):
            r = x[2::3]
            return -np.sum(r)
        
        constraints = []
        # Boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})
            
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
                })
        
        bounds = []
        for i in range(n):
            bounds.append((0, w))
            bounds.append((0, h))
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
                new_circles = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)]
                if _validate(new_circles, w, h, tol=1e-5):
                    return new_circles
        except Exception:
            pass
        return None

    for w, h in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize
        circles = _make_staggered_grid(w, h, N)
        
        # Relaxation to ensure initial validity
        # Simple relaxation: push apart overlapping circles and clamp to bounds
        for _ in range(50):
            valid = True
            # Boundary clamp
            for i in range(N):
                cx, cy, r = circles[i]
                cx = np.clip(cx, r, w - r)
                cy = np.clip(cy, r, h - r)
                circles[i] = (cx, cy, r)
            
            # Overlap resolution
            for i in range(N):
                for j in range(i + 1, N):
                    dx = circles[j][0] - circles[i][0]
                    dy = circles[j][1] - circles[i][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i][2] + circles[j][2]
                    if dist < min_dist:
                        if dist < 1e-9:
                            angle = np.random.uniform(0, 2*np.pi)
                            push = (min_dist - dist) / 2.0
                            circles[i] = (circles[i][0] - np.cos(angle)*push, circles[i][1] - np.sin(angle)*push, circles[i][2])
                            circles[j] = (circles[j][0] + np.cos(angle)*push, circles[j][1] + np.sin(angle)*push, circles[j][2])
                        else:
                            ux = dx / dist
                            uy = dy / dist
                            push = (min_dist - dist) / 2.0
                            circles[i] = (circles[i][0] - ux*push, circles[i][1] - uy*push, circles[i][2])
                            circles[j] = (circles[j][0] + ux*push, circles[j][1] + uy*push, circles[j][2])
                        valid = False
            if valid:
                break
        
        if not _validate(circles, w, h):
            continue
            
        current_score = _total_radius(circles)
        if current_score > best_score:
            best_score = current_score
            best_circles = circles
            best_w, best_h = w, h

        # Optimize with L-BFGS-B and penalty
        x0 = np.array([v for c in circles for v in c])
        
        # Increase penalty weight gradually
        lam = 100.0
        for _ in range(3):
            if time.time() > deadline - 20.0:
                break
                
            try:
                res = minimize(
                    _penalty_objective,
                    x0,
                    args=(w, h, lam),
                    method='L-BFGS-B',
                    bounds=[(0, w), (0, h), (1e-6, min(w, h)/2)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                x0 = res.x
                lam *= 5.0 # Increase penalty
            except Exception:
                break
        
        # Extract circles from optimized x0
        new_circles = [(x0[i*3], x0[i*3+1], x0[i*3+2]) for i in range(N)]
        
        # Validate and relax if needed
        if _validate(new_circles, w, h, tol=1e-5):
            score = _total_radius(new_circles)
            if score > best_score:
                best_score = score
                best_circles = new_circles
                best_w, best_h = w, h
        else:
            # Try to fix with relaxation
            fixed = True
            for _ in range(100):
                # Boundary clamp
                for i in range(N):
                    cx, cy, r = new_circles[i]
                    cx = np.clip(cx, r, w - r)
                    cy = np.clip(cy, r, h - r)
                    new_circles[i] = (cx, cy, r)
                
                # Overlap resolution
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = new_circles[j][0] - new_circles[i][0]
                        dy = new_circles[j][1] - new_circles[i][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = new_circles[i][2] + new_circles[j][2]
                        if dist < min_dist:
                            if dist < 1e-9:
                                angle = np.random.uniform(0, 2*np.pi)
                                push = (min_dist - dist) / 2.0
                                new_circles[i] = (new_circles[i][0] - np.cos(angle)*push, new_circles[i][1] - np.sin(angle)*push, new_circles[i][2])
                                new_circles[j] = (new_circles[j][0] + np.cos(angle)*push, new_circles[j][1] + np.sin(angle)*push, new_circles[j][2])
                            else:
                                ux = dx / dist
                                uy = dy / dist
                                push = (min_dist - dist) / 2.0
                                new_circles[i] = (new_circles[i][0] - ux*push, new_circles[i][1] - uy*push, new_circles[i][2])
                                new_circles[j] = (new_circles[j][0] + ux*push, new_circles[j][1] + uy*push, new_circles[j][2])
                            fixed = False
                if fixed:
                    break
            
            if _validate(new_circles, w, h, tol=1e-5):
                score = _total_radius(new_circles)
                if score > best_score:
                    best_score = score
                    best_circles = new_circles
                    best_w, best_h = w, h

        # Final Polish with SLSQP if time allows
        if time.time() < deadline - 15.0 and best_circles is not None and (best_w, best_h) == (w, h):
            polished = _polish_slsqp(best_circles, w, h, deadline)
            if polished is not None:
                p_score = _total_radius(polished)
                if p_score > best_score:
                    best_score = p_score
                    best_circles = polished
                    best_w, best_h = w, h

    if best_circles is None:
        # Fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.05) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.array(best_circles)

if __name__ == "__main__":
    res = construct()
    print(f"Score: {_total_radius(res)}")
