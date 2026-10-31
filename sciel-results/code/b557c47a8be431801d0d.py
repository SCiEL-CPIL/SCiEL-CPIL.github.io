"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 115.0  # safety margin before 120s kill

def construct():
    t_start = time.time()
    
    def elapsed():
        return time.time() - t_start
    
    def is_valid(circles, tol=1e-7):
        """Check if circles are valid (disjoint, inside unit square)."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol or r < -tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i][2] + circles[j][2] - tol:
                    return False
        return True
    
    def fix_validity(circles, tol=1e-9):
        """Fix small validity violations by shrinking radii and adjusting positions."""
        circles = np.array(circles, dtype=float).copy()
        n = len(circles)
        for _ in range(50):
            # Fix boundary violations
            for i in range(n):
                x, y, r = circles[i]
                if r < 1e-12:
                    circles[i] = [0.5, 0.5, 1e-12]
                    continue
                # Clamp position to keep circle inside
                min_r = max(r - x, r - (1-x), r - y, r - (1-y), 0)
                if min_r > 0:
                    # Move toward center
                    cx = np.clip(x, r, 1-r)
                    cy = np.clip(y, r, 1-r)
                    circles[i][0] = cx
                    circles[i][1] = cy
                    # If still violating, shrink
                    if cx - r < -tol or cx + r > 1+tol or cy - r < -tol or cy + r > 1+tol:
                        circles[i][2] = min(r - abs(cx - x) - 1e-9, r - abs(cy - y) - 1e-9, r - 1e-9)
                        circles[i][2] = max(circles[i][2], 1e-12)
            
            # Fix overlap violations
            for i in range(n):
                for j in range(i+1, n):
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i][2] + circles[j][2]
                    if dist < min_dist - tol:
                        overlap = min_dist - dist
                        # Shrink the smaller circle
                        if circles[i][2] < circles[j][2]:
                            circles[i][2] = max(circles[i][2] - overlap/2 - 1e-9, 1e-12)
                        else:
                            circles[j][2] = max(circles[j][2] - overlap/2 - 1e-9, 1e-12)
        
        # Final check and fix
        for i in range(n):
            x, y, r = circles[i]
            # Ensure inside
            r = min(r, x, 1-x, y, 1-y)
            circles[i][2] = max(r, 1e-12)
            circles[i][0] = np.clip(x, circles[i][2], 1-circles[i][2])
            circles[i][1] = np.clip(y, circles[i][2], 1-circles[i][2])
        
        # One more pass for overlaps
        for _ in range(20):
            for i in range(n):
                for j in range(i+1, n):
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i][2] + circles[j][2]
                    if dist < min_dist - tol:
                        overlap = min_dist - dist
                        if circles[i][2] < circles[j][2]:
                            circles[i][2] = max(circles[i][2] - overlap - 1e-9, 1e-12)
                        else:
                            circles[j][2] = max(circles[j][2] - overlap - 1e-9, 1e-12)
        
        return circles
    
    def objective_and_grad(params):
        """Penalized objective: minimize -sum(r) + penalty for violations."""
        n = len(params) // 3
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:]
        
        # Negative sum of radii
        obj = -np.sum(r)
        
        # Boundary penalty
        for i in range(n):
            if r[i] > 0:
                b_pen = 0
                if x[i] - r[i] < 0: b_pen += (x[i] - r[i])**2
                if x[i] + r[i] > 1: b_pen += (x[i] + r[i] - 1)**2
                if y[i] - r[i] < 0: b_pen += (y[i] - r[i])**2
                if y[i] + r[i] > 1: b_pen += (y[i] + r[i] - 1)**2
                obj += 1000 * b_pen
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    obj += 1000 * (min_dist - dist)**2
        
        # Gradient
        grad = np.zeros_like(params)
        for i in range(n):
            grad[2*n+i] = -1.0
            # Boundary gradient
            if r[i] > 0:
                if x[i] - r[i] < 0:
                    grad[i] += 2000*(x[i] - r[i])
                    grad[2*n+i] += -2000*(x[i] - r[i])
                if x[i] + r[i] > 1:
                    grad[i] += 2000*(x[i] + r[i] - 1)
                    grad[2*n+i] += 2000*(x[i] + r[i] - 1)
                if y[i] - r[i] < 0:
                    grad[n+i] += 2000*(y[i] - r[i])
                    grad[2*n+i] += -2000*(y[i] - r[i])
                if y[i] + r[i] > 1:
                    grad[n+i] += 2000*(y[i] + r[i] - 1)
                    grad[2*n+i] += 2000*(y[i] + r[i] - 1)
        
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    diff = min_dist - dist
                    grad[i] += 2000 * diff * (dx/dist)
                    grad[n+i] += 2000 * diff * (dy/dist)
                    grad[2*n+i] += 2000 * diff
                    grad[j] -= 2000 * diff * (dx/dist)
                    grad[n+j] -= 2000 * diff * (dy/dist)
                    grad[2*n+j] += 2000 * diff
        
        return obj, grad
    
    def optimize_from_start(start_circles, beta=100, max_iter=200):
        """Run SLSQP optimization from a starting point."""
        n = len(start_circles)
        x0 = np.array([start_circles[i][0] for i in range(n)])
        y0 = np.array([start_circles[i][1] for i in range(n)])
        r0 = np.array([start_circles[i][2] for i in range(n)])
        
        # Ensure r0 is positive
        r0 = np.maximum(r0, 1e-6)
        
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds: x,y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x
        for i in range(n):
            bounds.append((0.0, 1.0))  # y
        for i in range(n):
            bounds.append((1e-6, 0.5))  # r
        
        result = minimize(
            objective_and_grad,
            params0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        params = result.x
        circles = np.array([
            [params[i], params[n+i], params[2*n+i]] for i in range(n)
        ])
        return circles
    
    def coordinate_descent(circles, beta=500, max_iter=100):
        """Coordinate descent with binary search on radii."""
        n = len(circles)
        circles = np.array(circles, dtype=float).copy()
        
        for it in range(max_iter):
            if elapsed() > DEADLINE:
                break
            improved = False
            for i in range(n):
                # Try to increase radius of circle i
                lo = circles[i][2]
                hi = min(0.5, circles[i][0], 1-circles[i][0], circles[i][1], 1-circles[i][1])
                hi = min(hi, 0.5)
                
                if hi <= lo + 1e-12:
                    continue
                
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if mid <= lo:
                        break
                    # Check if we can set r_i = mid
                    valid = True
                    for j in range(n):
                        if j == i:
                            continue
                        dx = circles[i][0] - circles[j][0]
                        dy = circles[i][1] - circles[j][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < mid + circles[j][2] - 1e-12:
                            valid = False
                            break
                    if valid and mid <= circles[i][0] and mid <= 1-circles[i][0] and mid <= circles[i][1] and mid <= 1-circles[i][1]:
                        lo = mid
                    else:
                        hi = mid
                
                if lo > circles[i][2] + 1e-12:
                    circles[i][2] = lo
                    improved = True
            
            if not improved:
                break
        
        return circles
    
    def generate_grid_start():
        """Generate a grid-based starting configuration."""
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    def generate_random_start():
        """Generate a random starting configuration."""
        circles = []
        for _ in range(N):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.08)
            circles.append([x, y, r])
        return np.array(circles)
    
    def generate_packed_start():
        """Generate a better packed starting configuration using a simple algorithm."""
        circles = []
        # Place circles greedily
        np.random.seed(42)
        attempts = 0
        max_attempts = 5000
        while len(circles) < N and attempts < max_attempts:
            attempts += 1
            if elapsed() > DEADLINE:
                break
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            # Max radius based on boundaries
            r_max = min(x, 1-x, y, 1-y)
            # Check against existing circles
            for cx, cy, cr in circles:
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                r_max = min(r_max, dist - cr)
            if r_max > 0.02:
                r = r_max * 0.95
                circles.append([x, y, r])
        
        # If we don't have enough, add small circles
        while len(circles) < N:
            circles.append([0.5, 0.5, 0.01])
        
        return np.array(circles[:N])
    
    # Main optimization loop
    best_circles = None
    best_score = -np.inf
    
    # Try multiple starting configurations
    starts = [
        generate_grid_start(),
        generate_packed_start(),
    ]
    
    # Add a few random starts
    np.random.seed(0)
    for _ in range(3):
        if elapsed() > DEADLINE - 30:
            break
        starts.append(generate_random_start())
    
    for start in starts:
        if elapsed() > DEADLINE - 20:
            break
        
        # First optimization pass
        circles = optimize_from_start(start, max_iter=100)
        circles = fix_validity(circles)
        
        # Coordinate descent to increase radii
        circles = coordinate_descent(circles, max_iter=50)
        circles = fix_validity(circles)
        
        # Second optimization pass with the improved starting point
        if elapsed() < DEADLINE - 10:
            circles2 = optimize_from_start(circles, max_iter=150)
            circles2 = fix_validity(circles2)
            circles2 = coordinate_descent(circles2, max_iter=80)
            circles2 = fix_validity(circles2)
            
            score2 = np.sum(circles2[:, 2])
            if score2 > best_score and is_valid(circles2):
                best_score = score2
                best_circles = circles2
            else:
                score1 = np.sum(circles[:, 2])
                if score1 > best_score and is_valid(circles):
                    best_score = score1
                    best_circles = circles
        else:
            score1 = np.sum(circles[:, 2])
            if score1 > best_score and is_valid(circles):
                best_score = score1
                best_circles = circles
    
    # If we didn't find a valid solution, use the grid
    if best_circles is None or not is_valid(best_circles):
        best_circles = generate_grid_start()
        best_circles = fix_validity(best_circles)
    
    # Final coordinate descent squeeze
    if elapsed() < DEADLINE - 5:
        best_circles = coordinate_descent(best_circles, max_iter=60)
        best_circles = fix_validity(best_circles)
    
    return best_circles
