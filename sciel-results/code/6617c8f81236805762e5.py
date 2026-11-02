import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    def check_time():
        return (time.time() - start_time) > DEADLINE
    
    def perimeter(circles):
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        return 2.0 * ((x_max - x_min) + (y_max - y_min))
    
    def overlap_penalty(circles, soft=50.0):
        """Soft penalty for overlaps using smooth approximation."""
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx * dx + dy * dy)
                violation = r[i] + r[j] - dist
                if violation > 0:
                    pen += np.log1p(np.exp(soft * violation)) / soft
        return pen
    
    def is_valid(circles, tol=1e-5):
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(r < -tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - tol:
                    return False
        if perimeter(circles) > 4.0 + tol:
            return False
        return True
    
    def force_directed(init, max_iter=2000):
        """Force-directed relaxation."""
        x = init[0].copy()
        y = init[1].copy()
        r = init[2].copy()
        
        dt = 0.02
        for it in range(max_iter):
            if check_time():
                break
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Repulsion between overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < 1e-12:
                        dx, dy = 0.01, 0.01
                        dist = np.sqrt(2) * 0.01
                    if dist < min_dist:
                        overlap = min_dist - dist
                        force = overlap * 5.0
                        fx[i] -= force * dx / dist
                        fy[i] -= force * dy / dist
                        fx[j] += force * dx / dist
                        fy[j] += force * dy / dist
            
            x += dt * fx
            y += dt * fy
            
            # Adjust radii based on perimeter
            p = perimeter(np.column_stack([x, y, r]))
            if p > 4.0:
                scale = 4.0 / p
                r *= scale * 0.999
            else:
                # Grow radii proportionally to available space
                slack = (4.0 - p) / 2.0 / N
                r += max(0.001, slack * 0.1)
            
            r = np.maximum(r, 1e-4)
        
        return np.column_stack([x, y, r])
    
    def polish(circles, max_iter=500):
        """Polish with L-BFGS-B using softplus penalties."""
        n = N
        x0 = circles.ravel()
        
        def objective(params):
            x, y, r = params[:n], params[n:2*n], params[2*n:3*n]
            # Overlap penalty (softplus)
            pen = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx * dx + dy * dy)
                    v = r[i] + r[j] - dist
                    pen += np.log1p(np.exp(100.0 * v)) / 100.0
            # Perimeter penalty
            x_min = np.min(x - r)
            x_max = np.max(x + r)
            y_min = np.min(y - r)
            y_max = np.max(y + r)
            p = 2.0 * ((x_max - x_min) + (y_max - y_min))
            pen += 50.0 * np.log1p(np.exp(50.0 * (p - 4.0))) / 50.0
            # Negative radius penalty
            pen += 100.0 * np.sum(np.log1p(np.exp(-100.0 * r))) / 100.0
            return -np.sum(r) + pen
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < objective(x0) * 0.999:
                return res.x.reshape(-1, 3)
        except:
            pass
        return circles
    
    best_sum = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    # Multiple trials with different initializations
    for trial in range(15):
        if check_time():
            break
        
        # Create initialization
        if trial == 0:
            # Square grid
            side = 5
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                c = i % side
                rr = i // side
                x[i] = (c + 0.5) * 0.8 / side
                y[i] = (rr + 0.5) * 0.8 / side
                r[i] = 0.8 / side * 0.45
        elif trial == 1:
            # Rectangular grid
            cols = 6
            rows = 4
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                c = i % cols
                rr = i // cols
                x[i] = (c + 0.5) * 1.2 / cols
                y[i] = (rr + 0.5) * 0.8 / rows
                r[i] = min(1.2 / cols, 0.8 / rows) * 0.45
        else:
            # Random perturbation of grid
            side = 5
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                c = i % side
                rr = i // side
                x[i] = (c + 0.5) * 0.8 / side + np.random.normal(0, 0.03)
                y[i] = (rr + 0.5) * 0.8 / side + np.random.normal(0, 0.03)
                r[i] = 0.8 / side * 0.45
        
        init = np.column_stack([x, y, r])
        
        # Force-directed relaxation
        result = force_directed(init)
        
        # Polish
        result = polish(result)
        
        # Check validity
        if is_valid(result):
            s = np.sum(result[:, 2])
            if s > best_sum:
                best_sum = s
                best_result = result.copy()
        
        # Also try polishing the force-directed result without full validation
        # (in case is_valid is too strict)
        if not is_valid(result):
            # Try to fix small violations
            result2 = polish(result, max_iter=200)
            if is_valid(result2):
                s = np.sum(result2[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_result = result2.copy()
    
    if best_result is None:
        # Fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
