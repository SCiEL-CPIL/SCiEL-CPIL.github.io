import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                return -np.inf
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < ri + rj - 1e-6:
                    return -np.inf
        return float(np.sum(circles[:, 2]))
    
    def grid_solution(aspect, seed=0):
        """Generate initial grid packing for given aspect ratio."""
        rng = np.random.RandomState(seed)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Try different grid arrangements
        best = None
        for cols in range(1, N+1):
            rows = int(np.ceil(N / cols))
            if cols * rows < N:
                continue
            cell_w = w / cols
            cell_h = h / rows
            circles = []
            for i in range(N):
                c = i % cols
                r = i // cols
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                rad = 0.5 * min(cell_w, cell_h) - 1e-4
                circles.append([cx, cy, rad])
            circles = np.array(circles)
            val = evaluate(circles, w, h)
            if val > (best[0] if best else -np.inf):
                best = (val, circles, w, h)
        
        if best is None:
            # Fallback: single row
            cols = N
            rows = 1
            cell_w = w / cols
            cell_h = h
            circles = []
            for i in range(N):
                cx = (i + 0.5) * cell_w
                cy = 0.5 * cell_h
                rad = 0.5 * min(cell_w, cell_h) - 1e-4
                circles.append([cx, cy, rad])
            circles = np.array(circles)
            val = evaluate(circles, w, h)
            best = (val, circles, w, h)
        
        return best[1], best[2], best[3], best[0]
    
    def optimize(circles0, w, h, max_iter=200):
        """Optimize from initial configuration."""
        x0 = circles0.flatten()
        
        def objective(x):
            c = x.reshape(-1, 3)
            val = -np.sum(c[:, 2])
            penalty = 0.0
            for i in range(N):
                xi, yi, ri = c[i]
                if xi - ri < 0: penalty += (xi - ri)**2 * 50
                if xi + ri > w: penalty += (xi + ri - w)**2 * 50
                if yi - ri < 0: penalty += (yi - ri)**2 * 50
                if yi + ri > h: penalty += (yi + ri - h)**2 * 50
                if ri < 0: penalty += ri**2 * 50
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = c[i]
                    xj, yj, rj = c[j]
                    d = np.sqrt((xi-xj)**2 + (yi-yj)**2 + 1e-10)
                    overlap = ri + rj - d
                    if overlap > 0:
                        penalty += overlap**2 * 50
            return val + penalty
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6})
        
        circles = result.x.reshape(-1, 3)
        val = evaluate(circles, w, h)
        return circles, val
    
    best_value = -np.inf
    best_result = None
    
    # Try multiple aspect ratios
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 0.5, 2.0, 0.4, 2.5, 0.3, 3.0, 4.0, 0.25]
    
    for aspect in aspects:
        if time.time() - start_time > deadline - 5:
            break
        
        circles0, w, h, val0 = grid_solution(aspect)
        
        if val0 > best_value:
            best_value = val0
            best_result = circles0.copy()
        
        # Optimize
        circles, val = optimize(circles0, w, h, max_iter=150)
        
        if val > best_value:
            best_value = val
            best_result = circles.copy()
    
    # Try a few random restarts on the best aspect
    if best_result is not None and time.time() - start_time < deadline - 10:
        # Get the best aspect from best_result
        # We don't track which aspect was best, so just do a few random perturbations
        for _ in range(3):
            if time.time() - start_time > deadline - 5:
                break
            # Perturb the best result
            perturbed = best_result + np.random.randn(*best_result.shape) * 0.05
            # We need to know w, h... let's just re-optimize from scratch with random init
            # Actually, let's skip this for simplicity
    
    if best_result is None:
        # Fallback
        circles, w, h, _ = grid_solution(1.0)
        best_result = circles
    
    return best_result
