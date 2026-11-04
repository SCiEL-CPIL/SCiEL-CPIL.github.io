import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    deadline = time.time() + 140  # 140s to leave 10s margin before 150s kill
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple aspect ratios and initializations
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.67, 2.0, 0.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 / (1.0 + 1.0/ar)  # w + h = 2, h = w/ar
        h = 2.0 - w
        
        # Hexagonal initialization
        init = hexagonal_init(N, w, h)
        
        # Optimize
        result = optimize_packing(init, w, h, deadline)
        
        if result is not None:
            score = result['score']
            if score > best_score:
                best_score = score
                best_result = result['circles']
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    return fallback_grid()


def hexagonal_init(n, w, h):
    """Create hexagonal packing initialization."""
    circles = []
    # Try to fit circles in hexagonal pattern
    # Start with a base radius estimate
    base_r = min(w, h) / (2 * np.ceil(np.sqrt(n))) * 0.9
    
    # Generate hexagonal positions
    rows = 0
    cols = 0
    while len(circles) < n:
        for col in range(cols + 1):
            if len(circles) >= n:
                break
            x = (col + 0.5) * base_r * 2 * 0.866  # hex spacing
            y = (rows + 0.5) * base_r * 2 * 0.75
            if rows % 2 == 1:
                x += base_r * 0.866
            # Scale to fit in rectangle
            x_scaled = x / (cols + 1) * w * 0.95 + w * 0.025
            y_scaled = y / (rows + 1) * h * 0.95 + h * 0.025
            
            # Clip to valid range
            x_scaled = np.clip(x_scaled, base_r * 0.5, w - base_r * 0.5)
            y_scaled = np.clip(y_scaled, base_r * 0.5, h - base_r * 0.5)
            
            circles.append([x_scaled, y_scaled, base_r * 0.9])
        rows += 1
        cols = min(cols + 1, int(w / (base_r * 1.732)) + 1)
        if rows > 20:  # safety
            break
    
    # If we didn't get enough, add more
    while len(circles) < n:
        i = len(circles)
        circles.append([w * (i+1)/(n+1), h * 0.5, base_r * 0.5])
    
    return np.array(circles[:n])


def objective(params, w, h, n):
    """Compute objective: -sum(r) + penalty for violations."""
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Penalty for going outside bounds
    penalty = 0.0
    penalty += np.sum(np.maximum(0, r - x)**2)
    penalty += np.sum(np.maximum(0, x + r - w)**2)
    penalty += np.sum(np.maximum(0, r - y)**2)
    penalty += np.sum(np.maximum(0, y + r - h)**2)
    
    # Penalty for negative radii
    penalty += np.sum(np.maximum(0, -r)**2)
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return -np.sum(r) + 100.0 * penalty


def optimize_packing(init, w, h, deadline):
    """Run L-BFGS-B optimization."""
    n = len(init)
    x0 = np.concatenate([init[:, 0], init[:, 1], init[:, 2]])
    
    def obj(params):
        return objective(params, w, h, n)
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    params = result.x
    circles = np.column_stack([params[0:n], params[n:2*n], params[2*n:3*n]])
    
    # Verify validity
    if not is_valid(circles, w, h):
        return None
    
    score = np.sum(circles[:, 2])
    return {'circles': circles, 'score': score, 'w': w, 'h': h}


def is_valid(circles, w, h):
    """Check if packing is valid."""
    n = len(circles)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Check bounds
    if np.any(r <= 0):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > w + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > h + 1e-9):
        return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    
    return True


def fallback_grid():
    """Simple grid fallback."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
