"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _validate_and_compute(circles):
    """Compute the sum of radii if circles are valid (non-overlapping, in rectangle with perimeter <= 4)."""
    if circles is None or len(circles) < N:
        return 0.0
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return 0.0
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    w = xmax - xmin
    h = ymax - ymin
    if w < 0 or h < 0:
        return 0.0
    if w + h > 2.0 + 1e-7:  # perimeter = 2(w+h) <= 4
        return 0.0
    # Check non-overlapping
    for i in range(N):
        for j in range(i + 1, N):
            if np.hypot(x[i] - x[j], y[i] - y[j]) < r[i] + r[j] - 1e-7:
                return 0.0
    return np.sum(r)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial placement: grid-like
    side = int(np.ceil(np.sqrt(N)))
    x0 = []
    y0 = []
    r0 = []
    for i in range(N):
        col = i % side
        row = i // side
        x0.append((col + 0.5) / side)
        y0.append((row + 0.5) / side)
        r0.append(0.15)
    
    x0 = np.array(x0)
    y0 = np.array(y0)
    r0 = np.array(r0)
    
    # Objective: maximize sum of radii subject to constraints
    # Use penalty method: minimize -sum(r) + penalty * constraint_violation
    
    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        r = np.maximum(r, 1e-8)  # ensure positive
        sum_r = np.sum(r)
        
        # Constraint violations
        # 1. Non-overlapping
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.hypot(x[i] - x[j], y[i] - y[j])
                viol = r[i] + r[j] - dist
                if viol > 0:
                    overlap_viol += viol * viol
        
        # 2. Bounding rectangle perimeter constraint
        w = np.max(x) - np.min(x)
        h = np.max(y) - np.min(y)
        perim_viol = max(0.0, w + h - 2.0)
        perim_viol = perim_viol * perim_viol
        
        # 3. Keep circles in positive quadrant (for numerical stability)
        bound_viol = 0.0
        for i in range(N):
            bound_viol += max(0.0, -x[i] + r[i]) ** 2
            bound_viol += max(0.0, -y[i] + r[i]) ** 2
        
        penalty = 10.0 * overlap_viol + 10.0 * perim_viol + 1.0 * bound_viol
        return -sum_r + penalty
    
    def gradient(v):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros(3 * N)
        f0 = objective(v)
        for i in range(3 * N):
            v_p = v.copy()
            v_p[i] += eps
            grad[i] = (objective(v_p) - f0) / eps
        return grad
    
    # Use L-BFGS-B with bounds
    x0_flat = np.concatenate([x0, y0, r0])
    lower = np.concatenate([np.zeros(N), np.zeros(N), np.full(N, 1e-4)])
    upper = np.concatenate([np.full(N, 5.0), np.full(N, 5.0), np.full(N, 1.0)])
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple random initializations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            v_init = x0_flat.copy()
        else:
            rng = np.random.default_rng(restart * 42)
            # Random perturbation around grid
            x_rand = x0 + rng.normal(0, 0.05, N)
            y_rand = y0 + rng.normal(0, 0.05, N)
            r_rand = np.maximum(r0 + rng.normal(0, 0.02, N), 1e-4)
            v_init = np.concatenate([x_rand, y_rand, r_rand])
        
        try:
            result = minimize(objective, v_init, method='L-BFGS-B', 
                            bounds=list(zip(lower, upper)),
                            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < 1e-2:
                v_opt = result.x
                circles = np.column_stack([v_opt[:N], v_opt[N:2*N], v_opt[2*N:]])
                score = _validate_and_compute(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles
        except Exception:
            pass
    
    # If no valid solution found, return a valid fallback
    if best_circles is None or best_score <= 0:
        # Fallback: small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        r_small = 0.1
        cells = []
        for i in range(N):
            col = i % side
            row = i // side
            cells.append([r_small + col * 2 * r_small, r_small + row * 2 * r_small, r_small])
        best_circles = np.array(cells)
    
    # Final validation and adjustment
    final_score = _validate_and_compute(best_circles)
    if final_score <= 0:
        # Last resort: very small circles
        r_min = 0.01
        cells = [[(i + 0.5) * 0.1, (j + 0.5) * 0.1, r_min] for j in range(N) for i in range(1)]
        # Actually let's just make a simple valid configuration
        r_min = 0.05
        x = np.linspace(r_min, 2.0 - r_min, N)
        y = np.full(N, r_min)
        r = np.full(N, r_min)
        best_circles = np.column_stack([x, y, r])
    
    return best_circles
# EVOLVE-BLOCK-END

