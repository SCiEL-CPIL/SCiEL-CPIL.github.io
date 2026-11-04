import numpy as np
from scipy.optimize import minimize
import itertools
import math

N = 26

def compute_score(params):
    """Compute the sum of radii for a valid configuration, or -inf if invalid."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Check boundary
    if np.any(x < -1e-10) or np.any(x > 1 + 1e-10) or np.any(y < -1e-10) or np.any(y > 1 + 1e-10):
        return -np.inf
    if np.any(r < -1e-10):
        return -np.inf
    
    # Check wall distance
    dist_wall = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    if np.any(r > dist_wall + 1e-10):
        return -np.inf
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-10:
                return -np.inf
    
    return np.sum(r)

def construct():
    """
    Pack 26 circles in the unit square, maximizing sum of radii.
    Uses a combination of:
    1. Good initial configurations (grid, hexagonal)
    2. Local optimization with penalty-based objective
    3. Multiple restarts with different strategies
    """
    rng = np.random.default_rng(12345)
    
    def objective(params, eps=1e-5):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        score = -np.sum(r)
        penalty = 0.0
        
        # Wall violations
        dist_wall = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        wall_viol = np.maximum(0, r - dist_wall)
        penalty += np.sum(wall_viol ** 2)
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = math.sqrt(dx * dx + dy * dy)
                overlap = max(0, r[i] + r[j] - dist)
                penalty += overlap ** 2
        
        return score + 1000.0 * penalty
    
    def make_initial_grid(n_rows, n_cols, offset=0.0):
        """Create a grid of circles."""
        positions = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(positions) >= N:
                    break
                x = (j + 0.5) / n_cols + offset * 0.1
                y = (i + 0.5) / n_rows + offset * 0.1
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                r = min(0.5 / n_cols, 0.5 / n_rows) * 0.95
                positions.extend([x, y, r])
        while len(positions) < 3 * N:
            positions.extend([0.5, 0.5, 0.02])
        return np.array(positions[:3 * N])
    
    def make_hexagonal():
        """Create a hexagonal packing initial guess."""
        positions = []
        r_base = 0.08
        dx = 2 * r_base * 0.99
        dy = 2 * r_base * math.sqrt(3) / 2 * 0.99
        row = 0
        while len(positions) < 3 * N:
            col = 0
            while len(positions) < 3 * N:
                x = r_base + col * dx + (row % 2) * dx / 2
                y = r_base + row * dy
                if x < 1 - r_base and y < 1 - r_base and x > r_base and y > r_base:
                    positions.extend([x, y, r_base])
                col += 1
            row += 1
        while len(positions) < 3 * N:
            positions.extend([0.5, 0.5, 0.02])
        return np.array(positions[:3 * N])
    
    def random_init():
        """Random initial configuration."""
        positions = []
        for _ in range(N):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            r = rng.uniform(0.02, 0.08)
            positions.extend([x, y, r])
        return np.array(positions)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Grids with different aspect ratios
    for n_rows in range(3, 6):
        for n_cols in range(4, 8):
            if n_rows * n_cols >= N:
                initial_configs.append(make_initial_grid(n_rows, n_cols, offset=rng.uniform(0, 0.5)))
    
    # Hexagonal
    initial_configs.append(make_hexagonal())
    
    # Random
    for _ in range(10):
        initial_configs.append(random_init())
    
    # Jittered grids
    for _ in range(15):
        n_rows = rng.integers(3, 6)
        n_cols = rng.integers(4, 8)
        base = make_initial_grid(n_rows, n_cols)
        jitter = rng.uniform(-0.05, 0.05, size=3 * N)
        jitter[2::3] *= 0  # Don't jitter radii
        base = base + jitter
        base[0::3] = np.clip(base[0::3], 0, 1)
        base[1::3] = np.clip(base[1::3], 0, 1)
        base[2::3] = np.clip(base[2::3], 0.01, 0.2)
        initial_configs.append(base)
    
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    best_params = None
    best_score = -np.inf
    
    # Optimize from each initial config
    for init in initial_configs:
        try:
            res = minimize(objective, init, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.success or res.fun < 1e-3:
                score = compute_score(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Local refinement: try small perturbations around best solution
    if best_params is not None:
        for _ in range(50):
            perturbed = best_params + rng.normal(0, 0.01, size=3 * N)
            perturbed[0::3] = np.clip(perturbed[0::3], 0, 1)
            perturbed[1::3] = np.clip(perturbed[1::3], 0, 1)
            perturbed[2::3] = np.clip(perturbed[2::3], 0.01, 0.3)
            
            try:
                res = minimize(objective, perturbed, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-14})
                score = compute_score(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        # Fallback: simple grid
        best_params = make_initial_grid(5, 6)
    
    return best_params.reshape(N, 3)
