"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin out of 150s
    
    w = 1.0
    h = 1.0
    
    def unflatten(v):
        return v[:N], v[N:2*N], v[2*N:3*N]
    
    def flatten(x, y, r):
        return np.concatenate([x, y, r])
    
    def is_valid(x, y, r, tol=1e-5):
        """Check if a packing is feasible."""
        if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1+tol) or \
           np.any(y < -tol) or np.any(y > 1+tol):
            return False
        # Wall constraints
        if np.any(r > x + tol) or np.any(r > 1-x + tol) or \
           np.any(r > y + tol) or np.any(r > 1-y + tol):
            return False
        # Circle-circle constraints (vectorized)
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx*dx + dy*dy < (r[i]+r[j])**2 - tol:
                    return False
        return True
    
    def penalty_objective(v, penalty_weight=100.0):
        """Penalty-based objective: minimize -sum(r) + penalty for constraint violations."""
        x, y, r = unflatten(v)
        obj = -np.sum(r)
        # Wall violations
        wall_viol = np.maximum(0, r - x) + np.maximum(0, r - (1-x)) + \
                    np.maximum(0, r - y) + np.maximum(0, r - (1-y))
        obj += penalty_weight * np.sum(wall_viol)
        # Circle-circle violations
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                rsum = r[i] + r[j]
                if rsum > 0:
                    viol = max(0, rsum*rsum - d2) / max(rsum, 1e-10)
                    obj += penalty_weight * viol
        # Radius non-negativity
        obj += penalty_weight * np.sum(np.maximum(0, -r))
        return obj
    
    def expand_radii(x, y, r, max_iters=200, step=0.005):
        """Greedily expand each circle's radius as much as possible."""
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                # Max radius allowed by walls
                r_wall = min(x[i], 1-x[i], y[i], 1-y[i])
                # Max radius allowed by other circles
                r_max = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d > 0:
                        r_max = min(r_max, d - r[j])
                r_max = max(0, r_max)
                if r_max > r[i] + 1e-8:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return r
    
    def local_search(x, y, r, max_iters=100):
        """Local search: try small perturbations to improve total radius."""
        best_r_sum = np.sum(r)
        best_state = (x.copy(), y.copy(), r.copy())
        rng = np.random.RandomState(123)
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            # Pick a random circle and perturb its position
            i = rng.randint(N)
            dx = rng.normal(0, 0.01)
            dy = rng.normal(0, 0.01)
            new_x = x.copy()
            new_y = y.copy()
            new_x[i] = np.clip(x[i] + dx, 0, 1)
            new_y[i] = np.clip(y[i] + dy, 0, 1)
            
            # Try expanding all radii from new positions
            new_r = expand_radii(new_x, new_y, r.copy(), max_iters=50)
            new_sum = np.sum(new_r)
            
            if new_sum > best_r_sum + 1e-8:
                best_r_sum = new_sum
                best_state = (new_x.copy(), new_y.copy(), new_r.copy())
                x, y, r = best_state
        
        return best_state
    
    def slsqp_refine(x, y, r):
        """SLSQP refinement with vectorized penalty objective."""
        v0 = flatten(x, y, r)
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 0.5))
        
        try:
            result = minimize(
                lambda v: penalty_objective(v, 50.0),
                v0, method='SLSQP', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            x_r, y_r, r_r = unflatten(result.x)
            # Clamp and project
            r_r = np.clip(r_r, 0, 0.5)
            x_r = np.clip(x_r, 0, 1)
            y_r = np.clip(y_r, 0, 1)
            # Expand radii to fill available space
            r_r = expand_radii(x_r, y_r, r_r, max_iters=100)
            if is_valid(x_r, y_r, r_r):
                return x_r, y_r, r_r
        except Exception:
            pass
        return x, y, r
    
    # Phase 1: Generate diverse initial packings
    np.random.seed(42)
    best_value = -np.inf
    best_circles = None
    
    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    grid_x = np.array([((i % side) + 0.5) / side for i in range(N)])
    grid_y = np.array([((i // side) + 0.5) / side for i in range(N)])
    grid_r = np.full(N, 0.5 / side - 0.01)
    
    # Process multiple initializations
    initializations = []
    
    # 1. Grid
    initializations.append((grid_x.copy(), grid_y.copy(), grid_r.copy()))
    
    # 2. Perturbed grids
    for k in range(4):
        if time.time() > deadline - 60:
            break
        px = grid_x + np.random.randn(N) * 0.03
        py = grid_y + np.random.randn(N) * 0.03
        pr = grid_r + np.random.randn(N) * 0.01
        px = np.clip(px, 0.02, 0.98)
        py = np.clip(py, 0.02, 0.98)
        pr = np.clip(pr, 0.01, 0.3)
        initializations.append((px, py, pr))
    
    # 3. Random scattered
    for k in range(3):
        if time.time() > deadline - 60:
            break
        rx = np.random.uniform(0.1, 0.9, N)
        ry = np.random.uniform(0.1, 0.9, N)
        rr = np.full(N, 0.05)
        initializations.append((rx, ry, rr))
    
    # Process each initialization through the pipeline
    for idx, (x, y, r) in enumerate(initializations):
        if time.time() > deadline - 40:
            break
        
        # Step 1: Expand radii from initial positions
        r = expand_radii(x, y, r, max_iters=100)
        
        # Step 2: SLSQP refinement
        x, y, r = slsqp_refine(x, y, r)
        
        # Step 3: Local search
        x, y, r = local_search(x, y, r, max_iters=80)
        
        # Step 4: Final SLSQP + expand
        x, y, r = slsqp_refine(x, y, r)
        
        # Verify and record
        if is_valid(x, y, r):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_circles = np.column_stack([x, y, r])
    
    # Final polish: one more expansion pass
    if best_circles is not None:
        x, y, r = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
        r = expand_radii(x, y, r, max_iters=200)
        if is_valid(x, y, r):
            best_circles = np.column_stack([x, y, r])
    
    if best_circles is None:
        # Fallback
        best_circles = np.column_stack([grid_x, grid_y, grid_r])
    
    return best_circles

# EVOLVE-BLOCK-END
