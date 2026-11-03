import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0  # Keep a safety margin
    
    best_sum = 0.0
    best_circles = None
    
    def check_valid(circles):
        """Check if circles are valid (inside unit square, no overlaps)."""
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(x < 0) or np.any(y < 0) or np.any(x > 1) or np.any(y > 1):
            return False
        if np.any(r <= 0):
            return False
        # Check boundary: each circle must fit
        if np.any(r > x) or np.any(r > y) or np.any(r > 1 - x) or np.any(r > 1 - y):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-9:
                    return False
        return True
    
    def objective(params):
        """Penalized objective: minimize negative sum of radii + penalties."""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Negative sum of radii (we want to maximize)
        obj = -np.sum(r)
        
        # Boundary penalty: circles must stay inside [0,1]^2
        # For each circle: r <= min(x, y, 1-x, 1-y)
        boundary_viol = np.maximum(0, r - np.minimum(x, np.minimum(y, np.minimum(1-x, 1-y))))
        obj += 100.0 * np.sum(boundary_viol**2)
        
        # Overlap penalty: for each pair, dist >= r_i + r_j
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = np.maximum(0, r[i] + r[j] - d)
                obj += 50.0 * overlap**2
        
        # Radius positivity penalty
        obj += 10.0 * np.sum(np.maximum(0, -r)**2)
        
        return obj
    
    def objective_vectorized(params):
        """Vectorized penalized objective."""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        obj = -np.sum(r)
        
        # Boundary penalty
        boundary_viol = np.maximum(0, r - np.minimum(x, np.minimum(y, np.minimum(1-x, 1-y))))
        obj += 100.0 * np.sum(boundary_viol**2)
        
        # Overlap penalty - vectorized
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(r_sum, 0)
        overlap = np.maximum(0, r_sum - dists)
        obj += 50.0 * np.sum(overlap**2)
        
        # Radius positivity
        obj += 10.0 * np.sum(np.maximum(0, -r)**2)
        
        return obj
    
    def gradient_vectorized(params):
        """Gradient of vectorized objective."""
        x = params[:N].copy()
        y = params[N:2*N].copy()
        r = params[2*N:3*N].copy()
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        gr = np.zeros(N)
        
        # From -sum(r)
        gr -= 1.0
        
        # Boundary penalty gradient
        min_dist = np.minimum(x, np.minimum(y, np.minimum(1-x, 1-y)))
        boundary_viol = np.maximum(0, r - min_dist)
        # d(obj)/dx from boundary
        for i in range(N):
            if boundary_viol[i] > 0:
                # Which constraint is active?
                min_val = min_dist[i]
                if x[i] == min_val:
                    gx[i] += 200.0 * boundary_viol[i]
                if y[i] == min_val:
                    gy[i] += 200.0 * boundary_viol[i]
                if (1-x[i]) == min_val:
                    gx[i] -= 200.0 * boundary_viol[i]
                if (1-y[i]) == min_val:
                    gy[i] -= 200.0 * boundary_viol[i]
                gr[i] += 200.0 * boundary_viol[i]
        
        # Overlap penalty gradient
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(dists, 1e-10)
        np.fill_diagonal(r_sum, 0)
        overlap = np.maximum(0, r_sum - dists)
        
        for i in range(N):
            for j in range(i+1, N):
                if overlap[i, j] > 0:
                    d = dists[i, j]
                    if d > 1e-10:
                        grad_d = 100.0 * overlap[i, j]
                        gx[i] += grad_d * (x[i] - x[j]) / d
                        gy[i] += grad_d * (y[i] - y[j]) / d
                        gx[j] -= grad_d * (x[i] - x[j]) / d
                        gy[j] -= grad_d * (y[i] - y[j]) / d
                        gr[i] += grad_d
                        gr[j] += grad_d
        
        # Radius positivity gradient
        neg_r = np.maximum(0, -r)
        gr += 20.0 * np.where(neg_r > 0, -1, 0)
        
        return np.concatenate([gx, gy, gr])
    
    def initial_hexagonal(seed=0):
        """Create a hexagonal-like initial layout."""
        np.random.seed(seed)
        # Use a grid with some jitter
        rows = 5
        cols = 6  # 5*6 = 30, we'll use 26
        positions = []
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= N:
                    break
                # Hexagonal offset
                x_offset = 0.5 / cols if row % 2 == 0 else 1.0 / cols
                x = x_offset + col * (1.0 - 2*x_offset) / (cols - 1) if cols > 1 else 0.5
                y = (row + 0.5) / rows
                # Add small random jitter
                x += np.random.randn() * 0.01
                y += np.random.randn() * 0.01
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                positions.append((x, y))
        
        positions = np.array(positions[:N])
        # Initial radii: small, based on minimum distance to boundary
        min_r = 0.03
        radii = np.full(N, min_r)
        return positions, radii
    
    def initial_random(seed=0):
        """Random initial layout with small radii."""
        np.random.seed(seed)
        positions = np.random.rand(N, 2) * 0.8 + 0.1
        radii = np.random.rand(N) * 0.02 + 0.01
        return positions, radii
    
    def optimize_from_initial(pos, rad, seed=0):
        """Run L-BFGS-B optimization from given initial conditions."""
        if time.time() - start_time > time_limit - 5:
            return None, 0
        
        params0 = np.concatenate([pos.flatten(), rad])
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
        for i in range(N):
            bounds.append((0.0, 1.0))  # y
        for i in range(N):
            bounds.append((0.001, 0.5))  # r
        
        try:
            result = minimize(
                objective_vectorized,
                params0,
                method='L-BFGS-B',
                jac=gradient_vectorized,
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            return None, 0
        
        if time.time() - start_time > time_limit - 5:
            return None, 0
        
        params = result.x
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        circles = np.column_stack([x, y, r])
        
        # Check validity
        if not check_valid(circles):
            # Try to fix by shrinking radii
            for _ in range(50):
                r *= 0.95
                circles = np.column_stack([x, y, r])
                if check_valid(circles):
                    break
        
        if not check_valid(circles):
            return None, 0
        
        total = np.sum(r)
        return circles, total
    
    # Try multiple initializations
    initializers = [
        ("hex0", lambda: initial_hexagonal(0)),
        ("hex1", lambda: initial_hexagonal(1)),
        ("hex2", lambda: initial_hexagonal(42)),
        ("rand0", lambda: initial_random(0)),
        ("rand1", lambda: initial_random(1)),
        ("rand2", lambda: initial_random(7)),
        ("rand3", lambda: initial_random(123)),
        ("rand4", lambda: initial_random(999)),
    ]
    
    for name, init_func in initializers:
        if time.time() - start_time > time_limit - 10:
            break
        
        pos, rad = init_func()
        circles, total = optimize_from_initial(pos, rad)
        
        if circles is not None and total > best_sum:
            best_sum = total
            best_circles = circles.copy()
    
    # If no valid result, use fallback
    if best_circles is None:
        # Fallback: small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append((x, y, r))
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
