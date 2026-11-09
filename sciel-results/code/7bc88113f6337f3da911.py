"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds total budget
    
    best_result = None
    best_value = 0.0
    
    def evaluate(circles):
        """Evaluate the sum of radii, checking validity."""
        if len(circles) != N:
            return 0.0
        for i in range(N):
            x, y, r = circles[i]
            if r <= 0:
                return 0.0
            if x < r or x > 1 - r or y < r or y > 1 - r:
                return 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                if dist < circles[i][2] + circles[j][2] - 1e-9:
                    return 0.0
        return sum(circles[:, 2])
    
    def objective(params):
        """Objective function for optimization: minimize negative sum of radii with penalty for violations."""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Penalty for boundary violations
        penalty = 0.0
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] > 1 - r[i]:
                penalty += (x[i] - (1 - r[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] > 1 - r[i]:
                penalty += (y[i] - (1 - r[i]))**2
        
        # Penalty for circle overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(r) + 100.0 * penalty
    
    def make_circles(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        return np.column_stack([x, y, r])
    
    # Try multiple random initializations
    np.random.seed(42)
    
    # Start with a reasonable grid-based initialization
    side = int(np.ceil(np.sqrt(N)))
    base_r = 0.5 / side
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Create initial positions
        if restart == 0:
            # Grid initialization
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        positions.append([(i + 0.5) / side, (j + 0.5) / side, base_r * 0.9])
            circles_init = np.array(positions[:N])
        else:
            # Random initialization with some structure
            circles_init = np.zeros((N, 3))
            for i in range(N):
                # Place in a loose grid with jitter
                row = i // side
                col = i % side
                circles_init[i, 0] = (col + 0.5) / side + np.random.uniform(-0.05, 0.05)
                circles_init[i, 1] = (row + 0.5) / side + np.random.uniform(-0.05, 0.05)
                circles_init[i, 2] = base_r * np.random.uniform(0.7, 1.1)
            
            # Ensure within bounds
            circles_init[:, 0] = np.clip(circles_init[:, 0], 0.05, 0.95)
            circles_init[:, 1] = np.clip(circles_init[:, 1], 0.05, 0.95)
            circles_init[:, 2] = np.clip(circles_init[:, 2], 0.01, 0.3)
        
        # Flatten for optimization
        params_init = circles_init.flatten()
        
        # Bounds: x, y in [0, 1], r in [0.001, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.001, 0.5))  # r
        
        # Run L-BFGS-B
        try:
            result = minimize(
                objective,
                params_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < objective(params_init):
                circles = make_circles(result.x)
                val = evaluate(circles)
                if val > best_value:
                    best_value = val
                    best_result = circles.copy()
        except Exception:
            pass
    
    # Fallback to trivial seed if nothing better found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
