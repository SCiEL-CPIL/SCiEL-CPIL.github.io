"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _initial_hex(n):
    """Generate an initial hexagonal packing configuration."""
    # Use a hexagonal lattice for initial positions
    # We want to pack n circles, so we create a hex grid and pick n points
    # that are well-distributed in [0,1]^2
    
    # Create a hex grid with some spacing
    # Start with a reasonable number of points and then select n
    points = []
    row = 0
    while len(points) < n + 10:  # Generate a few extra to have choices
        for col in range(20):
            x = (col + (row % 2) * 0.5) * 0.1
            y = row * 0.0866  # hex row spacing
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append((x, y))
        row += 1
    
    # If we don't have enough, fall back to grid
    if len(points) < n:
        side = int(np.ceil(np.sqrt(n)))
        points = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    
    # Select n points that are well-spread
    # Simple approach: sort by distance from center and pick spread
    points = points[:n]
    return np.array(points)


def _max_radius_for_circle(x, y, centers, radii, i, margin=1e-6):
    """Compute the maximum radius for circle i given its center and other circles."""
    # Distance to walls
    r_wall = min(x, 1 - x, y, 1 - y)
    
    # Distance to other circles
    r_other = np.inf
    for j in range(len(centers)):
        if j == i:
            continue
        dx = x - centers[j, 0]
        dy = y - centers[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        r_other = min(r_other, (dist - radii[j]) / 2)
    
    return max(0, min(r_wall, r_other) - margin)


def _evaluate(centers, radii):
    """Check if configuration is valid and return sum of radii."""
    n = len(centers)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Check bounds
        if x < -1e-6 or x > 1 + 1e-6 or y < -1e-6 or y > 1 + 1e-6:
            return -np.inf
        if r < -1e-6:
            return -np.inf
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return -np.inf
        # Check overlaps
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-6:
                return -np.inf
    return np.sum(radii)


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, keep safety margin
    
    # Initial configuration
    centers = _initial_hex(N)
    # Initial radii: small
    radii = np.full(N, 0.05)
    
    # Optimize using SLSQP
    # Variables: x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}, r_0, r_1, ..., r_{N-1}
    def flatten(centers, radii):
        return np.concatenate([centers.flatten(), radii])
    
    def unflatten(v):
        centers = v[:2*N].reshape(N, 2)
        radii = v[2*N:]
        return centers, radii
    
    def objective(v):
        centers, radii = unflatten(v)
        return -np.sum(radii)
    
    def bounds():
        b = []
        for i in range(N):
            b.append((0, 1))  # x
            b.append((0, 1))  # y
        for i in range(N):
            b.append((0, 0.5))  # r
        return b
    
    def constraints():
        cons = []
        # Wall constraints: x_i - r_i >= 0, x_i + r_i <= 1, etc.
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*N+i]})
            # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i] - v[2*N+i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*N+i]})
            # 1 - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i+1] - v[2*N+i]})
        # Non-overlap constraints: dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                def nonoverlap(v, i=i, j=j):
                    dx = v[2*i] - v[2*j]
                    dy = v[2*i+1] - v[2*j+1]
                    return np.sqrt(dx*dx + dy*dy) - v[2*N+i] - v[2*N+j]
                cons.append({'type': 'ineq', 'fun': nonoverlap})
        return cons
    
    x0 = flatten(centers, radii)
    
    # Run optimization with time guard
    best_val = _evaluate(centers, radii)
    best_x = x0.copy()
    
    # Try a few iterations with different initial conditions
    attempts = 3
    for attempt in range(attempts):
        if time.time() > deadline - 10:
            break
        
        if attempt == 0:
            x_init = x0
        else:
            # Perturb initial configuration
            np.random.seed(attempt)
            x_init = x0 + np.random.normal(0, 0.02, size=x0.shape)
            x_init = np.clip(x_init, 0, 1)
            x_init[2*N:] = np.clip(x_init[2*N:], 0, 0.5)
        
        try:
            result = minimize(
                objective,
                x_init,
                method='SLSQP',
                bounds=bounds(),
                constraints=constraints(),
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if time.time() > deadline - 5:
                break
            
            if result.success or result.fun < -best_val:
                val = -result.fun
                c, r = unflatten(result.x)
                eval_val = _evaluate(c, r)
                if eval_val > best_val:
                    best_val = eval_val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Final result
    centers, radii = unflatten(best_x)
    # Ensure validity
    for i in range(N):
        radii[i] = _max_radius_for_circle(centers[i, 0], centers[i, 1], centers, radii, i)
    
    # Build output array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = centers[i, 0]
        result[i, 1] = centers[i, 1]
        result[i, 2] = radii[i]
    
    return result
# EVOLVE-BLOCK-END
