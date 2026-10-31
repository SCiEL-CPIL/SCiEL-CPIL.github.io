"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 21
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

def compute_value(circles):
    """Compute the sum of radii. Returns -inf if invalid."""
    if len(circles) != N:
        return -np.inf
    # Check radii positive
    if np.any(circles[:, 2] <= 0):
        return -np.inf
    # Check perimeter constraint
    # Bounding rectangle of circles
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    w = x_max - x_min
    h = y_max - y_min
    perimeter = 2 * (w + h)
    if perimeter > 4.0 + 1e-6:
        return -np.inf
    # Check disjoint interiors (distances >= sum of radii)
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                valid = False
                break
        if not valid:
            break
    if not valid:
        return -np.inf
    return np.sum(rs)

def initial_grid():
    """Create initial grid packing."""
    # Use a 5x5 grid for 21 circles (4x5=20, so 5x5=25, take 21)
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
    return np.array(cells)

def optimize_packing(circles, time_limit=60.0):
    """Optimize the packing using SLSQP."""
    start_time = time.time()
    
    def objective(x):
        """Negative sum of radii (we minimize)."""
        if time.time() - start_time > time_limit:
            return 1e10
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        # Perimeter constraint
        x_min = np.min(xs - rs)
        x_max = np.max(xs + rs)
        y_min = np.min(ys - rs)
        y_max = np.max(ys + rs)
        w = x_max - x_min
        h = y_max - y_min
        perimeter = 2 * (w + h)
        # Maximize sum of radii subject to perimeter <= 4
        # We'll use a penalty approach or direct constraint
        return -np.sum(rs)
    
    def constraint_perimeter(x):
        """Perimeter <= 4."""
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        x_min = np.min(xs - rs)
        x_max = np.max(xs + rs)
        y_min = np.min(ys - rs)
        y_max = np.max(ys + rs)
        w = x_max - x_min
        h = y_max - y_min
        return 4.0 - 2 * (w + h)
    
    def constraint_separation(x, i, j):
        """Distance between circles i and j >= sum of radii."""
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - rs[i] - rs[j]
    
    def constraint_positive_radius(x, i):
        """Radius must be positive."""
        return x[2*N + i]
    
    # Build constraints
    constraints = [{'type': 'ineq', 'fun': constraint_perimeter}]
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: constraint_separation(x, i, j)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_positive_radius(x, i)})
    
    # Initial point
    x0 = circles.flatten()
    
    # Bounds: positions in [0, 2], radii in [0.01, 1.0]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))  # x
    for i in range(N):
        bounds.append((0.0, 2.0))  # y
    for i in range(N):
        bounds.append((0.01, 1.0))  # r
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    if result.success:
        x_opt = result.x
        new_circles = np.array([[x_opt[i], x_opt[N+i], x_opt[2*N+i]] for i in range(N)])
        return new_circles
    else:
        return circles

def local_search(circles, n_iterations=100):
    """Local search with 8-direction moves."""
    for _ in range(n_iterations):
        improved = False
        for i in range(N):
            best_circle = circles[i].copy()
            best_value = compute_value(circles)
            # Try 8 directions
            for dx, dy in [(-0.01, 0), (0.01, 0), (0, -0.01), (0, 0.01), 
                          (-0.01, -0.01), (0.01, -0.01), (-0.01, 0.01), (0.01, 0.01)]:
                test_circles = circles.copy()
                test_circles[i, 0] += dx
                test_circles[i, 1] += dy
                val = compute_value(test_circles)
                if val > best_value:
                    best_value = val
                    best_circle = test_circles[i].copy()
            if not np.allclose(best_circle, circles[i]):
                circles[i] = best_circle
                improved = True
        if not improved:
            break
    return circles

def expand_radii(circles, n_passes=10):
    """Try to expand radii while maintaining constraints."""
    for _ in range(n_passes):
        improved = False
        for i in range(N):
            best_r = circles[i, 2]
            # Try increasing radius
            for delta in [0.005, 0.002, 0.001, 0.0005]:
                test_circles = circles.copy()
                test_circles[i, 2] += delta
                if compute_value(test_circles) > compute_value(circles):
                    best_r += delta
                    improved = True
            circles[i, 2] = best_r
        if not improved:
            break
    return circles

def construct():
    start_time = time.time()
    
    # Initial grid
    circles = initial_grid()
    
    # First optimization pass
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        circles = optimize_packing(circles, time_limit=min(remaining - 5, 60))
    
    # Local search
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        circles = local_search(circles, n_iterations=min(50, int(remaining // 2)))
    
    # Expand radii
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        circles = expand_radii(circles, n_passes=min(20, int(remaining // 2)))
    
    # Second optimization pass
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 15:
        circles = optimize_packing(circles, time_limit=min(remaining - 5, 45))
    
    # Final local search
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        circles = local_search(circles, n_iterations=min(30, int(remaining // 2)))
    
    # Final expansion
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        circles = expand_radii(circles, n_passes=min(10, int(remaining)))
    
    # Ensure validity
    if compute_value(circles) < 0:
        # Fallback to initial grid
        circles = initial_grid()
    
    return circles
# EVOLVE-BLOCK-END
