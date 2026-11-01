"""Circle packing in unit square n=32: repulsive-particle optimization with polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validity(circles):
    """Check if circles are valid (inside unit square, disjoint)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < -r or x > 1 + r or y < -r or y > 1 + r:
            return False
        # Check boundary: circle must be fully inside [0,1]^2
        if x - r < -1e-10 or x + r > 1 + 1e-10 or y - r < -1e-10 or y + r > 1 + 1e-10:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d = np.hypot(xi - xj, yi - yj)
            if d < ri + rj - 1e-10:
                return False
    return True

def _objective(params, n=N):
    """Objective: maximize sum of radii, penalize violations.
    params: 3n array [x1,y1,r1, x2,y2,r2, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Sum of radii (to maximize, so negate)
    val = -np.sum(r)
    
    # Boundary penalties
    for i in range(n):
        if r[i] <= 0:
            val += 1e6 * (0 - r[i])
        # Circle must be inside [0,1]^2
        val += 1e4 * max(0, -x[i] + r[i])**2
        val += 1e4 * max(0, x[i] + r[i] - 1)**2
        val += 1e4 * max(0, -y[i] + r[i])**2
        val += 1e4 * max(0, y[i] + r[i] - 1)**2
    
    # Overlap penalties (soft)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            min_d = r[i] + r[j]
            if d < min_d:
                # Soft penalty: quadratic in overlap
                overlap = min_d - d
                val += 1e3 * overlap**2
    
    return val

def _objective_gradient(params, n=N):
    """Gradient of the objective."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Boundary penalties
    for i in range(n):
        # x - r < 0 penalty: 1e4 * max(0, -x+r)^2
        if -x[i] + r[i] > 0:
            grad[3*i] += 1e4 * 2 * (-x[i] + r[i]) * (-1)
            grad[3*i+2] += 1e4 * 2 * (-x[i] + r[i]) * (1)
        # x + r - 1 > 0 penalty
        if x[i] + r[i] - 1 > 0:
            grad[3*i] += 1e4 * 2 * (x[i] + r[i] - 1) * (1)
            grad[3*i+2] += 1e4 * 2 * (x[i] + r[i] - 1) * (1)
        # y - r < 0 penalty
        if -y[i] + r[i] > 0:
            grad[3*i+1] += 1e4 * 2 * (-y[i] + r[i]) * (-1)
            grad[3*i+2] += 1e4 * 2 * (-y[i] + r[i]) * (1)
        # y + r - 1 > 0 penalty
        if y[i] + r[i] - 1 > 0:
            grad[3*i+1] += 1e4 * 2 * (y[i] + r[i] - 1) * (1)
            grad[3*i+2] += 1e4 * 2 * (y[i] + r[i] - 1) * (1)
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            min_d = r[i] + r[j]
            if d < min_d and d > 1e-15:
                overlap = min_d - d
                # d/dxi of overlap^2 = 2*overlap * d/dxi(min_d - d) = 2*overlap * (-dx/d)
                grad[3*i] += 1e3 * 2 * overlap * (-dx / d)
                grad[3*i+1] += 1e3 * 2 * overlap * (-dy / d)
                grad[3*j] += 1e3 * 2 * overlap * (dx / d)
                grad[3*j+1] += 1e3 * 2 * overlap * (dy / d)
                # dr terms
                grad[3*i+2] += 1e3 * 2 * overlap * (1)
                grad[3*j+2] += 1e3 * 2 * overlap * (1)
    
    return grad

def _initial_grid():
    """6x6 grid initialization."""
    side = 6
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.005
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_clustered():
    """Clustered initialization with some randomness."""
    np.random.seed(42)
    circles = []
    # Place circles in a rough grid with jitter
    side = 6
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side + np.random.uniform(-0.02, 0.02)
            y = (j + 0.5) / side + np.random.uniform(-0.02, 0.02)
            r = 0.5 / side - 0.005
            circles.append([x, y, r])
    return np.array(circles[:N])

def _polish(circles, max_iter=500, tol=1e-10):
    """High-precision polish using gradient descent with adaptive step."""
    n = len(circles)
    params = circles.flatten()
    
    best_params = params.copy()
    best_val = _objective(params)
    
    lr = 0.01
    for it in range(max_iter):
        val = _objective(params)
        if val < best_val:
            best_val = val
            best_params = params.copy()
        
        grad = _objective_gradient(params)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive step size
        step = lr / (grad_norm + 1e-10)
        params_new = params - step * grad
        
        # Clip to valid ranges
        params_new[0::3] = np.clip(params_new[0::3], 0.0, 1.0)
        params_new[1::3] = np.clip(params_new[1::3], 0.0, 1.0)
        params_new[2::3] = np.clip(params_new[2::3], 0.001, 0.5)
        
        new_val = _objective(params_new)
        if new_val < val:
            params = params_new
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_params.reshape(-1, 3)

def _perturb_restart(circles, n_restarts=5, seed=0):
    """Try perturbations and restarts to escape local minima."""
    best = circles.copy()
    best_val = _objective(circles.flatten())
    
    rng = np.random.RandomState(seed)
    for k in range(n_restarts):
        perturbed = circles.copy()
        # Small random perturbation to positions and radii
        perturbed[:, 0] += rng.uniform(-0.05, 0.05, N)
        perturbed[:, 1] += rng.uniform(-0.05, 0.05, N)
        perturbed[:, 2] *= (1 + rng.uniform(-0.1, 0.1, N))
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0.01, 0.99)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0.01, 0.99)
        perturbed[:, 2] = np.clip(perturbed[:, 2], 0.005, 0.45)
        
        polished = _polish(perturbed, max_iter=200, tol=1e-8)
        val = _objective(polished)
        if val < best_val:
            best_val = val
            best = polished
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Try multiple initializations
    inits = [
        _initial_grid(),
        _initial_clustered(),
    ]
    
    # Add a few more diverse inits
    np.random.seed(123)
    for seed_val in [7, 42, 99]:
        rng = np.random.RandomState(seed_val)
        circles = []
        for i in range(N):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            r = rng.uniform(0.03, 0.08)
            circles.append([x, y, r])
        inits.append(np.array(circles))
    
    best_result = None
    best_val = np.inf
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        
        # Polish the initialization
        polished = _polish(init, max_iter=300, tol=1e-9)
        val = _objective(polished)
        
        if val < best_val:
            best_val = val
            best_result = polished
        
        # Try perturbation restarts if time permits
        if time.time() > deadline - 15.0:
            break
        restarted = _perturb_restart(polished, n_restarts=3, seed=seed_val if 'seed_val' in dir() else 0)
        val2 = _objective(restarted)
        if val2 < best_val:
            best_val = val2
            best_result = restarted
    
    if best_result is None:
        best_result = _initial_grid()
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        final = _polish(best_result, max_iter=500, tol=1e-11)
        val_final = _objective(final)
        if val_final < _objective(best_result):
            best_result = final
    
    # Ensure validity: clip and fix any small violations
    circles = best_result.copy()
    for i in range(N):
        x, y, r = circles[i]
        # Ensure circle is inside [0,1]^2
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(r, 0.001)
        x = min(max(x, r), 1 - r)
        y = min(max(y, r), 1 - r)
        circles[i] = [x, y, r]
    
    # Final overlap check and fix: if any overlap, shrink the smaller circle
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d = np.hypot(xi - xj, yi - yj)
            if d < ri + rj:
                # Shrink the smaller one
                if ri <= rj:
                    new_ri = max(0.001, (d - rj) * 0.99)
                    circles[i][2] = new_ri
                else:
                    new_rj = max(0.001, (d - ri) * 0.99)
                    circles[j][2] = new_rj
    
    return circles
# EVOLVE-BLOCK-END
