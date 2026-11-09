"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _penalty_and_grad(pos, radii):
    """
    Compute penalty (sum of squared violations) and its gradient.
    pos: (n, 2) positions
    radii: (n,) radii
    Returns (penalty, grad_pos, grad_radii)
    """
    n = len(pos)
    penalty = 0.0
    grad_pos = np.zeros_like(pos)
    grad_radii = np.zeros_like(radii)

    # Wall constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # Violation if pos - r < 0 or 1 - pos - r < 0
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        # Left wall: x - r < 0
        v = x - r
        if v < 0:
            v2 = v * v
            penalty += v2
            grad_pos[i, 0] += -2 * v  # d/dx of (x-r)^2 = 2(x-r), but violation is when x-r<0
            # Actually penalty = (x-r)^2 when x-r<0. d/dx = 2(x-r). Since x-r<0, this is negative, pushing x up.
            # Wait, we want to push x up if x-r<0. Gradient of penalty w.r.t x is 2(x-r). 
            # In minimization, we go opposite to gradient. So if grad is negative, we increase x. Correct.
            grad_radii[i] += 2 * v  # d/dr of (x-r)^2 = -2(x-r) = 2(r-x). Since x-r<0, r-x>0, so positive grad on r, we decrease r. Correct.
        
        # Right wall: 1 - x - r < 0
        v = 1 - x - r
        if v < 0:
            v2 = v * v
            penalty += v2
            grad_pos[i, 0] += 2 * v  # d/dx of (1-x-r)^2 = -2(1-x-r). Since 1-x-r<0, -2(1-x-r)>0, so grad positive, we decrease x. Correct.
            grad_radii[i] += 2 * v  # d/dr of (1-x-r)^2 = -2(1-x-r). Same sign. We decrease r. Correct.
        
        # Bottom wall: y - r < 0
        v = y - r
        if v < 0:
            v2 = v * v
            penalty += v2
            grad_pos[i, 1] += -2 * v
            grad_radii[i] += 2 * v
        
        # Top wall: 1 - y - r < 0
        v = 1 - y - r
        if v < 0:
            v2 = v * v
            penalty += v2
            grad_pos[i, 1] += 2 * v
            grad_radii[i] += 2 * v

    # Pairwise constraints: dist(i,j) - r_i - r_j >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                # Avoid division by zero
                dist = 1e-12
            v = dist - radii[i] - radii[j]
            if v < 0:
                v2 = v * v
                penalty += v2
                # d/dpos_i of (dist - ri - rj)^2 = 2(dist - ri - rj) * (dx/dist, dy/dist)
                factor = 2 * v / dist
                grad_pos[i, 0] += factor * dx
                grad_pos[i, 1] += factor * dy
                grad_pos[j, 0] -= factor * dx
                grad_pos[j, 1] -= factor * dy
                # d/dri = -2(dist - ri - rj) = -2v. Since v<0, -2v>0. We decrease ri.
                grad_radii[i] += -2 * v
                grad_radii[j] += -2 * v

    return penalty, grad_pos, grad_radii


def _objective_and_grad(x, n):
    """
    x: (2n + n,) vector = [pos_x, pos_y, radii]
    Returns (objective_value, gradient)
    objective = -sum(radii) + lambda * penalty
    """
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    
    penalty, grad_pos, grad_radii = _penalty_and_grad(pos, radii)
    
    lam = 10.0  # Penalty weight
    
    obj = -np.sum(radii) + lam * penalty
    
    grad = np.zeros_like(x)
    grad[:2 * n] = grad_pos.reshape(-1) * lam
    grad[2 * n:] = -1.0 + grad_radii * lam
    
    return obj, grad


def _random_initialization(n, seed):
    """Generate random initial positions and small radii."""
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    radii = np.full(n, 0.05)
    x = np.concatenate([pos.flatten(), radii])
    return x


def _evaluate_solution(x, n):
    """Check validity and compute score (sum of radii) for a solution."""
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    
    # Check walls
    for i in range(n):
        if pos[i, 0] - radii[i] < -1e-6 or 1 - pos[i, 0] - radii[i] < -1e-6:
            return -np.inf
        if pos[i, 1] - radii[i] < -1e-6 or 1 - pos[i, 1] - radii[i] < -1e-6:
            return -np.inf
    
    # Check pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-6:
                return -np.inf
    
    return np.sum(radii)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    best_score = 0.0
    best_x = None
    
    n = N
    
    # Try multiple restarts with different seeds
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = restart * 1000 + 42
        x0 = _random_initialization(n, seed)
        
        # Use L-BFGS-B with analytical gradient
        res = minimize(
            _objective_and_grad,
            x0,
            args=(n,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            break
        
        # Evaluate the result
        score = _evaluate_solution(res.x, n)
        if score > best_score:
            best_score = score
            best_x = res.x.copy()
    
    if best_x is None:
        # Fallback to trivial solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Convert to (n, 3) array
    pos = best_x[:2 * n].reshape(n, 2)
    radii = best_x[2 * n:]
    result = np.column_stack([pos, radii])
    
    return result

# EVOLVE-BLOCK-END
