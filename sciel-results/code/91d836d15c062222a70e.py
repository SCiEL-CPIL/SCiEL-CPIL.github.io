"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_init(n, offset=0.0):
    """Generate hexagonal packing initialization for n circles."""
    points = []
    for q in range(-12, 13):
        for r in range(-12, 13):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            points.append([x + offset, y + offset])
    points = np.array(points)
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    radii = np.full(n, 0.08)
    return init_points, radii

def _random_init(n, seed=0):
    """Generate random initialization."""
    rng = np.random.RandomState(seed)
    points = rng.rand(n, 2)
    radii = rng.uniform(0.05, 0.1, n)
    return points, radii

def _compute_penalty_and_grad(vars, C=1000.0):
    """Compute penalty value and gradient."""
    grad = np.zeros(3 * N)
    penalty = 0.0
    
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    grad[2::3] -= 1.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq)
            r_sum = r[i] + r[j]
            
            if d < 1e-9:
                dx = 1e-9
                d = 1e-9
            
            if d < r_sum:
                V = r_sum - d
                penalty += C * V * V
                coeff = 2.0 * C * V
                inv_d = 1.0 / d
                
                grad[3*i + 0] += coeff * (-dx * inv_d)
                grad[3*i + 1] += coeff * (-dy * inv_d)
                grad[3*i + 2] += coeff * 1.0
                
                grad[3*j + 0] += coeff * (dx * inv_d)
                grad[3*j + 1] += coeff * (dy * inv_d)
                grad[3*j + 2] += coeff * 1.0
    
    for i in range(N):
        if x[i] < r[i]:
            V = r[i] - x[i]
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 0] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
            
        if x[i] + r[i] > 1.0:
            V = x[i] + r[i] - 1.0
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 0] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
            
        if y[i] < r[i]:
            V = r[i] - y[i]
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 1] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
            
        if y[i] + r[i] > 1.0:
            V = y[i] + r[i] - 1.0
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 1] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
    
    return penalty, grad

def _is_valid(x, y, r, tol=1e-6):
    """Check if configuration is valid."""
    for i in range(N):
        if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _push_apart(centers, radii, iters=100, tol=1e-8):
    """Iteratively push overlapping circles apart to find feasible starting points."""
    centers = centers.copy()
    radii = radii.copy()
    n = len(centers)
    
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                r_sum = radii[i] + radii[j]
                
                if d < r_sum:
                    if d < 1e-12:
                        dx = 1e-8
                        dy = 0.0
                        d = 1e-8
                    push = (r_sum - d) / 2.0 + 1e-6
                    nx = dx / d
                    ny = dy / d
                    centers[i][0] += push * nx
                    centers[i][1] += push * ny
                    centers[j][0] -= push * nx
                    centers[j][1] -= push * ny
                    moved = True
        
        # Clip to boundary
        for i in range(n):
            centers[i][0] = np.clip(centers[i][0], radii[i], 1.0 - radii[i])
            centers[i][1] = np.clip(centers[i][1], radii[i], 1.0 - radii[i])
        
        if not moved:
            break
    
    return centers, radii

def _compute_radii(centers):
    """Compute max radii for given centers."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        r_wall = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
        min_dist = r_wall
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, d / 2)
        radii[i] = min_dist
    return radii

def _optimize(vars0, start_time, deadline, C=1000.0):
    """Run gradient descent with momentum."""
    vars = vars0.copy()
    velocity = np.zeros_like(vars)
    best_val = -np.sum(vars0[2::3])
    best_vars = vars0.copy()
    
    lr = 0.005
    momentum = 0.9
    steps = 3000
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        penalty, g = _compute_penalty_and_grad(vars, C)
        
        velocity = momentum * velocity - lr * g
        vars += velocity
        
        vars[2::3] = np.maximum(vars[2::3], 0.01)
        vars[0::3] = np.clip(vars[0::3], 0.0, 1.0)
        vars[1::3] = np.clip(vars[1::3], 0.0, 1.0)
        
        current_sum = np.sum(vars[2::3])
        if current_sum > best_val:
            best_val = current_sum
            best_vars = vars.copy()
        
        lr *= 0.998
    
    return best_vars, best_val

def _polish(vars, start_time, deadline):
    """High-precision polish pass."""
    if time.time() > deadline:
        return vars
    
    C = 5000.0
    lr = 0.001
    momentum = 0.9
    steps = 1000
    
    vars = vars.copy()
    velocity = np.zeros_like(vars)
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        penalty, g = _compute_penalty_and_grad(vars, C)
        
        velocity = momentum * velocity - lr * g
        vars += velocity
        
        vars[2::3] = np.maximum(vars[2::3], 0.01)
        vars[0::3] = np.clip(vars[0::3], 0.0, 1.0)
        vars[1::3] = np.clip(vars[1::3], 0.0, 1.0)
        
        lr *= 0.999
    
    return vars

def _scipy_optimize(centers, radii, start_time, deadline):
    """Use scipy L-BFGS-B to optimize centers with radii computed from positions."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return centers, radii, np.sum(radii)
    
    n = len(centers)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    def obj_and_grad(x):
        c = x.reshape(n, 2)
        c = np.clip(c, 1e-6, 1 - 1e-6)
        
        # Compute radii
        radii_local = np.zeros(n)
        for i in range(n):
            r_wall = min(c[i][0], 1 - c[i][0], c[i][1], 1 - c[i][1])
            min_d = r_wall
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(c[i] - c[j])
                    min_d = min(min_d, d / 2)
            radii_local[i] = min_d
        
        f = -np.sum(radii_local)
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for k in range(len(x)):
            x_plus = x.copy()
            x_plus[k] += eps
            c_plus = x_plus.reshape(n, 2)
            c_plus = np.clip(c_plus, 1e-6, 1 - 1e-6)
            radii_plus = np.zeros(n)
            for i in range(n):
                r_wall = min(c_plus[i][0], 1 - c_plus[i][0], c_plus[i][1], 1 - c_plus[i][1])
                min_d = r_wall
                for j in range(n):
                    if i != j:
                        d = np.linalg.norm(c_plus[i] - c_plus[j])
                        min_d = min(min_d, d / 2)
                radii_plus[i] = min_d
            f_plus = -np.sum(radii_plus)
            grad[k] = (f_plus - f) / eps
        
        return f, grad
    
    # Multi-start with perturbations
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            c0 = centers.copy()
        else:
            c0 = centers + np.random.randn(n, 2) * 0.05
            c0 = np.clip(c0, 0.01, 0.99)
        
        try:
            result = minimize(
                lambda x: obj_and_grad(x)[0],
                c0.flatten(),
                jac=lambda x: obj_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < obj_and_grad(c0.flatten())[0]:
                c_new = np.clip(result.x.reshape(n, 2), 1e-6, 1 - 1e-6)
                r_new = _compute_radii(c_new)
                if np.sum(r_new) > best_sum:
                    best_centers = c_new
                    best_radii = r_new
                    best_sum = np.sum(r_new)
        except Exception:
            pass
    
    return best_centers, best_radii, best_sum

def _local_refinement(centers, radii, start_time, deadline):
    """Simple iterative refinement: try small moves for each circle."""
    n = len(centers)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    for iteration in range(200):
        if time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(n):
            best_pos = best_centers[i].copy()
            best_r = best_radii[i]
            
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    new_pos = best_centers[i] + np.array([dx, dy])
                    if new_pos[0] < 0.001 or new_pos[0] > 0.999 or new_pos[1] < 0.001 or new_pos[1] > 0.999:
                        continue
                    
                    new_c = best_centers.copy()
                    new_c[i] = new_pos
                    r_wall = min(new_pos[0], 1 - new_pos[0], new_pos[1], 1 - new_pos[1])
                    min_d = r_wall
                    for j in range(n):
                        if i != j:
                            d = np.linalg.norm(new_pos - new_c[j])
                            min_d = min(min_d, d / 2)
                    
                    if min_d > best_r + 1e-10:
                        best_r = min_d
                        best_pos = new_pos.copy()
                        improved = True
            
            if improved:
                best_centers[i] = best_pos
                best_radii = _compute_radii(best_centers)
    
    return best_centers, best_radii, np.sum(best_radii)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_vars = None
    best_sum = 0.0
    
    # Try multiple initializations
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            points, radii = _hex_init(N)
        elif restart == 1:
            points, radii = _hex_init(N, offset=0.05)
        else:
            points, radii = _random_init(N, seed=restart * 42 + 13)
        
        # Push apart to get feasible starting point
        points, radii = _push_apart(points, radii, iters=50)
        
        # Recompute radii after push apart
        radii = _compute_radii(points)
        
        # Build initial variables
        x0 = np.zeros(3 * N)
        x0[0::3] = points[:, 0]
        x0[1::3] = points[:, 1]
        x0[2::3] = radii
        
        # Optimize with gradient descent
        vars_opt, val = _optimize(x0, start_time, deadline)
        
        if val > best_sum:
            best_sum = val
            best_vars = vars_opt.copy()
    
    if best_vars is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Extract centers and radii from best gradient descent result
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3].copy()
    
    # Try scipy optimization
    if time.time() < deadline - 20:
        centers, radii, scipy_sum = _scipy_optimize(centers, radii, start_time, deadline)
        if scipy_sum > best_sum:
            best_sum = scipy_sum
            best_vars = np.zeros(3 * N)
            best_vars[0::3] = centers[:, 0]
            best_vars[1::3] = centers[:, 1]
            best_vars[2::3] = radii
    
    # Local refinement
    if time.time() < deadline - 5:
        centers, radii, local_sum = _local_refinement(centers, radii, start_time, deadline)
        if local_sum > best_sum:
            best_sum = local_sum
            best_vars = np.zeros(3 * N)
            best_vars[0::3] = centers[:, 0]
            best_vars[1::3] = centers[:, 1]
            best_vars[2::3] = radii
    
    # Polish the best result
    best_vars = _polish(best_vars, start_time, deadline)
    
    # Final validation and repair
    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    r_best = best_vars[2::3]
    
    if not _is_valid(x_best, y_best, r_best):
        r_best = r_best * 0.95
        if not _is_valid(x_best, y_best, r_best):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    
    result = np.column_stack((x_best, y_best, r_best))
    return result

# EVOLVE-BLOCK-END
