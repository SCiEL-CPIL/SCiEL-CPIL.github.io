"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_violations_and_grad(pos, radii):
    """Compute constraint violations and their gradients.
    pos: (N, 2) array of centers
    radii: (N,) array of radii
    Returns: (violation_value, grad_pos, grad_radii)
    """
    n = len(radii)
    violation = 0.0
    grad_pos = np.zeros_like(pos)
    grad_radii = np.zeros_like(radii)
    
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # Violation: r_i - min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        # Distance to each boundary
        d_left = x
        d_right = 1.0 - x
        d_bottom = y
        d_top = 1.0 - y
        d_min = min(d_left, d_right, d_bottom, d_top)
        viol = r - d_min
        if viol > 0:
            violation += viol
            # Gradient w.r.t. radius
            grad_radii[i] += 1.0
            # Gradient w.r.t. position: push away from nearest boundary
            if d_min == d_left:
                grad_pos[i, 0] += -1.0  # increase x
            elif d_min == d_right:
                grad_pos[i, 0] += 1.0  # decrease x
            elif d_min == d_bottom:
                grad_pos[i, 1] += -1.0  # increase y
            else:  # d_top
                grad_pos[i, 1] += 1.0  # decrease y
    
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-10:
                dist = 1e-10
                dx, dy = 1e-10, 0.0
            required = radii[i] + radii[j]
            viol = required - dist
            if viol > 0:
                violation += viol
                # Gradient w.r.t. radii
                grad_radii[i] += 1.0
                grad_radii[j] += 1.0
                # Gradient w.r.t. positions: push apart
                inv_dist = 1.0 / dist
                grad_pos[i, 0] += dx * inv_dist
                grad_pos[i, 1] += dy * inv_dist
                grad_pos[j, 0] -= dx * inv_dist
                grad_pos[j, 1] -= dy * inv_dist
    
    return violation, grad_pos, grad_radii


def _objective_and_grad(params, lam=1.0):
    """Objective: -sum(radii) + lam * violation
    params: (3N,) array = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = N
    pos = params[0:2*n:3].reshape(n, 2)  # This is wrong, let's use interleaved
    # Actually, let's use params as [x0, y0, r0, x1, y1, r1, ...]
    pos = np.column_stack([params[0::3], params[1::3]])
    radii = params[2::3]
    
    viol, grad_pos, grad_radii = _compute_violations_and_grad(pos, radii)
    
    obj = -np.sum(radii) + lam * viol
    
    # Gradient
    grad = np.zeros(3*n)
    grad[0::3] = lam * grad_pos[:, 0]
    grad[1::3] = lam * grad_pos[:, 1]
    grad[2::3] = -1.0 + lam * grad_radii
    
    return obj, grad


def _project_to_feasible(pos, radii, max_iter=1000, tol=1e-10):
    """Iteratively project to feasibility by reducing radii and adjusting positions."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        # Check boundary
        for i in range(len(radii)):
            x, y = pos[i]
            r = radii[i]
            d_min = min(x, 1-x, y, 1-y)
            if r > d_min:
                radii[i] = max(0, d_min)
        
        # Check overlaps
        converged = True
        for i in range(len(radii)):
            for j in range(i+1, len(radii)):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dist = 1e-10
                required = radii[i] + radii[j]
                if required > dist:
                    converged = False
                    # Reduce radii proportionally
                    excess = (required - dist) / 2.0
                    radii[i] = max(0, radii[i] - excess)
                    radii[j] = max(0, radii[j] - excess)
        
        if converged:
            break
    
    return pos, radii


def _is_feasible(pos, radii, tol=1e-7):
    """Check if configuration is feasible."""
    n = len(radii)
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _optimize_from_initial(pos0, radii0, time_budget, lam0=10.0):
    """Run optimization from initial configuration. Returns (pos, radii, score)."""
    pos = pos0.copy()
    radii = radii0.copy()
    
    # Project to feasible first
    pos, radii = _project_to_feasible(pos, radii)
    
    current_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    # Penalty-based optimization
    lam = lam0
    start_time = time.time()
    
    for outer_iter in range(50):
        elapsed = time.time() - start_time
        if elapsed > time_budget - 5:
            break
        
        # Build parameter vector
        params = np.zeros(3*N)
        params[0::3] = pos[:, 0]
        params[1::3] = pos[:, 1]
        params[2::3] = radii
        
        # Define objective and gradient for scipy
        def obj_func(p):
            p_pos = np.column_stack([p[0::3], p[1::3]])
            p_rad = p[2::3]
            # Clip radii to be non-negative
            p_rad = np.maximum(p_rad, 0)
            viol, _, _ = _compute_violations_and_grad(p_pos, p_rad)
            return -np.sum(p_rad) + lam * viol
        
        def grad_func(p):
            p_pos = np.column_stack([p[0::3], p[1::3]])
            p_rad = np.maximum(p[2::3], 0)
            viol, grad_pos, grad_radii = _compute_violations_and_grad(p_pos, p_rad)
            grad = np.zeros(3*N)
            grad[0::3] = lam * grad_pos[:, 0]
            grad[1::3] = lam * grad_pos[:, 1]
            grad[2::3] = -1.0 + lam * grad_radii
            # Zero out gradient for negative radii
            neg_mask = p[2::3] < 0
            grad[2::3][neg_mask] = 0
            return grad
        
        # Bounds: positions in [0,1], radii >= 0
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((0, 0.5))  # r
        
        # Use L-BFGS-B
        result = minimize(
            obj_func, params, jac=grad_func,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_params = result.x
        new_pos = np.column_stack([new_params[0::3], new_params[1::3]])
        new_radii = np.maximum(new_params[2::3], 0)
        
        # Project to feasible
        new_pos, new_radii = _project_to_feasible(new_pos, new_radii)
        new_score = np.sum(new_radii)
        
        if new_score > best_score:
            best_score = new_score
            best_pos = new_pos.copy()
            best_radii = new_radii.copy()
        
        pos = new_pos
        radii = new_radii
        
        # Increase penalty if still infeasible
        viol, _, _ = _compute_violations_and_grad(pos, radii)
        if viol > 1e-6:
            lam *= 2.0
        elif viol < 1e-10:
            lam = max(lam0, lam * 0.5)
    
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to leave safety margin
    
    best_overall_score = 0.0
    best_overall_result = None
    
    # Multiple restarts with different initial configurations
    n_restarts = 0
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 30:  # Leave 30s for final polish
        n_restarts += 1
        
        # Different initial configurations
        if n_restarts == 1:
            # Grid-based initial
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append((x, y, r))
            pos = np.array([(c[0], c[1]) for c in cells[:N]])
            radii = np.array([c[2] for c in cells[:N]])
        elif n_restarts == 2:
            # Random initial with larger radii
            pos = rng.uniform(0.1, 0.9, (N, 2))
            radii = rng.uniform(0.03, 0.06, N)
        elif n_restarts == 3:
            # Hexagonal packing initial
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            r = 0.05
            dx = 2 * r
            dy = 2 * r * np.sqrt(3) / 2
            idx = 0
            row = 0
            while idx < N:
                for col in range(20):
                    if idx >= N:
                        break
                    x = 0.1 + col * dx + (row % 2) * dx/2
                    y = 0.1 + row * dy
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        pos[idx, 0] = x
                        pos[idx, 1] = y
                        radii[idx] = r
                        idx += 1
                row += 1
        else:
            # More random restarts with varying density
            scale = 0.8 + 0.4 * rng.random()
            pos = rng.uniform(0.05, 0.95, (N, 2))
            radii = rng.uniform(0.02, 0.08, N)
        
        # Run optimization
        budget = min(10.0, max(2.0, (deadline - time.time()) / 3))
        pos_opt, radii_opt, score = _optimize_from_initial(pos, radii, budget)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_result = (pos_opt, radii_opt)
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    if best_overall_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append((x, y, r))
        result = np.array(cells[:N])
        return result
    
    pos_opt, radii_opt = best_overall_result
    
    # Final polish: try to increase radii slightly while maintaining feasibility
    if time.time() < deadline - 10:
        for _ in range(200):
            improved = False
            for i in range(N):
                # Try to increase radius of circle i
                for dr in [0.001, 0.0005, 0.0002, 0.0001]:
                    test_radii = radii_opt.copy()
                    test_radii[i] += dr
                    if _is_feasible(pos_opt, test_radii, tol=1e-8):
                        radii_opt = test_radii
                        improved = True
                        break
            if not improved:
                break
    
    # Build result
    result = np.zeros((N, 3))
    result[:, 0] = pos_opt[:, 0]
    result[:, 1] = pos_opt[:, 1]
    result[:, 2] = radii_opt
    
    return result
# EVOLVE-BLOCK-END
