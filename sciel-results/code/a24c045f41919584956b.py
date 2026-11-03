"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 26
DEADLINE = 120.0
SAFETY_MARGIN = 5.0  # seconds


def _softplus(x, beta=100.0):
    """Numerically stable softplus approximation of max(0, x)."""
    # For large beta, softplus(x) ≈ max(0, x)
    # Use the log-sum-exp trick for stability
    return np.where(x > 20 / beta, x, np.logaddexp(0, beta * x) / beta)


def _compute_objective_and_grad(x, y, r, beta=100.0, lam=100.0):
    """
    Compute the smooth objective and its gradient.
    
    Variables: x (N,), y (N,), r (N,)
    Objective: sum(r) - lam * sum of violation softplus terms
    
    Constraints (converted to violations that should be <= 0):
    - x_i >= r_i  -> violation = r_i - x_i
    - x_i <= 1 - r_i -> violation = x_i + r_i - 1
    - y_i >= r_i  -> violation = r_i - y_i
    - y_i <= 1 - r_i -> violation = y_i + r_i - 1
    - r_i >= 0    -> violation = -r_i
    - For i != j: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
      -> violation = (r_i + r_j)^2 - (x_i - x_j)^2 - (y_i - y_j)^2
    """
    n = len(x)
    obj = np.sum(r)
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.ones(n)  # d(sum r)/dr = 1
    
    # Boundary constraints
    # x_i >= r_i: violation = r_i - x_i
    viol_x_lower = r - x
    sp = _softplus(viol_x_lower, beta)
    obj -= lam * np.sum(sp)
    # d/dx_i of softplus(r_i - x_i) = -sigmoid(beta*(r_i - x_i))
    sig = 1.0 / (1.0 + np.exp(-beta * viol_x_lower))
    grad_x -= lam * (-sig)
    # d/dr_i of softplus(r_i - x_i) = sigmoid(beta*(r_i - x_i))
    grad_r -= lam * sig
    
    # x_i <= 1 - r_i: violation = x_i + r_i - 1
    viol_x_upper = x + r - 1.0
    sp = _softplus(viol_x_upper, beta)
    obj -= lam * np.sum(sp)
    sig = 1.0 / (1.0 + np.exp(-beta * viol_x_upper))
    grad_x -= lam * sig
    grad_r -= lam * sig
    
    # y_i >= r_i: violation = r_i - y_i
    viol_y_lower = r - y
    sp = _softplus(viol_y_lower, beta)
    obj -= lam * np.sum(sp)
    sig = 1.0 / (1.0 + np.exp(-beta * viol_y_lower))
    grad_y -= lam * (-sig)
    grad_r -= lam * sig
    
    # y_i <= 1 - r_i: violation = y_i + r_i - 1
    viol_y_upper = y + r - 1.0
    sp = _softplus(viol_y_upper, beta)
    obj -= lam * np.sum(sp)
    sig = 1.0 / (1.0 + np.exp(-beta * viol_y_upper))
    grad_y -= lam * sig
    grad_r -= lam * sig
    
    # r_i >= 0: violation = -r_i
    viol_r = -r
    sp = _softplus(viol_r, beta)
    obj -= lam * np.sum(sp)
    sig = 1.0 / (1.0 + np.exp(-beta * viol_r))
    grad_r -= lam * (-sig)
    
    # Pairwise non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    # violation = (r_i+r_j)^2 - (x_i-x_j)^2 - (y_i-y_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            rsum = r[i] + r[j]
            viol = rsum * rsum - dist_sq
            
            if viol > 20.0 / beta:
                # In the linear regime
                sp_val = viol
                # d/dx_i of softplus(viol) where viol = rsum^2 - dx^2 - dy^2
                # d(viol)/dx_i = -2*dx (since dx = x_i - x_j, d/dx_i = 1)
                # d(viol)/dy_i = -2*dy
                # d(viol)/dr_i = 2*rsum
                # d(viol)/dr_j = 2*rsum
                dviol_dx_i = -2.0 * dx
                dviol_dy_i = -2.0 * dy
                dviol_dr_i = 2.0 * rsum
                dviol_dr_j = 2.0 * rsum
                
                obj -= lam * sp_val
                grad_x[i] -= lam * dviol_dx_i
                grad_x[j] -= lam * (-dviol_dx_i)  # d/dx_j of viol = 2*dx
                grad_y[i] -= lam * dviol_dy_i
                grad_y[j] -= lam * (-dviol_dy_i)
                grad_r[i] -= lam * dviol_dr_i
                grad_r[j] -= lam * dviol_dr_j
            else:
                sp_val = np.logaddexp(0, beta * viol) / beta
                sig = 1.0 / (1.0 + np.exp(-beta * viol))
                dviol_dx_i = -2.0 * dx
                dviol_dy_i = -2.0 * dy
                dviol_dr_i = 2.0 * rsum
                dviol_dr_j = 2.0 * rsum
                
                obj -= lam * sp_val
                grad_x[i] -= lam * sig * dviol_dx_i
                grad_x[j] -= lam * sig * (-dviol_dx_i)
                grad_y[i] -= lam * sig * dviol_dy_i
                grad_y[j] -= lam * sig * (-dviol_dy_i)
                grad_r[i] -= lam * sig * dviol_dr_i
                grad_r[j] -= lam * sig * dviol_dr_j
    
    return obj, grad_x, grad_y, grad_r


def _repair(x, y, r):
    """Ensure strict validity by shrinking circles if needed."""
    n = len(x)
    for _ in range(50):
        modified = False
        # Check boundaries
        for i in range(n):
            # x_i must be in [r_i, 1 - r_i]
            min_x = r[i]
            max_x = 1.0 - r[i]
            if x[i] < min_x:
                r[i] = max(0.0, min(r[i], x[i]))
                modified = True
            elif x[i] > max_x:
                r[i] = max(0.0, min(r[i], 1.0 - x[i]))
                modified = True
            
            min_y = r[i]
            max_y = 1.0 - r[i]
            if y[i] < min_y:
                r[i] = max(0.0, min(r[i], y[i]))
                modified = True
            elif y[i] > max_y:
                r[i] = max(0.0, min(r[i], 1.0 - y[i]))
                modified = True
        
        # Check pairwise
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                rsum = r[i] + r[j]
                if rsum > dist:
                    # Shrink both proportionally
                    scale = dist / rsum * 0.999
                    r[i] *= scale
                    r[j] *= scale
                    modified = True
        
        if not modified:
            break
    
    # Final safety: clamp radii to be non-negative and ensure boundary feasibility
    for i in range(n):
        r[i] = max(0.0, r[i])
        # Ensure circle fits in square
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        r[i] = max(0.0, r[i])
    
    return x, y, r


def _is_valid(x, y, r, tol=1e-6):
    """Check if the packing is valid."""
    n = len(x)
    for i in range(n):
        if x[i] < r[i] - tol or x[i] > 1.0 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1.0 - r[i] + tol:
            return False
        if r[i] < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            rsum = r[i] + r[j]
            if dist_sq < rsum * rsum - tol:
                return False
    return True


def _objective_wrapper(z, beta, lam):
    """Objective function for scipy minimize."""
    n = N
    x = z[:n]
    y = z[n:2*n]
    r = z[2*n:]
    
    obj, gx, gy, gr = _compute_objective_and_grad(x, y, r, beta, lam)
    return obj, np.concatenate([gx, gy, gr])


def _grid_init():
    """Generate a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cr = 0.5 / side - 1e-4
            cells.append((cx, cy, cr))
    cells = cells[:N]
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.array([c[2] for c in cells])
    return x, y, r


def _random_init(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    # Place circles at random positions with small initial radii
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = rng.uniform(0.02, 0.06, N)
    return x, y, r


def _optimize_single(x0, y0, r0, beta, lam, max_iter=500):
    """Run L-BFGS-B optimization from a single start."""
    z0 = np.concatenate([x0, y0, r0])
    
    def fun(z):
        obj, grad = _objective_wrapper(z, beta, lam)
        return obj, grad
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    res = minimize(fun, z0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    x_opt = res.x[:N]
    y_opt = res.x[N:2*N]
    r_opt = res.x[2*N:]
    
    return x_opt, y_opt, r_opt, res.fun


def construct():
    start_time = time.time()
    
    best_x, best_y, best_r = None, None, None
    best_score = -np.inf
    
    # Start with grid initialization
    x0, y0, r0 = _grid_init()
    
    # Try multiple random starts and the grid start
    starts = [(x0, y0, r0, "grid")]
    
    # Add random starts
    for seed in range(20):
        xr, yr, rr = _random_init(seed)
        starts.append((xr, yr, rr, f"random_{seed}"))
    
    # Adaptive beta and lambda
    beta = 50.0
    lam = 50.0
    
    for idx, (xs, ys, rs, name) in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        
        try:
            # Run optimization with current beta and lam
            x_opt, y_opt, r_opt, score = _optimize_single(xs, ys, rs, beta, lam, max_iter=300)
            
            # Repair to ensure validity
            x_rep, y_rep, r_rep = _repair(x_opt.copy(), y_opt.copy(), r_opt.copy())
            
            # Validate
            if _is_valid(x_rep, y_rep, r_rep):
                val = np.sum(r_rep)
                if val > best_score:
                    best_score = val
                    best_x = x_rep.copy()
                    best_y = y_rep.copy()
                    best_r = r_rep.copy()
            
            # Also try the un-repaired version (it might be slightly better if valid)
            if _is_valid(x_opt, y_opt, r_opt):
                val = np.sum(r_opt)
                if val > best_score:
                    best_score = val
                    best_x = x_opt.copy()
                    best_y = y_opt.copy()
                    best_r = r_opt.copy()
                    
        except Exception:
            pass
        
        # Gradually increase penalty parameters for later iterations
        if idx % 5 == 0 and idx > 0:
            beta *= 1.5
            lam *= 1.5
    
    # If we still have time, do a final polish with higher penalties
    elapsed = time.time() - start_time
    if best_x is not None and elapsed < DEADLINE - SAFETY_MARGIN:
        beta_final = 200.0
        lam_final = 200.0
        try:
            x_opt, y_opt, r_opt, score = _optimize_single(
                best_x, best_y, best_r, beta_final, lam_final, max_iter=200)
            x_rep, y_rep, r_rep = _repair(x_opt.copy(), y_opt.copy(), r_opt.copy())
            if _is_valid(x_rep, y_rep, r_rep):
                val = np.sum(r_rep)
                if val > best_score:
                    best_score = val
                    best_x = x_rep.copy()
                    best_y = y_rep.copy()
                    best_r = r_rep.copy()
        except Exception:
            pass
    
    # Fallback to grid if nothing worked
    if best_x is None:
        best_x, best_y, best_r = _grid_init()
    
    return np.column_stack([best_x, best_y, best_r])
