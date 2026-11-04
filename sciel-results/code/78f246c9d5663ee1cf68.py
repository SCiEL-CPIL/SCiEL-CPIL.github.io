"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _objective_and_grad(theta, beta):
    """
    theta: (3N,) vector of [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    Returns (objective, gradient) for:
      maximize sum(r_i) subject to:
        x_i - r_i >= 0, y_i - r_i >= 0
        1 - x_i - r_i >= 0, 1 - y_i - r_i >= 0
        ||p_i - p_j|| - r_i - r_j >= 0 for i < j
    
    We minimize -sum(r_i) + penalty for violations.
    Using a smooth penalty: for each constraint g_k >= 0, add max(0, -g_k)^2 * beta
    """
    n = N
    x = theta[0::3]
    y = theta[1::3]
    r = theta[2::3]
    
    obj = -np.sum(r)
    grad = np.zeros_like(theta)
    grad[2::3] = -1.0
    
    # Boundary constraints: x - r >= 0, y - r >= 0, 1-x-r >= 0, 1-y-r >= 0
    for i in range(n):
        # x_i - r_i >= 0
        g = x[i] - r[i]
        if g < 0:
            viol = -g
            obj += beta * viol * viol
            # d/dx_i: 2*beta*viol * 1
            # d/d(-x_i + r_i): the gradient of -g w.r.t x_i is -1, w.r.t r_i is 1
            # d(obj)/dx_i += 2*beta*viol * (-1) * (-1) ... let me be careful
            # g = x_i - r_i, violation v = -g = r_i - x_i
            # penalty = beta * v^2
            # d(penalty)/dx_i = 2*beta*v * dv/dx_i = 2*beta*v * (-1) = -2*beta*v
            # d(penalty)/dr_i = 2*beta*v * dv/dr_i = 2*beta*v * (1) = 2*beta*v
            grad[3*i] += -2 * beta * viol
            grad[3*i + 2] += 2 * beta * viol
        
        # y_i - r_i >= 0
        g = y[i] - r[i]
        if g < 0:
            viol = -g
            obj += beta * viol * viol
            grad[3*i + 1] += -2 * beta * viol
            grad[3*i + 2] += 2 * beta * viol
        
        # 1 - x_i - r_i >= 0
        g = 1.0 - x[i] - r[i]
        if g < 0:
            viol = -g
            obj += beta * viol * viol
            grad[3*i] += 2 * beta * viol
            grad[3*i + 2] += 2 * beta * viol
        
        # 1 - y_i - r_i >= 0
        g = 1.0 - y[i] - r[i]
        if g < 0:
            viol = -g
            obj += beta * viol * viol
            grad[3*i + 1] += 2 * beta * viol
            grad[3*i + 2] += 2 * beta * viol
    
    # Pairwise constraints: ||p_i - p_j|| - r_i - r_j >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            g = dist - r[i] - r[j]
            if g < 0:
                viol = -g
                obj += beta * viol * viol
                # d(penalty)/dx_i = 2*beta*viol * d(-g)/dx_i
                # -g = r_i + r_j - dist
                # d(-g)/dx_i = -dx/dist (since d(dist)/dx_i = dx/dist)
                # d(-g)/dx_j = dx/dist
                # d(-g)/dr_i = 1, d(-g)/dr_j = 1
                if dist > 1e-12:
                    grad[3*i] += 2 * beta * viol * (-dx / dist)
                    grad[3*i + 1] += 2 * beta * viol * (-dy / dist)
                    grad[3*j] += 2 * beta * viol * (dx / dist)
                    grad[3*j + 1] += 2 * beta * viol * (dy / dist)
                grad[3*i + 2] += 2 * beta * viol
                grad[3*j + 2] += 2 * beta * viol
    
    return obj, grad


def _is_valid(theta, tol=1e-9):
    """Check if all constraints are satisfied."""
    n = N
    x = theta[0::3]
    y = theta[1::3]
    r = theta[2::3]
    
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(1 - x - r < -tol) or np.any(1 - y - r < -tol):
        return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _clamp_and_fix(theta):
    """Clamp radii to be non-negative and positions to be feasible-ish."""
    theta = theta.copy()
    r = theta[2::3]
    # Ensure radii are at least a small positive
    r = np.maximum(r, 1e-6)
    theta[2::3] = r
    
    # Clamp x, y based on radii
    for i in range(N):
        ri = r[i]
        theta[3*i] = np.clip(theta[3*i], ri, 1.0 - ri)
        theta[3*i + 1] = np.clip(theta[3*i + 1], ri, 1.0 - ri)
    return theta


def _evaluate(theta):
    """Return sum of radii if valid, else -inf."""
    theta = _clamp_and_fix(theta)
    if _is_valid(theta):
        return np.sum(theta[2::3])
    return -np.inf


def _grid_init():
    """Grid initialization: 5x5 grid, take first 26."""
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side
            cells.append((cx, cy, r))
    # Take first 26
    cells = cells[:N]
    theta = np.array([c for cell in cells for c in cell]).flatten()
    return theta


def _random_init():
    """Random initialization with small circles."""
    np.random.seed(42)
    theta = np.zeros(3 * N)
    for i in range(N):
        r = 0.05
        x = r + np.random.random() * (1.0 - 2 * r)
        y = r + np.random.random() * (1.0 - 2 * r)
        theta[3*i] = x
        theta[3*i + 1] = y
        theta[3*i + 2] = r
    return theta


def _lbfgs_optimize(theta0, beta, maxiter=2000, time_start=None, time_limit=None):
    """Run L-BFGS-B with smooth penalty objective."""
    
    def fun_grad(theta):
        obj, grad = _objective_and_grad(theta, beta)
        return obj, grad
    
    result = minimize(fun_grad, theta0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                                         'ftol': 1e-15, 'gtol': 1e-10})
    return result.x


def _polish(theta, time_start, time_limit):
    """High-precision polish with increasing beta."""
    best = theta.copy()
    best_val = _evaluate(best)
    
    betas = [100, 1000, 10000, 100000, 1000000]
    for beta in betas:
        if time.time() - time_start > time_limit - 5:
            break
        try:
            res = _lbfgs_optimize(best, beta, maxiter=1000, time_start=time_start, time_limit=time_limit)
            res = _clamp_and_fix(res)
            val = _evaluate(res)
            if val > best_val:
                best = res
                best_val = val
        except Exception:
            break
    
    return best, best_val


def construct():
    time_start = time.time()
    
    # Start with grid init
    theta0 = _grid_init()
    
    # Multi-stage optimization with increasing penalty
    best = theta0.copy()
    best_val = _evaluate(best)
    
    betas = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    
    for beta in betas:
        if time.time() - time_start > TIME_LIMIT - 10:
            break
        try:
            res = _lbfgs_optimize(best, beta, maxiter=1500)
            res = _clamp_and_fix(res)
            val = _evaluate(res)
            if val > best_val:
                best = res
                best_val = val
        except Exception:
            continue
    
    # Polish phase
    if time.time() - time_start < TIME_LIMIT - 5:
        best, best_val = _polish(best, time_start, TIME_LIMIT)
    
    # Try a few random restarts if time permits
    if time.time() - time_start < TIME_LIMIT - 15:
        for seed in range(5):
            if time.time() - time_start > TIME_LIMIT - 10:
                break
            np.random.seed(seed + 100)
            theta_r = _random_init()
            # Quick optimization
            for beta in [100, 1000, 10000]:
                if time.time() - time_start > TIME_LIMIT - 8:
                    break
                try:
                    res = _lbfgs_optimize(theta_r, beta, maxiter=500)
                    res = _clamp_and_fix(res)
                    val = _evaluate(res)
                    if val > best_val:
                        best = res
                        best_val = val
                except Exception:
                    break
    
    # Final polish
    if time.time() - time_start < TIME_LIMIT - 3:
        best, best_val = _polish(best, time_start, TIME_LIMIT)
    
    # Ensure final validity
    best = _clamp_and_fix(best)
    if not _is_valid(best):
        # Fallback to grid
        best = _grid_init()
    
    # Return as (n, 3) array
    result = np.zeros((N, 3))
    result[:, 0] = best[0::3]
    result[:, 1] = best[1::3]
    result[:, 2] = best[2::3]
    
    return result
