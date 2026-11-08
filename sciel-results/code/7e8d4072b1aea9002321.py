"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 118.0          # safety margin before 150 s wall-clock kill
RNG = np.random.default_rng(20240517)

# --- force parameters (tuned for stability / speed) ---
K_WALL = 500.0
K_SEP = 40.0
K_R = 10.0
EPS = 1e-9
STEP = 0.03
MAXIT = 4000
R_MIN = 1e-6
R_MAX = 0.5


def _objective(c):
    """Sum of radii if the packing is valid, else -inf."""
    if c is None:
        return -np.inf
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if (x.min() < -EPS) or (x.max() > 1 + EPS) or (y.min() < -EPS) or (y.max() > 1 + EPS):
        return -np.inf
    if r.min() < R_MIN:
        return -np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - EPS:
                return -np.inf
    return float(np.sum(r))


def _forces(c):
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)

    # wall forces
    fx += K_WALL * np.where(x < r, 1.0, 0.0) - K_WALL * np.where(x > 1 - r, 1.0, 0.0)
    fy += K_WALL * np.where(y < r, 1.0, 0.0) - K_WALL * np.where(y > 1 - r, 1.0, 0.0)
    fr += K_WALL * (np.where(x < r, 1.0, 0.0) + np.where(x > 1 - r, 1.0, 0.0)
                    + np.where(y < r, 1.0, 0.0) + np.where(y > 1 - r, 1.0, 0.0))

    # separation forces
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            s = r[i] + r[j]
            if d < s:
                if d < 1e-9:
                    dx = RNG.standard_normal(); dy = RNG.standard_normal()
                    d = np.hypot(dx, dy)
                ux = dx / d; uy = dy / d
                f = K_SEP * (s - d)
                fx[i] += f * ux; fy[i] += f * uy
                fx[j] -= f * ux; fy[j] -= f * uy
                fr[i] += f; fr[j] += f

    # maximize radii
    fr += K_R

    # clamp radii
    r = np.clip(r, R_MIN, R_MAX)
    return fx, fy, fr, r


def _optimize(c0):
    c = c0.copy()
    best = _objective(c)
    step = STEP
    for it in range(MAXIT):
        fx, fy, fr, r = _forces(c)
        c[:, 0] += step * fx
        c[:, 1] += step * fy
        c[:, 2] += step * fr
        c[:, 0] = np.clip(c[:, 0], 1e-6, 1 - 1e-6)
        c[:, 1] = np.clip(c[:, 1], 1e-6, 1 - 1e-6)
        c[:, 2] = np.clip(c[:, 2], R_MIN, R_MAX)
        val = _objective(c)
        if val > best:
            best = val
            step = min(step * 1.12, 0.12)
        else:
            step *= 0.82
            if it > 5:
                c = c0.copy()
            if step < 1e-4:
                break
        if time.time() > DEADLINE - 1.0:
            break
    return c, best


def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _smooth_penalty_objective(flat):
    """
    Smooth surrogate objective: -sum(r) + penalty for violations.
    Uses soft-min (log-sum-exp style) for smooth constraint handling.
    """
    c = flat.reshape(N, 3)
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    
    # Clip to valid range for numerical stability
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    r = np.clip(r, R_MIN, R_MAX)
    
    obj = -np.sum(r)
    
    # Wall violations: circle must stay inside [0,1]^2
    # x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    wall_violations = np.concatenate([
        np.maximum(0, r - x),
        np.maximum(0, r - (1 - x)),
        np.maximum(0, r - y),
        np.maximum(0, r - (1 - y))
    ])
    # Smooth sum of violations using log-sum-exp
    if np.any(wall_violations > 0):
        log_sum_exp = np.log(np.sum(np.exp(wall_violations / 0.01))) * 0.01
        obj += 100.0 * log_sum_exp
    else:
        obj += 100.0 * np.sum(wall_violations)
    
    # Circle-circle separation violations
    sep_violations = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            s = r[i] + r[j]
            violation = max(0, s - d)
            sep_violations.append(violation)
    
    if sep_violations:
        sep_arr = np.array(sep_violations)
        if np.any(sep_arr > 0):
            log_sum_exp = np.log(np.sum(np.exp(sep_arr / 0.01))) * 0.01
            obj += 50.0 * log_sum_exp
        else:
            obj += 50.0 * np.sum(sep_arr)
    
    return obj


def _smooth_penalty_gradient(flat):
    """
    Numerical gradient of the smooth penalty objective.
    """
    eps = 1e-7
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        f_plus = flat.copy()
        f_plus[i] += eps
        f_minus = flat.copy()
        f_minus[i] -= eps
        grad[i] = (_smooth_penalty_objective(f_plus) - _smooth_penalty_objective(f_minus)) / (2 * eps)
    return grad


def _lbfgs_polish(c0):
    """
    Use L-BFGS-B to polish a packing by minimizing the smooth penalty objective.
    """
    flat = c0.flatten()
    
    # Define bounds
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 1 - 1e-6))  # x
        bounds.append((1e-6, 1 - 1e-6))  # y
        bounds.append((R_MIN, R_MAX))     # r
    
    # Use numerical gradient for reliability
    try:
        result = minimize(
            _smooth_penalty_objective,
            flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _smooth_penalty_objective(flat):
            c_new = result.x.reshape(N, 3)
            # Verify and clean up
            c_new[:, 0] = np.clip(c_new[:, 0], 1e-6, 1 - 1e-6)
            c_new[:, 1] = np.clip(c_new[:, 1], 1e-6, 1 - 1e-6)
            c_new[:, 2] = np.clip(c_new[:, 2], R_MIN, R_MAX)
            
            # Only accept if it's valid and better
            val_new = _objective(c_new)
            val_old = _objective(c0)
            if val_new > val_old:
                return c_new, val_new
    except Exception:
        pass
    
    return c0, _objective(c0)


def construct():
    t0 = time.time()
    c_grid = _seed_grid()
    best_c, best_val = _optimize(c_grid)
    
    # Apply L-BFGS polish to the grid-based result
    if time.time() < DEADLINE - 5.0:
        cc, val = _lbfgs_polish(best_c)
        if val > best_val:
            best_val, best_c = val, cc

    # diverse random starts
    for _ in range(60):
        if time.time() > DEADLINE - 5.0:
            break
        x = RNG.uniform(0.08, 0.92, N)
        y = RNG.uniform(0.08, 0.92, N)
        r = RNG.uniform(0.02, 0.08, N)
        c0 = np.column_stack([x, y, r])
        cc, val = _optimize(c0)
        if val > best_val:
            best_val, best_c = val, cc
            # Polish with L-BFGS if we found something good
            if time.time() < DEADLINE - 5.0:
                cc2, val2 = _lbfgs_polish(cc)
                if val2 > best_val:
                    best_val, best_c = val2, cc2

    # final polish on the best packing
    if time.time() < DEADLINE - 1.0:
        cc, val = _optimize(best_c)
        if val > best_val:
            best_val, best_c = val, cc
    
    # Final L-BFGS polish
    if time.time() < DEADLINE - 1.0:
        cc, val = _lbfgs_polish(best_c)
        if val > best_val:
            best_val, best_c = val, cc

    # fallback: seed grid is always valid
    if best_val <= 0:
        return c_grid
    return best_c
