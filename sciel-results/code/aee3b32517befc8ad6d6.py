"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Uses direct non-convex optimization with L-BFGS-B and smooth penalty functions
to avoid local minima traps of physics-based relaxation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 116.0  # safety margin before 150s limit
RNG = np.random.default_rng(20240517)

R_MIN = 1e-6
R_MAX = 0.5


def _is_valid(c, tol=1e-6):
    """Return True if packing is inside [0,1]^2 and circles are disjoint."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if (x.min() < -tol) or (x.max() > 1 + tol) or (y.min() < -tol) or (y.max() > 1 + tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf


def _clamp(c):
    """Clamp positions/radii to a feasible box (keeps circles inside square)."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


def _smooth_penalty_and_grad(params, T=0.01, a=50.0):
    """Compute smooth penalty and its gradient for circle packing constraints.
    
    Uses soft-min approximation: max(0, v) ≈ -T * log(1 + exp(-v/T))
    """
    c = params.reshape(-1, 3)
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    
    penalty = 0.0
    grad = np.zeros_like(params)
    
    # Wall violations: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    wall_terms = [
        (x - r, 0),        # left wall
        (1 - x - r, 1),    # right wall
        (y - r, 3),        # bottom wall
        (1 - y - r, 4),    # top wall
    ]
    
    for vals, idx in wall_terms:
        # Soft max(0, -vals) = -T * log(1 + exp(-vals/T))
        # d/dv = -exp(-vals/T) / (1 + exp(-vals/T)) = -1/(1+exp(vals/T))
        # But we want penalty for violation, so we use max(0, -vals)
        # Let u = -vals, then max(0, u) ≈ -T * log(1 + exp(-u/T))
        # d/du = -exp(-u/T)/(1+exp(-u/T)) = -1/(1+exp(u/T))
        # d/dv = -d/du * du/dv = -(-1/(1+exp(u/T))) * (-1) = -1/(1+exp(u/T))
        # where u = -vals
        u = -vals
        exp_u_T = np.exp(u / T)
        penalty += T * np.sum(np.log1p(exp_u_T))
        # Gradient with respect to vals: -1/(1+exp(u/T)) = -1/(1+exp(-vals/T))
        grad_wall = -1.0 / (1.0 + np.exp(-vals / T))
        
        # Map back to parameter indices
        if idx == 0:  # x - r: dx, dr
            grad[0::3] += grad_wall  # x component
            grad[2::3] -= grad_wall  # r component
        elif idx == 1:  # 1 - x - r: -dx, -dr
            grad[0::3] -= grad_wall
            grad[2::3] -= grad_wall
        elif idx == 3:  # y - r: dy, dr
            grad[1::3] += grad_wall
            grad[2::3] -= grad_wall
        elif idx == 4:  # 1 - y - r: -dy, -dr
            grad[1::3] -= grad_wall
            grad[2::3] -= grad_wall
    
    # Circle-circle separation: distance >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            s = r[i] + r[j]
            
            # Constraint: d - s >= 0
            # Penalty for violation: max(0, s - d) ≈ -T * log(1 + exp(-(d-s)/T))
            if d < s * 1.2:  # Only compute for near/overlapping pairs
                u = s - d  # violation amount
                exp_u_T = np.exp(u / T)
                penalty += T * np.log1p(exp_u_T)
                
                # Gradient: d/du * du/dparams
                # d/du of -T*log(1+exp(-u/T)) = -exp(-u/T)/(1+exp(-u/T)) = -1/(1+exp(u/T))
                # But we have max(0, u) ≈ T*log(1+exp(u/T))
                # d/du = exp(u/T)/(1+exp(u/T)) = 1/(1+exp(-u/T))
                d_du = 1.0 / (1.0 + np.exp(-u / T))
                
                # du/dparams where u = s - d = r_i + r_j - sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
                if d > 1e-9:
                    ux = dx / d
                    uy = dy / d
                else:
                    ux, uy = 1.0, 0.0
                
                # du/dx_i = -ux, du/dy_i = -uy
                # du/dx_j = ux, du/dy_j = uy
                # du/dr_i = 1, du/dr_j = 1
                
                idx_i = i * 3
                idx_j = j * 3
                
                grad[idx_i] += -d_du * ux      # x_i
                grad[idx_i + 1] += -d_du * uy  # y_i
                grad[idx_i + 2] += d_du        # r_i
                
                grad[idx_j] += d_du * ux       # x_j
                grad[idx_j + 1] += d_du * uy   # y_j
                grad[idx_j + 2] += d_du        # r_j
    
    return penalty, grad


def _objective_and_grad(params):
    """Objective: minimize -sum(r) + penalty_weight * penalty."""
    c = params.reshape(-1, 3)
    r = c[:, 2]
    
    penalty, pen_grad = _smooth_penalty_and_grad(params, T=0.01, a=50.0)
    
    obj = -np.sum(r) + 100.0 * penalty
    
    grad = -np.zeros_like(params)
    grad[2::3] += 1.0  # d(-sum(r))/dr = -1, so we negate for minimization
    grad += 100.0 * pen_grad
    
    return obj, grad


def _seed_grid():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _seed_random():
    """Random initialization with small radii."""
    x = RNG.uniform(0.08, 0.92, N)
    y = RNG.uniform(0.08, 0.92, N)
    r = RNG.uniform(0.02, 0.08, N)
    return np.column_stack([x, y, r])


def _seed_symmetric():
    """Symmetric corner initialization."""
    positions = []
    radii = []
    for cx, cy in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
        positions.append((cx, cy))
        radii.append(0.12)
    positions.append((0.5, 0.5))
    radii.append(0.15)
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r = RNG.uniform(0.05, 0.1)
        positions.append((x, y))
        radii.append(r)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    return np.column_stack([x, y, r])


def _lbfgs_optimize(c0, maxiter=300, maxfun=10000):
    """Run L-BFGS-B optimization from initial configuration."""
    if time.time() > DEADLINE - 5.0:
        return c0, _score(c0)
    
    x0 = c0.flatten()
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        c_opt = result.x.reshape(-1, 3)
        c_opt = _clamp(c_opt)
        
        if _is_valid(c_opt, tol=1e-5):
            return c_opt, _score(c_opt)
    except Exception:
        pass
    
    return c0, _score(c0)


def construct():
    t0 = time.time()
    
    # Generate diverse initial configurations
    seeds = []
    seeds.append(_seed_grid())
    seeds.append(_seed_symmetric())
    
    # Add random seeds
    for _ in range(8):
        seeds.append(_seed_random())
    
    best_c = None
    best_val = -np.inf
    
    # Optimize from each seed
    for i, c0 in enumerate(seeds):
        if time.time() > DEADLINE - 10.0:
            break
        
        # Use more iterations for earlier seeds
        maxiter = 400 if i < 3 else 200
        maxfun = 15000 if i < 3 else 8000
        
        cc, val = _lbfgs_optimize(c0, maxiter=maxiter, maxfun=maxfun)
        
        if val > best_val:
            best_val = val
            best_c = cc
    
    # Final polish on best result
    if best_c is not None and time.time() < DEADLINE - 5.0:
        cc, val = _lbfgs_optimize(best_c, maxiter=200, maxfun=8000)
        if val > best_val:
            best_val = val
            best_c = cc
    
    # Fallback
    if best_val <= 0 or best_c is None:
        return _seed_grid()
    
    return best_c
