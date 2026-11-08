"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Improved version using smooth log-sum-exp penalties for overlaps and walls,
optimized via L-BFGS-B with adaptive temperature cooling.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 116.0  # Safety margin
RNG = np.random.default_rng(20240517)

R_MIN = 1e-6
R_MAX = 0.5

# Force parameters for PBD phase
K_WALL = 500.0
K_SEP = 40.0
K_R = 10.0
STEP = 0.03

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

def _pbd_step(c):
    """One force-based step with adaptive step size and clamping."""
    c = _clamp(c)
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)

    # Wall forces
    fx -= K_WALL * np.clip(r - x, 0, None)
    fx += K_WALL * np.clip(x + r - 1, 0, None)
    fy -= K_WALL * np.clip(r - y, 0, None)
    fy += K_WALL * np.clip(y + r - 1, 0, None)
    fr += K_WALL * (np.clip(r - x, 0, None) + np.clip(x + r - 1, 0, None) + 
                    np.clip(r - y, 0, None) + np.clip(y + r - 1, 0, None))

    # Separation forces
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            s = r[i] + r[j]
            if d < s * 1.05:
                if d < 1e-9:
                    dx = RNG.standard_normal(); dy = RNG.standard_normal()
                    d = np.hypot(dx, dy)
                ux = dx / d; uy = dy / d
                overlap = s - d
                if overlap > 0:
                    f = K_SEP * overlap
                    fx[i] += f * ux; fy[i] += f * uy
                    fx[j] -= f * ux; fy[j] -= f * uy
                    fr[i] += f * 0.5; fr[j] += f * 0.5

    # Maximize radii
    fr += K_R

    fmag = np.hypot(fx, fy)
    step = STEP * 0.5 / (1.0 + 0.02 * fmag.max())

    x = c[:, 0] + step * fx
    y = c[:, 1] + step * fy
    r = c[:, 2] + step * fr

    return np.column_stack([x, y, r])

def _optimize_pbd(c0):
    """Force-based optimizer with adaptive step size and time-aware stopping."""
    c = c0.copy()
    best = _score(c)
    best_c = c.copy()
    step = STEP
    it = 0
    no_improve_count = 0
    max_no_improve = 200
    
    while it < 8000:
        c = _pbd_step(c)
        val = _score(c)
        if val > best:
            best = val
            best_c = c.copy()
            step = min(step * 1.12, 0.15)
            no_improve_count = 0
        else:
            step *= 0.85
            no_improve_count += 1
            if it > 10:
                c = best_c.copy()
            if step < 1e-5 or no_improve_count > max_no_improve:
                break
        it += 1
        if time.time() > DEADLINE - 2.0:
            break
    return best_c, best

def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

def _seed_random():
    x = RNG.uniform(0.08, 0.92, N)
    y = RNG.uniform(0.08, 0.92, N)
    r = RNG.uniform(0.02, 0.08, N)
    return np.column_stack([x, y, r])

def _seed_symmetric():
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

def _smooth_max_0(v, T):
    """Smooth approximation of max(0, v) using log-sum-exp."""
    # log(1 + exp(v/T)) * T
    # To avoid overflow, use:
    # if v/T > 0: v + T*log(1+exp(-v/T))
    # else: T*log(1+exp(v/T))
    a = v / T
    return T * np.where(a > 0, a + np.log1p(np.exp(-a)), np.log1p(np.exp(a)))

def _smooth_max_v(v, T):
    """Smooth approximation of max(v, 0) is same as _smooth_max_0(v, T)."""
    return _smooth_max_0(v, T)

def _lbfgs_polish(c0, T_start=0.05, T_end=0.001, max_iter=500):
    """L-BFGS-B polish with cooling temperature."""
    if time.time() > DEADLINE - 8.0:
        return c0, _score(c0)
    
    x0 = c0.flatten()
    n_params = x0.size
    current_T = T_start
    best_c = c0.copy()
    best_val = _score(c0)
    
    # We will run L-BFGS-B multiple times with decreasing T
    # This helps escape local minima of the penalty function
    
    for stage in range(3):
        T = current_T * (0.2 ** stage)
        
        def _objective(params):
            c = params.reshape(-1, 3)
            x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
            
            # Smooth penalties
            # Walls: max(0, r-x), max(0, x+r-1), etc.
            pen_wall = (
                _smooth_max_0(r - x, T) + _smooth_max_0(x + r - 1, T) +
                _smooth_max_0(r - y, T) + _smooth_max_0(y + r - 1, T)
            )
            pen_wall = np.sum(pen_wall)
            
            # Overlaps: max(0, r_i + r_j - d_ij)
            pen_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]; dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    s = r[i] + r[j]
                    if d < s * 1.1: # Optimization: only compute if close
                        pen_overlap += _smooth_max_0(s - d, T)
            
            # Objective: Minimize -sum(r) + weight * penalties
            weight = 100.0
            return -np.sum(r) + weight * (pen_wall + pen_overlap)
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_polished = result.x.reshape(-1, 3)
            c_polished = _clamp(c_polished)
            
            if _is_valid(c_polished, tol=1e-6):
                val = _score(c_polished)
                if val > best_val:
                    best_val = val
                    best_c = c_polished.copy()
            
            # Update x0 for next stage
            x0 = c_polished.flatten()
            
        except Exception:
            pass
        
        if time.time() > DEADLINE - 5.0:
            break
            
    return best_c, best_val

def construct():
    t0 = time.time()
    
    # Start with grid initialization
    c_grid = _seed_grid()
    best_c, best_val = _optimize_pbd(c_grid)
    
    # Try diverse random starts
    n_random = 25
    for i in range(n_random):
        if time.time() > DEADLINE - 20.0:
            break
        
        if i % 5 == 0:
            c0 = _seed_symmetric()
        else:
            c0 = _seed_random()
        
        cc, val = _optimize_pbd(c0)
        if val > best_val:
            best_val, best_c = val, cc
    
    # Polish the best result
    if time.time() < DEADLINE - 10.0:
        cc, val = _lbfgs_polish(best_c)
        if val > best_val:
            best_val, best_c = val, cc
    
    # If we have time, try polishing other good candidates?
    # For now, just ensure the best is polished.
    
    if best_val <= 0:
        return c_grid
    
    return best_c
