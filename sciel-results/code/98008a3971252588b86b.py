"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Improved version using vectorized force-based relaxation with radius growth,
multi-start with diverse initializations, and L-BFGS-B polishing.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 115.0  # Safety margin
RNG = np.random.default_rng(20240517)

R_MIN = 1e-6
R_MAX = 0.5

def _is_valid(c, tol=1e-6):
    """Return True if packing is inside [0,1]^2 and circles are disjoint."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    # Vectorized pairwise distance check
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    s = r[:, None] + r[None, :]
    # Check upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[mask] < s[mask] - tol):
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

def _vectorized_forces(c):
    """Compute all forces in a vectorized manner."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)

    # Wall forces
    fx -= 500.0 * np.clip(r - x, 0, None)
    fx += 500.0 * np.clip(x + r - 1, 0, None)
    fy -= 500.0 * np.clip(r - y, 0, None)
    fy += 500.0 * np.clip(y + r - 1, 0, None)
    fr += 500.0 * (np.clip(r - x, 0, None) + np.clip(x + r - 1, 0, None) + 
                   np.clip(r - y, 0, None) + np.clip(y + r - 1, 0, None))

    # Separation forces (fully vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)  # Avoid division by zero
    s = r[:, None] + r[None, :]
    overlap = np.clip(s - d, 0, None)
    
    # Only consider pairs with significant overlap
    active = overlap > 1e-8
    f_mag = 40.0 * overlap * active
    
    # Unit vectors (safe division since d is filled with 1.0 on diagonal)
    ux = dx / d
    uy = dy / d
    
    # Sum forces: for each i, sum over j of f[i,j] * u[i,j]
    # f[i,j] is the force magnitude, u[i,j] points from j to i
    fx += np.sum(f_mag * ux, axis=1)
    fy += np.sum(f_mag * uy, axis=1)
    
    # Radius growth from separation (each overlapping pair contributes to both circles)
    fr += 0.5 * np.sum(f_mag, axis=1)

    # Maximize radii
    fr += 10.0

    return fx, fy, fr

def _relax_and_grow(c0, max_iters=2000, grow_rate=0.0005):
    """Iteratively relax overlaps and grow radii."""
    c = c0.copy()
    best = _score(c)
    best_c = c.copy()
    step = 0.03
    no_improve = 0
    max_no_improve = 80
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 3.0:
            break
        
        c = _clamp(c)
        fx, fy, fr = _vectorized_forces(c)
        
        fmag = np.hypot(fx, fy)
        adaptive_step = step / (1.0 + 0.01 * fmag.max())
        
        x = c[:, 0] + adaptive_step * fx
        y = c[:, 1] + adaptive_step * fy
        r = c[:, 2] + adaptive_step * fr + grow_rate
        
        c = np.column_stack([x, y, r])
        c = _clamp(c)
        
        val = _score(c)
        if val > best:
            best = val
            best_c = c.copy()
            no_improve = 0
            step = min(step * 1.05, 0.1)
        else:
            no_improve += 1
            step *= 0.95
            if it > 5:
                c = best_c.copy()
            if step < 1e-6 or no_improve > max_no_improve:
                break
    
    return best_c, best

def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

def _seed_hex():
    """Hexagonal packing initialization."""
    positions = []
    r = 0.18
    dx = 2 * r
    dy = r * np.sqrt(3)
    for row in range(5):
        for col in range(5):
            x = r + col * dx + (row % 2) * dx / 2
            y = r + row * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append((x, y, r))
    # Fill remaining with smaller circles
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r_small = RNG.uniform(0.05, 0.1)
        positions.append((x, y, r_small))
    return np.array(positions[:N])

def _seed_random():
    x = RNG.uniform(0.08, 0.92, N)
    y = RNG.uniform(0.08, 0.92, N)
    r = RNG.uniform(0.03, 0.1, N)
    return np.column_stack([x, y, r])

def _seed_corner_heavy():
    """Initialize with larger circles in corners, smaller in center."""
    positions = []
    radii = []
    # Corners: larger circles
    for cx, cy in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
        positions.append((cx, cy))
        radii.append(0.13)
    # Edges: medium circles
    for cx, cy in [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]:
        positions.append((cx, cy))
        radii.append(0.1)
    # Center and rest: smaller
    positions.append((0.5, 0.5))
    radii.append(0.12)
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r = RNG.uniform(0.04, 0.08)
        positions.append((x, y))
        radii.append(r)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    return np.column_stack([x, y, r])

def _smooth_max_0(v, T):
    """Smooth approximation of max(0, v) using log-sum-exp."""
    a = v / T
    return T * np.where(a > 0, a + np.log1p(np.exp(-a)), np.log1p(np.exp(a)))

def _lbfgs_polish(c0, T_start=0.03, max_iter=300):
    """L-BFGS-B polish with cooling temperature."""
    if time.time() > DEADLINE - 8.0:
        return c0, _score(c0)
    
    x0 = c0.flatten()
    best_c = c0.copy()
    best_val = _score(c0)
    
    for stage in range(3):
        T = T_start * (0.3 ** stage)
        
        def _objective(params):
            c = params.reshape(-1, 3)
            x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
            
            pen_wall = (
                _smooth_max_0(r - x, T) + _smooth_max_0(x + r - 1, T) +
                _smooth_max_0(r - y, T) + _smooth_max_0(y + r - 1, T)
            )
            pen_wall = np.sum(pen_wall)
            
            pen_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]; dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    s = r[i] + r[j]
                    if d < s * 1.2:
                        pen_overlap += _smooth_max_0(s - d, T)
            
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
            
            x0 = c_polished.flatten()
            
        except Exception:
            pass
        
        if time.time() > DEADLINE - 5.0:
            break
            
    return best_c, best_val

def _radial_expansion(c0, max_iters=1000, growth_rate=0.0008):
    """Try to expand all radii uniformly while maintaining feasibility."""
    c = c0.copy()
    best = _score(c)
    best_c = c.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5.0:
            break
        
        # Try to grow all radii slightly
        c[:, 2] += growth_rate
        c = _clamp(c)
        
        # Relax to fix any overlaps created by growth
        for _ in range(20):
            fx, fy, fr = _vectorized_forces(c)
            step = 0.02
            x = c[:, 0] + step * fx
            y = c[:, 1] + step * fy
            r = c[:, 2] + step * fr
            c = np.column_stack([x, y, r])
            c = _clamp(c)
        
        val = _score(c)
        if val > best:
            best = val
            best_c = c.copy()
        else:
            # Can't grow further, revert
            break
    
    return best_c, best

def construct():
    t0 = time.time()
    
    # Try multiple diverse initializations
    initializers = [_seed_grid, _seed_hex, _seed_corner_heavy, _seed_random]
    best_c = None
    best_val = -np.inf
    
    # Phase 1: Relax and grow from each initialization
    for init_func in initializers:
        if time.time() > DEADLINE - 35.0:
            break
        
        c0 = init_func()
        cc, val = _relax_and_grow(c0, max_iters=1200, grow_rate=0.0004)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 2: Additional random starts if time permits
    n_random_extra = 12
    for i in range(n_random_extra):
        if time.time() > DEADLINE - 30.0:
            break
        
        c0 = _seed_random()
        cc, val = _relax_and_grow(c0, max_iters=600, grow_rate=0.0003)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 3: Radial expansion of best result
    if best_c is not None and time.time() < DEADLINE - 18.0:
        cc, val = _radial_expansion(best_c, max_iters=500, growth_rate=0.0006)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 4: Polish the best result
    if best_c is not None and time.time() < DEADLINE - 12.0:
        cc, val = _lbfgs_polish(best_c)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Final safety: ensure valid output
    if best_c is None or best_val <= 0:
        # Fallback to a simple valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_c = np.array(cells[:N])
    
    return best_c
