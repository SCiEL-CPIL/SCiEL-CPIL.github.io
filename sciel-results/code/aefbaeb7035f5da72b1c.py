"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise(i, j, x, y, r):
    """Distance between circle i and j."""
    return np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

def _all_pairs(n):
    idx = []
    for i in range(n):
        for j in range(i+1, n):
            idx.append((i, j))
    return idx

PAIRS = _all_pairs(N)
NP = len(PAIRS)

def _compute_score(x, y, r):
    """Sum of radii."""
    return np.sum(r)

def _check_validity(x, y, r, tol=1e-9):
    """Check if configuration is valid."""
    n = len(r)
    # Check boundaries
    for i in range(n):
        if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol:
            return False
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > 1 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1 + tol:
            return False
    # Check pairwise
    for (i, j) in PAIRS:
        d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
        if d < r[i] + r[j] - tol:
            return False
    return True

def _make_valid(x, y, r, tol=1e-7):
    """Ensure validity by shrinking radii and adjusting positions."""
    n = len(r)
    # Clip positions and radii to be within bounds
    for i in range(n):
        r[i] = max(0, min(r[i], min(x[i], 1 - x[i], y[i], 1 - y[i])))
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    # Resolve overlaps by shrinking smaller circles
    for _ in range(100):
        overlap = False
        for (i, j) in PAIRS:
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            req = r[i] + r[j]
            if d < req - tol:
                overlap = True
                # Shrink the smaller circle
                if r[i] <= r[j]:
                    shrink = (req - d) / 2 + tol
                    r[i] = max(0, r[i] - shrink)
                else:
                    shrink = (req - d) / 2 + tol
                    r[j] = max(0, r[j] - shrink)
                # Re-clip to bounds
                r[i] = max(0, min(r[i], min(x[i], 1 - x[i], y[i], 1 - y[i])))
                r[j] = max(0, min(r[j], min(x[j], 1 - x[j], y[j], 1 - y[j])))
                x[i] = np.clip(x[i], r[i], 1 - r[i])
                y[i] = np.clip(y[i], r[i], 1 - r[i])
                x[j] = np.clip(x[j], r[j], 1 - r[j])
                y[j] = np.clip(y[j], r[j], 1 - r[j])
        if not overlap:
            break
    return x, y, r

def _objective(params, x0, y0, r0):
    """Penalized objective: minimize -sum(r) + penalty for violations."""
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Boundary violations
    pen = 0.0
    for i in range(n):
        # x bounds
        if x[i] < r[i]:
            pen += (x[i] - r[i])**2
        if x[i] > 1 - r[i]:
            pen += (x[i] - (1 - r[i]))**2
        if y[i] < r[i]:
            pen += (y[i] - r[i])**2
        if y[i] > 1 - r[i]:
            pen += (y[i] - (1 - r[i]))**2
        # Radius lower bound
        if r[i] < 0:
            pen += r[i]**2
    
    # Pairwise violations
    for (i, j) in PAIRS:
        d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
        req = r[i] + r[j]
        if d < req:
            pen += (d - req)**2
    
    return -np.sum(r) + 100 * pen

def _constraints(params):
    """Hard constraints for SLSQP."""
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    cons = []
    # Boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[n+i] - p[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i]})
    
    # Pairwise constraints
    for (i, j) in PAIRS:
        cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
            np.sqrt((p[i]-p[j])**2 + (p[n+i]-p[n+j])**2) - p[2*n+i] - p[2*n+j]})
    
    return cons

def _optimize_from_initial(x0, y0, r0, maxiter=500, ftol=1e-12):
    """Run L-BFGS-B then SLSQP polish."""
    n = N
    params0 = np.concatenate([x0, y0, r0])
    
    # Phase 1: L-BFGS-B with penalty
    result1 = minimize(
        _objective, params0, args=(x0, y0, r0),
        method='L-BFGS-B',
        bounds=[(0, 1)] * (2*n) + [(0, 0.5)] * n,
        options={'maxiter': 200, 'ftol': 1e-14}
    )
    params1 = result1.x
    
    # Ensure validity
    x1 = params1[:n].copy()
    y1 = params1[n:2*n].copy()
    r1 = params1[2*n:3*n].copy()
    x1, y1, r1 = _make_valid(x1, y1, r1)
    score1 = _compute_score(x1, y1, r1)
    
    # Phase 2: SLSQP with hard constraints
    params2_init = np.concatenate([x1, y1, r1])
    try:
        result2 = minimize(
            lambda p: -np.sum(p[2*n:3*n]),
            params2_init,
            method='SLSQP',
            constraints=_constraints(params2_init),
            bounds=[(0, 1)] * (2*n) + [(0, 0.5)] * n,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        if result2.success:
            x2 = result2.x[:n].copy()
            y2 = result2.x[n:2*n].copy()
            r2 = result2.x[2*n:3*n].copy()
            x2, y2, r2 = _make_valid(x2, y2, r2)
            score2 = _compute_score(x2, y2, r2)
            if score2 >= score1 - 1e-9:
                return x2, y2, r2, score2
    except Exception:
        pass
    
    return x1, y1, r1, score1

def _greedy_init(rng):
    """Greedy circle placement with randomization."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Start with a few random circles
    for i in range(N):
        # Try to place circle with random radius
        max_attempts = 200
        placed = False
        for _ in range(max_attempts):
            # Random radius between 0.02 and 0.3
            r_try = rng.uniform(0.02, 0.25)
            x_try = rng.uniform(r_try, 1 - r_try)
            y_try = rng.uniform(r_try, 1 - r_try)
            
            # Check against existing circles
            valid = True
            for j in range(i):
                d = np.sqrt((x_try - x[j])**2 + (y_try - y[j])**2)
                if d < r_try + r[j] - 1e-9:
                    valid = False
                    break
            
            # Check boundaries
            if x_try < r_try or x_try > 1 - r_try or y_try < r_try or y_try > 1 - r_try:
                valid = False
            
            if valid:
                x[i] = x_try
                y[i] = y_try
                r[i] = r_try
                placed = True
                break
        
        if not placed:
            # Fallback: small circle in a corner
            r[i] = 0.01
            x[i] = r[i] + 0.001
            y[i] = r[i] + 0.001
            # Try to find a valid spot
            for _ in range(100):
                x_try = rng.uniform(r[i], 1 - r[i])
                y_try = rng.uniform(r[i], 1 - r[i])
                valid = True
                for j in range(i):
                    d = np.sqrt((x_try - x[j])**2 + (y_try - y[j])**2)
                    if d < r_try + r[j] - 1e-9:
                        valid = False
                        break
                if valid:
                    x[i] = x_try
                    y[i] = y_try
                    break
    
    return x, y, r

def _grid_init():
    """Grid-based initialization."""
    side = int(np.ceil(np.sqrt(N)))
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        col = i % side
        row = i // side
        x[i] = (col + 0.5) / side
        y[i] = (row + 0.5) / side
        r[i] = 0.5 / side - 0.005
    
    return x, y, r

def _random_init(rng):
    """Fully random initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        r[i] = rng.uniform(0.02, 0.15)
        x[i] = rng.uniform(r[i], 1 - r[i])
        y[i] = rng.uniform(r[i], 1 - r[i])
    
    # Make valid
    x, y, r = _make_valid(x, y, r)
    return x, y, r

def construct():
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_x = None
    best_y = None
    best_r = None
    
    # Generate multiple initializations
    inits = []
    
    # Grid init
    xg, yg, rg = _grid_init()
    inits.append((xg, yg, rg))
    
    # Greedy inits
    for _ in range(3):
        xg, yg, rg = _greedy_init(rng)
        inits.append((xg, yg, rg))
    
    # Random inits
    for _ in range(3):
        xr, yr, rr = _random_init(rng)
        inits.append((xr, yr, rr))
    
    # Try each initialization
    for x0, y0, r0 in inits:
        if time.time() > deadline - 20:
            break
        
        try:
            x_opt, y_opt, r_opt, score = _optimize_from_initial(x0, y0, r0)
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            continue
    
    # Final high-precision polish on best solution
    if best_x is not None and time.time() < deadline - 5:
        try:
            x_pol, y_pol, r_pol, score_pol = _optimize_from_initial(
                best_x, best_y, best_r, maxiter=1000, ftol=1e-15
            )
            if score_pol > best_score:
                best_score = score_pol
                best_x = x_pol.copy()
                best_y = y_pol.copy()
                best_r = r_pol.copy()
        except Exception:
            pass
    
    # Ensure final validity
    if best_x is not None:
        best_x, best_y, best_r = _make_valid(best_x, best_y, best_r)
    
    # If we didn't get a valid solution, fall back to grid
    if best_x is None or not _check_validity(best_x, best_y, best_r):
        xg, yg, rg = _grid_init()
        best_x, best_y, best_r = xg, yg, rg
    
    result = np.column_stack([best_x, best_y, best_r])
    return result
# EVOLVE-BLOCK-END
