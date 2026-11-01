import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 32
_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)

def _validity_check(arr, eps=1e-9):
    """Return True if all circles are disjoint (interiors) and inside [0,1]^2."""
    if arr is None or len(arr) != N:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0):
        return False
    # inside square
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    # disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - eps:
                return False
    return True

def _score(arr):
    """Sum of radii if valid, else -inf."""
    if _validity_check(arr):
        return float(np.sum(arr[:, 2]))
    return -np.inf

def _hex_seed():
    """Staggered hexagonal-ish lattice trimmed to N circles."""
    pts = []
    row_h = 0.8660254  # sqrt(3)/2
    cols = 0
    centers = []
    while len(centers) < N + 10:
        y = row_h * cols
        xoff = 0.5 if (cols % 2) else 0.0
        for i in range(0, 20):
            centers.append((xoff + i, y))
            if len(centers) >= N + 10:
                break
        cols += 1
    centers = centers[:N]
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    # normalize into [0,1] with margin
    cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-12)
    cy = (cy - cy.min()) / (cy.max() - cy.min() + 1e-12)
    # base radius: min distance to boundary /2, and half min pair distance
    r = np.full(N, 0.0)
    for i in range(N):
        dmin = min(cx[i], 1 - cx[i], cy[i], 1 - cy[i])
        r[i] = dmin * 0.5
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            r[i] = min(r[i], d * 0.5)
    return np.column_stack([cx, cy, r])

def _shrink_to_valid(arr):
    """Given a candidate, shrink radii minimally to guarantee validity."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    # ensure inside
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
    # ensure disjoint: iteratively shrink
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s:
                    # shrink the larger one (or both half)
                    need = s - d
                    # reduce r[i] and r[j] so that r[i]+r[j] <= d
                    # keep ratio
                    tot = r[i] + r[j]
                    if tot <= 0:
                        continue
                    new_tot = max(0.0, d)
                    r[i] = r[i] * new_tot / tot
                    r[j] = r[j] * new_tot / tot
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])

def _penalty_objective(vars, wall_k=100.0, pair_k=100.0):
    """
    Objective function to minimize.
    vars: array of length 3*N: [x1, y1, r1, x2, y2, r2, ...]
    """
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Penalty for negative radii
    pen = np.sum(np.maximum(0, -r)) ** 2
    
    # Penalty for wall violations
    for i in range(N):
        if x[i] - r[i] < 0: pen += wall_k * (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1: pen += wall_k * (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0: pen += wall_k * (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1: pen += wall_k * (y[i] + r[i] - 1) ** 2
        
    # Penalty for pair overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            s = r[i] + r[j]
            if d < s:
                pen += pair_k * (s - d) ** 2
                
    # We want to maximize sum(r), so we minimize -sum(r)
    return -np.sum(r) + pen

def _optimize_from_start(start_arr, time_limit):
    """
    Run scipy minimize from a starting configuration.
    """
    x0 = start_arr[:, 0].copy()
    y0 = start_arr[:, 1].copy()
    r0 = start_arr[:, 2].copy()
    
    # Flatten variables
    vars0 = np.zeros(3 * N)
    vars0[0::3] = x0
    vars0[1::3] = y0
    vars0[2::3] = r0
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    start_time = time.time()
    best_local = start_arr.copy()
    best_local_score = _score(best_local)
    
    # Use Powell method as it doesn't require gradients and is robust to non-smooth penalties
    # Nelder-Mead is also an option, but Powell is often better for this dimensionality
    method = 'Powell'
    
    # We will run the optimizer in chunks to respect time limit
    # Powell doesn't easily support early stopping via callback in all versions, 
    # but we can wrap it or just run a few iterations and check time.
    # Actually, scipy.optimize.minimize supports 'callback' which is called after each iteration.
    
    def callback(vars):
        nonlocal best_local, best_local_score
        # Unpack
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        arr = np.column_stack([x, y, r])
        
        # Check time
        if time.time() > time_limit:
            raise Exception("TimeLimitExceeded")
            
        # Check validity and update best
        if _validity_check(arr):
            sc = _score(arr)
            if sc > best_local_score:
                best_local_score = sc
                best_local = arr.copy()
                
    try:
        res = minimize(_penalty_objective, vars0, method=method, bounds=bounds, 
                       options={'maxiter': 500, 'callback': callback})
    except Exception as e:
        if "TimeLimitExceeded" not in str(e):
            pass # Ignore other errors, return best found so far
    
    # Final check on result
    x = res.x[0::3]
    y = res.x[1::3]
    r = res.x[2::3]
    arr = np.column_stack([x, y, r])
    if _validity_check(arr):
        sc = _score(arr)
        if sc > best_local_score:
            best_local_score = sc
            best_local = arr.copy()
            
    return best_local, best_local_score

def construct():
    t0 = time.time()
    
    # Baseline from hex seed
    base = _hex_seed()
    base = _shrink_to_valid(base)
    best = base
    best_score = _score(best)
    
    rng = np.random.default_rng(12345)
    
    restart = 0
    # We have ~145 seconds. Each Powell run might take a few seconds.
    # Let's allow up to 20-30 restarts.
    
    while time.time() < _DEADLINE - 5:
        # Build initial configuration
        # Jitter the hex seed
        x0 = base[:, 0].copy()
        y0 = base[:, 1].copy()
        r0 = base[:, 2].copy()
        
        # Random jitter
        jit = 0.05
        x0 += rng.uniform(-jit, jit, N)
        y0 += rng.uniform(-jit, jit, N)
        x0 = np.clip(x0, 0.0, 1.0)
        y0 = np.clip(y0, 0.0, 1.0)
        
        # Random initial radius scale
        scale = rng.uniform(0.7, 1.3)
        r0 = r0 * scale
        
        # Randomize order / positions for diversity occasionally
        if restart % 5 == 0:
            perm = rng.permutation(N)
            x0 = x0[perm]; y0 = y0[perm]; r0 = r0[perm]
            
        start_arr = np.column_stack([x0, y0, r0])
        # Ensure start is valid-ish for optimizer to work well, though penalty handles it
        # start_arr = _shrink_to_valid(start_arr)
        
        arr, sc = _optimize_from_start(start_arr, _DEADLINE - 5)
        
        # Shrink to valid to ensure we return a valid packing
        arr = _shrink_to_valid(arr)
        sc = _score(arr)
        
        if sc > best_score:
            best_score = sc
            best = arr.copy()
            
        restart += 1
        
    # Final safety: ensure validity
    best = _shrink_to_valid(best)
    if not _validity_check(best):
        # Fall back to a guaranteed-valid trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
        
    return best
