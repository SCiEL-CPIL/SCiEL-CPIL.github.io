import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 21

def _compute_score(pos, w, h):
    """Compute sum of radii for given positions in w x h rectangle."""
    n = len(pos)
    radii = np.zeros(n)
    for i in range(n):
        min_d = min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2) / 2.0
                min_d = min(min_d, d)
        radii[i] = max(0, min_d)
    return np.sum(radii), radii

def _objective(x, w, h):
    """Objective for optimization: negative sum of radii."""
    pos = x.reshape(-1, 2)
    score, _ = _compute_score(pos, w, h)
    return -score

def _grid_init(w, h, n, cols=None):
    """Initialize positions on a grid."""
    if cols is None:
        aspect = w / h
        cols = max(1, int(np.sqrt(n * aspect)))
    rows = int(np.ceil(n / cols))
    pos = np.zeros((n, 2))
    for i in range(n):
        c = i % cols
        r = i // cols
        pos[i, 0] = (c + 0.5) * w / cols
        pos[i, 1] = (r + 0.5) * h / rows
    return pos

def _random_init(w, h, n, seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    margin = 0.05
    pos = rng.uniform(margin, 1-margin, size=(n, 2))
    pos[:, 0] *= w
    pos[:, 1] *= h
    return pos

def _pack_for_aspect(w, h, time_budget=8.0):
    """Pack circles in w x h rectangle. Returns (pos, radii, score)."""
    start = time.time()
    deadline = start + time_budget
    
    best_pos = None
    best_score = -np.inf
    best_radii = None
    
    # Try multiple initializations
    initializations = []
    
    # Grid init
    for cols in range(2, 7):
        initializations.append(_grid_init(w, h, N, cols=cols))
    
    # Random inits
    for s in range(5):
        initializations.append(_random_init(w, h, N, seed=s*42))
    
    # Centroid-based init: place circles in a hexagonal-like pattern
    # Try different aspect-adjusted grids
    aspect = w / h
    for cols in [3, 4, 5, 6]:
        rows = int(np.ceil(N / cols))
        pos = np.zeros((N, 2))
        for i in range(N):
            c = i % cols
            r = i // cols
            # Hexagonal offset
            offset = 0.5 * (w / cols) if r % 2 == 1 else 0.0
            pos[i, 0] = (c + 0.5) * w / cols + offset
            pos[i, 1] = (r + 0.5) * h / rows
            # Clip to bounds
            pos[i, 0] = np.clip(pos[i, 0], 0.01, w-0.01)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, h-0.01)
        initializations.append(pos)
    
    bounds = [(0, w)] * N + [(0, h)] * N
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        x0 = init.flatten()
        
        # Try L-BFGS-B
        try:
            result = minimize(_objective, x0, args=(w, h), method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
            pos_opt = result.x.reshape(-1, 2)
            score, radii = _compute_score(pos_opt, w, h)
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = radii.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
        
        # Try Nelder-Mead (no bounds, but we can clip)
        try:
            result = minimize(_objective, x0, args=(w, h), method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            pos_opt = result.x.reshape(-1, 2)
            # Clip to bounds
            pos_opt[:, 0] = np.clip(pos_opt[:, 0], 0, w)
            pos_opt[:, 1] = np.clip(pos_opt[:, 1], 0, h)
            score, radii = _compute_score(pos_opt, w, h)
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = radii.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # If we still have time, try differential evolution with coarse bounds
    if time.time() < deadline - 2.0:
        try:
            bounds_de = [(0, w)] * N + [(0, h)] * N
            # Use a subset of bounds for speed
            result = differential_evolution(_objective, bounds_de, args=(w, h),
                                           maxiter=50, popsize=15, seed=42,
                                           tol=1e-8, polish=False)
            pos_opt = result.x.reshape(-1, 2)
            pos_opt[:, 0] = np.clip(pos_opt[:, 0], 0, w)
            pos_opt[:, 1] = np.clip(pos_opt[:, 1], 0, h)
            score, radii = _compute_score(pos_opt, w, h)
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = radii.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: grid
        pos = _grid_init(w, h, N)
        score, radii = _compute_score(pos, w, h)
        return pos, radii, score
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_score = 0
    best_result = None
    
    # Perimeter 4 => w + h = 2.
    # Symmetry: w and 2-w are equivalent.
    # Focus on w in [0.5, 1.0] for square and wider rectangles.
    
    # Sample aspect ratios - focus on likely optimal range
    w_values = np.concatenate([
        np.linspace(0.5, 1.0, 15),
        np.linspace(0.6, 0.9, 10)
    ])
    w_values = np.unique(np.sort(w_values))
    
    # First pass: coarse
    for w in w_values:
        if time.time() > deadline:
            break
        h = 2.0 - w
        pos, radii, score = _pack_for_aspect(w, h, time_budget=5.0)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, radii])
    
    # Second pass: refine around best
    if best_result is not None and time.time() < deadline - 10:
        # Find the w that gave best score
        # Re-run with more time for top candidates
        top_w = w_values[np.argmin(np.abs(w_values - (best_result[:, 0].max() - best_result[:, 0].min() if False else 1.0)))]
        # Actually, let's just refine around the aspect that worked best
        # We don't track which w gave the best, so let's try a few near 0.7-0.9
        refine_w = np.linspace(0.65, 0.95, 10)
        for w in refine_w:
            if time.time() > deadline - 5:
                break
            h = 2.0 - w
            pos, radii, score = _pack_for_aspect(w, h, time_budget=8.0)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([pos, radii])
    
    # Fallback
    if best_result is None:
        side = 5
        pos = _grid_init(1.0, 1.0, N, cols=side)
        radii = np.full(N, 0.5/side - 0.01)
        best_result = np.column_stack([pos, radii])
    
    return best_result
