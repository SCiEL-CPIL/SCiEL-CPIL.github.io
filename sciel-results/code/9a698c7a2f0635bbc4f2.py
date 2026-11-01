import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 14
D = 3

# Precompute pair indices as arrays for fast indexing
pair_indices = list(itertools.combinations(range(N), 2))
M = len(pair_indices)
PI = np.array([p[0] for p in pair_indices])
PJ = np.array([p[1] for p in pair_indices])

def _pairwise_sq(X):
    """Compute squared pairwise distances for N points in D dims. Returns (M,) array."""
    diff = X[PI] - X[PJ]
    return np.einsum('ij,ij->i', diff, diff)

def _score(X):
    """Score: (max/min)^2 of pairwise distances."""
    d2 = _pairwise_sq(X)
    mn = d2.min()
    mx = d2.max()
    if mn < 1e-14:
        return 1e12
    r = mx / mn
    return r * r

def _fibonacci_sphere(n, seed=0):
    """Generate n points roughly uniformly on the unit sphere."""
    rng = np.random.default_rng(seed)
    # Fibonacci sphere
    points = []
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    pts = np.array(points, dtype=float)
    # Small random perturbation to break symmetry
    pts += rng.normal(0, 0.01, size=pts.shape)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _objective_direct(x_flat):
    """Direct ratio objective: (max d2 / min d2)."""
    X = x_flat.reshape(N, D)
    X = X - X.mean(axis=0, keepdims=True)
    d2 = _pairwise_sq(X)
    mn = d2.min()
    mx = d2.max()
    if mn < 1e-14:
        return 1e12
    return mx / mn

def _objective_log(x_flat):
    """Log ratio: log(max d2) - log(min d2)."""
    X = x_flat.reshape(N, D)
    X = X - X.mean(axis=0, keepdims=True)
    d2 = _pairwise_sq(X)
    mn = d2.min()
    mx = d2.max()
    if mn < 1e-14:
        return 1e12
    return np.log(mx) - np.log(mn)

def _soft_objective(x_flat, tau):
    """Soft min/max objective using log-sum-exp."""
    X = x_flat.reshape(N, D)
    X = X - X.mean(axis=0, keepdims=True)
    d2 = _pairwise_sq(X)
    
    # Soft max via log-sum-exp
    max_d2 = d2.max()
    logsumexp_max = max_d2 + np.log(np.sum(np.exp(tau * (d2 - max_d2))))
    soft_max = logsumexp_max / tau
    
    # Soft min
    min_d2 = d2.min()
    logsumexp_min = min_d2 - np.log(np.sum(np.exp(-tau * (d2 - min_d2))))
    soft_min = logsumexp_min / tau
    
    if soft_min < 1e-14:
        return 1e12
    return soft_max / soft_min

def _polish_nm(x_flat, time_budget=5.0):
    """Polish using Nelder-Mead on the direct objective."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_budget:
            return _objective_direct(x)
        return _objective_direct(x)
    
    try:
        res = minimize(
            obj, x_flat, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x_flat

def _polish_bfgs(x_flat, time_budget=3.0):
    """Polish using L-BFGS-B on log objective."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_budget:
            return _objective_log(x)
        return _objective_log(x)
    
    try:
        res = minimize(
            obj, x_flat, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 5000}
        )
        return res.x
    except Exception:
        return x_flat

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_score = np.inf
    best_X = None
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return deadline - time.time()
    
    def try_update(x_flat):
        nonlocal best_score, best_X
        X = x_flat.reshape(N, D)
        X = X - X.mean(axis=0, keepdims=True)
        d2 = _pairwise_sq(X)
        if d2.min() < 1e-10:
            return
        s = _score(X)
        if s < best_score:
            best_score = s
            best_X = X.copy()
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Fibonacci sphere with perturbations
    for seed in range(8):
        pts = _fibonacci_sphere(N, seed=seed)
        # Random radial scaling
        scales = np.random.default_rng(seed + 100).uniform(0.5, 1.5, size=(N, 1))
        pts_scaled = pts * scales
        inits.append(pts_scaled.flatten())
    
    # 2. Random in sphere
    rng = np.random.default_rng(42)
    for seed in range(8):
        r = np.random.default_rng(seed + 200)
        X = r.normal(size=(N, D))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms * r.uniform(0.3, 1.2, size=(N, 1))
        inits.append(X.flatten())
    
    # 3. Random in cube
    for seed in range(4):
        r = np.random.default_rng(seed + 300)
        X = r.uniform(-1, 1, size=(N, D))
        inits.append(X.flatten())
    
    # 4. Perturbed regular polytope-like structures
    # Try icosahedron + extra points
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norms
    # Add 2 more points
    extra = rng.normal(size=(2, D))
    extra_norms = np.linalg.norm(extra, axis=1, keepdims=True)
    extra = extra / extra_norms
    full = np.vstack([ico, extra])
    inits.append(full.flatten())
    
    # Process each initialization
    for init_idx, x_flat in enumerate(inits):
        if remaining() < 10.0:
            break
        
        # Center
        X = x_flat.reshape(N, D)
        X = X - X.mean(axis=0, keepdims=True)
        x_flat = X.flatten()
        
        # Phase 1: Soft objective optimization with increasing tau
        taus = [20, 50, 100, 200, 400]
        for tau in taus:
            if remaining() < 8.0:
                break
            
            try:
                res = minimize(
                    _soft_objective, x_flat, args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000}
                )
                x_flat = res.x
            except Exception:
                pass
        
        # Phase 2: Direct objective with L-BFGS-B
        if remaining() > 6.0:
            try:
                res = minimize(
                    _objective_direct, x_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 5000}
                )
                x_flat = res.x
            except Exception:
                pass
        
        # Phase 3: Log objective polish
        if remaining() > 4.0:
            x_flat = _polish_bfgs(x_flat, time_budget=min(3.0, remaining() * 0.3))
        
        # Phase 4: Nelder-Mead final polish
        if remaining() > 2.0:
            x_flat = _polish_nm(x_flat, time_budget=min(5.0, remaining() * 0.4))
        
        # Update best
        try_update(x_flat)
        
        # If we have good time left, do local perturbation search around best
        if best_X is not None and remaining() > 20.0:
            # Try small perturbations
            for pert_trial in range(5):
                if remaining() < 15.0:
                    break
                X_pert = best_X + np.random.default_rng(pert_trial + init_idx).normal(0, 0.01, size=(N, D))
                X_pert = X_pert - X_pert.mean(axis=0, keepdims=True)
                x_pert = X_pert.flatten()
                
                # Quick optimize
                try:
                    res = minimize(
                        _objective_direct, x_pert,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'maxfun': 2000}
                    )
                    x_pert = res.x
                    x_pert = _polish_nm(x_pert, time_budget=min(3.0, remaining() * 0.2))
                    try_update(x_pert)
                except Exception:
                    pass
    
    # Final: if we still have time, do more polish on the best
    if best_X is not None and remaining() > 10.0:
        x_best = best_X.flatten()
        # Multiple polish passes
        for _ in range(3):
            if remaining() < 5.0:
                break
            x_best = _polish_nm(x_best, time_budget=min(5.0, remaining() * 0.3))
            try_update(x_best)
        # One more BFGS pass
        if remaining() > 5.0:
            x_best = _polish_bfgs(x_best, time_budget=min(3.0, remaining() * 0.3))
            try_update(x_best)
    
    if best_X is None:
        # Fallback
        side = int(np.ceil(N ** (1.0 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_X = np.array(grid, dtype=float)
    
    return best_X
