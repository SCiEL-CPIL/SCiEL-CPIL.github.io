"""Packing 12 unit hexagons in a minimal regular hexagon:
   hex-grid init + global-then-local (DE + Powell) with vectorized overlap penalty."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = np.sqrt(3.0)
APO = SQRT3 / 2.0  # apothem of unit hexagon

# Unit flat-top hexagon: vertices (radius 1) and outward unit edge-normals
_HEX_V = np.array([
    [1.0, 0.0], [0.5, APO], [-0.5, APO],
    [-1.0, 0.0], [-0.5, -APO], [0.5, -APO],
])
_HEX_N = np.array([
    [SQRT3/2, 0.5], [0.0, 1.0], [-SQRT3/2, 0.5],
    [-SQRT3/2, -0.5], [0.0, -1.0], [SQRT3/2, -0.5],
])
_OUTER_N = _HEX_N

_DEADLINE = 135.0
_T0 = [0.0]

def _elapsed():
    return time.time() - _T0[0]

def _compute(x, y, a, lam):
    """Return (S, overlap_penalty). x,y,a are (N,) arrays."""
    ca = np.cos(a); sa = np.sin(a)
    vx = _HEX_V[:, 0]; vy = _HEX_V[:, 1]
    all_vx = ca[:, None] * vx[None, :] - sa[:, None] * vy[None, :] + x[:, None]
    all_vy = sa[:, None] * vx[None, :] + ca[:, None] * vy[None, :] + y[:, None]
    on0 = _OUTER_N[:, 0]; on1 = _OUTER_N[:, 1]
    dots = all_vx[:, :, None] * on0[None, None, :] + all_vy[:, :, None] * on1[None, None, :]
    S = float(np.max(dots)) / APO
    if S < 0.0:
        S = 0.0
    if lam == 0.0:
        return S, 0.0
    nx = _HEX_N[:, 0]; ny = _HEX_N[:, 1]
    all_nx = ca[:, None] * nx[None, :] - sa[:, None] * ny[None, :]
    all_ny = sa[:, None] * nx[None, :] + ca[:, None] * ny[None, :]
    pen = 0.0
    for i in range(N):
        vi_x = all_vx[i]; vi_y = all_vy[i]
        ni_x = all_nx[i]; ni_y = all_ny[i]
        for j in range(i + 1, N):
            ax = np.empty(12); ay = np.empty(12)
            ax[:6] = ni_x; ay[:6] = ni_y
            ax[6:] = all_nx[j]; ay[6:] = all_ny[j]
            pi = vi_x[:, None] * ax[None, :] + vi_y[:, None] * ay[None, :]
            pj = all_vx[j, :, None] * ax[None, :] + all_vy[j, :, None] * ay[None, :]
            min_i = pi.min(axis=0); max_i = pi.max(axis=0)
            min_j = pj.min(axis=0); max_j = pj.max(axis=0)
            sep = np.maximum(min_i - max_j, min_j - max_i).max()
            if sep < 0.0:
                d = -sep
                pen += d * d
    return S, lam * pen

def _objective(v, lam):
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]
    S, pen = _compute(x, y, a, lam)
    return S + pen

def _is_valid(x, y, a, tol=1e-6):
    S, raw_pen = _compute(x, y, a, 1.0)
    return raw_pen < tol, S

def _hex_grid_init(rng, spacing_scale=1.0, jitter=0.0):
    """Generate initial placement on a hexagonal lattice."""
    candidates = []
    for q in range(4):
        for r in range(3):
            candidates.append((q, r))
    rng.shuffle(candidates)
    chosen = candidates[:12]
    
    hex_cells = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                hex_cells.append((q, r))
    rng.shuffle(hex_cells)
    chosen_hex = hex_cells[:12]
    
    if rng.random() < 0.5:
        coords = chosen
    else:
        coords = chosen_hex
    
    x = np.zeros(N)
    y = np.zeros(N)
    a = np.zeros(N)
    
    for i, (q, r) in enumerate(coords):
        x[i] = SQRT3 * spacing_scale * (q + r / 2.0)
        y[i] = 1.5 * spacing_scale * r
        a[i] = 0.0
    
    if jitter > 0:
        x += rng.normal(0, jitter, N)
        y += rng.normal(0, jitter, N)
        a += rng.normal(0, jitter * 0.1, N)
    
    cx = x.mean(); cy = y.mean()
    x -= cx; y -= cy
    
    return np.concatenate([x, y, a])

def _fallback():
    inner = []
    for r in range(2):
        for c in range(6):
            inner.append([SQRT3 * (c - 2.5), 1.5 * r, 0.0])
    x = np.array([v[0] for v in inner])
    y = np.array([v[1] for v in inner])
    a = np.array([np.radians(v[2]) for v in inner])
    _, S = _compute(x, y, a, 0.0)
    return inner, (0.0, 0.0), float(S), 0.0

def _powell_refine(v0, lam, maxiter=100, xatol=1e-7, fatol=1e-9):
    """Run Powell local refinement with timeout guard."""
    try:
        res = minimize(lambda v: _objective(v, lam), v0, method='Powell',
                       options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol})
        if res.x is not None:
            return np.asarray(res.x, dtype=float)
    except Exception:
        pass
    return None

def construct():
    _T0[0] = time.time()
    rng = np.random.default_rng(42)
    
    best = None
    best_S = np.inf
    best_f = np.inf
    
    def _update_best(v, f):
        nonlocal best, best_S, best_f
        if v is not None and f < best_f:
            x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]
            ok, S = _is_valid(x, y, a)
            if ok and S < best_S:
                best_S = S
                best = v.copy()
            # Also track by objective for refinement chaining
            if f < best_f:
                best_f = f
                if best is None:
                    best = v.copy()
    
    # Phase 1: Global search with Differential Evolution
    if _elapsed() < _DEADLINE - 40:
        try:
            # Bounds: positions in [-6,6], angles in [0, 2*pi/3]
            bounds = [(-6.0, 6.0)] * N + [(-6.0, 6.0)] * N + [(0.0, 2.0*np.pi/3)] * N
            
            def de_obj(v):
                return _objective(v, 50.0)
            
            de_res = differential_evolution(
                de_obj, bounds,
                seed=42, maxiter=30, popsize=15,
                tol=1e-6, mutation=(0.3, 1.0), recombination=0.8,
                polish=False, init='sobol'
            )
            if de_res.x is not None:
                _update_best(np.asarray(de_res.x, dtype=float), float(de_res.fun))
        except Exception:
            pass
    
    # Phase 2: Multi-start Powell from diverse initializations
    inits = []
    
    # Hex grid starts
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.0))
    inits.append(_hex_grid_init(rng, spacing_scale=1.05, jitter=0.0))
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.1))
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.3))
    inits.append(_hex_grid_init(rng, spacing_scale=0.95, jitter=0.0))
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.0))
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.0))
    
    # If DE found a good solution, use it as a start
    if best is not None:
        inits.insert(0, best.copy())
    
    # Random starts
    for _ in range(4):
        x = rng.uniform(-4, 4, N)
        y = rng.uniform(-4, 4, N)
        a = rng.uniform(0, np.pi/3, N)
        inits.append(np.concatenate([x, y, a]))
    
    # Local optimization from each start
    for init in inits:
        if _elapsed() > _DEADLINE - 15:
            break
        try:
            # Phase A: moderate penalty to resolve overlaps
            r1 = _powell_refine(init, lam=10.0, maxiter=80, xatol=1e-6, fatol=1e-8)
            if r1 is not None:
                # Phase B: high penalty to tighten
                r2 = _powell_refine(r1, lam=50.0, maxiter=80, xatol=1e-7, fatol=1e-9)
                if r2 is not None:
                    f2 = _objective(r2, 50.0)
                    _update_best(r2, f2)
        except Exception:
            pass
    
    # Phase 3: Polish the best solution with very high penalty
    if best is not None and _elapsed() < _DEADLINE - 5:
        try:
            r3 = _powell_refine(best, lam=200.0, maxiter=100, xatol=1e-8, fatol=1e-10)
            if r3 is not None:
                x = r3[0:N]; y = r3[N:2*N]; a = r3[2*N:3*N]
                ok, S = _is_valid(x, y, a)
                if ok and S < best_S:
                    best_S = S
                    best = r3.copy()
        except Exception:
            pass
    
    # Final validation
    if best is None:
        return _fallback()
    
    x = best[0:N]; y = best[N:2*N]; a = best[2*N:3*N]
    ok, S = _is_valid(x, y, a)
    
    if not ok:
        # Try one more polish
        try:
            r4 = _powell_refine(best, lam=500.0, maxiter=50, xatol=1e-9, fatol=1e-11)
            if r4 is not None:
                x = r4[0:N]; y = r4[N:2*N]; a = r4[2*N:3*N]
                ok, S = _is_valid(x, y, a)
        except Exception:
            pass
    
    if not ok:
        return _fallback()
    
    inner = [[float(x[i]), float(y[i]), float(np.degrees(a[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
