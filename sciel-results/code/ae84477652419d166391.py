"""Packing 12 unit hexagons in a minimal regular hexagon: SLSQP with hex-grid init + multi-start."""
import time
import numpy as np
from scipy.optimize import minimize

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
_OUTER_N = _HEX_N  # outer hexagon (flat-top) unit normals

_DEADLINE = 140.0
_T0 = [0.0]

def _elapsed():
    return time.time() - _T0[0]

def _hex_vertices(x, y, a):
    """Return (6,2) vertices of a unit hexagon at center (x,y) rotated by angle a."""
    ca = np.cos(a); sa = np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return R @ _HEX_V.T + np.array([x, y])[:, None]

def _hex_normals(a):
    """Return (6,2) outward normals of a unit hexagon rotated by angle a."""
    ca = np.cos(a); sa = np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return R @ _HEX_N.T

def _separation(i_verts, i_norms, j_verts, j_norms):
    """Compute minimum separation between two hexagons using separating axis theorem."""
    # All 12 normals (6 from each hexagon)
    axes = np.vstack([i_norms, j_norms])
    min_sep = np.inf
    for k in range(12):
        n = axes[k]
        # Project i's vertices onto axis n
        proj_i = i_verts @ n
        # Project j's vertices onto axis n
        proj_j = j_verts @ n
        min_i, max_i = proj_i.min(), proj_i.max()
        min_j, max_j = proj_j.min(), proj_j.max()
        # Separation along this axis
        sep = max(min_i - max_j, min_j - max_i)
        min_sep = min(min_sep, sep)
    return min_sep

def _compute_S(x, y, a):
    """Compute minimal enclosing side length S for outer hexagon centered at origin, rotation 0."""
    S = 0.0
    for i in range(N):
        verts = _hex_vertices(x[i], y[i], a[i])
        # Project all vertices onto the 6 outer hexagon normals
        for k in range(6):
            n = _OUTER_N[k]
            proj = verts @ n
            S = max(S, proj.max() / APO)
    return S

def _objective(v):
    """Objective: S + penalty for overlaps."""
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]
    
    S = _compute_S(x, y, a)
    
    # Penalty for overlaps
    pen = 0.0
    for i in range(N):
        vi = _hex_vertices(x[i], y[i], a[i])
        ni = _hex_normals(a[i])
        for j in range(i+1, N):
            vj = _hex_vertices(x[j], y[j], a[j])
            nj = _hex_normals(a[j])
            sep = _separation(vi, ni, vj, nj)
            if sep < 0.0:
                pen += sep * sep
    return S + 100.0 * pen

def _is_valid(x, y, a, tol=1e-5):
    """Check if configuration is valid (no overlaps) and return S."""
    for i in range(N):
        vi = _hex_vertices(x[i], y[i], a[i])
        ni = _hex_normals(a[i])
        for j in range(i+1, N):
            vj = _hex_vertices(x[j], y[j], a[j])
            nj = _hex_normals(a[j])
            sep = _separation(vi, ni, vj, nj)
            if sep < -tol:
                return False, None
    S = _compute_S(x, y, a)
    return True, S

def _fallback():
    """Guaranteed-valid: 2 rows x 6, wide spacing."""
    inner = []
    for r in range(2):
        for c in range(6):
            inner.append([3.2 * (c - 2.5), 2.9 * r, 0.0])
    return inner, (0.0, 0.0), 22.0, 0.0

def _hex_grid_initials():
    """Generate initial configurations based on honeycomb lattice."""
    rng = np.random.default_rng(42)
    initials = []
    
    # Honeycomb lattice: flat-top hexagons touch at distance sqrt(3)
    # Basis vectors for centers of adjacent hexagons
    v1 = np.array([SQRT3, 0.0])
    v2 = np.array([SQRT3/2, 1.5])
    
    # Generate lattice points in a region
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            points.append(p)
    points = np.array(points)
    
    # Try several cluster shapes
    # Shape 1: 2x6 rectangle
    for offset_i in range(-2, 3):
        for offset_j in range(-2, 3):
            cluster = []
            for r in range(2):
                for c in range(6):
                    idx = (offset_i + c) * len(range(-4,5)) + (offset_j + r)
                    if 0 <= idx < len(points):
                        cluster.append(points[idx])
            if len(cluster) == N:
                cluster = np.array(cluster)
                center = cluster.mean(axis=0)
                cluster = cluster - center
                x = cluster[:, 0]
                y = cluster[:, 1]
                a = np.zeros(N)
                v = np.concatenate([x, y, a])
                initials.append(v)
    
    # Shape 2: 3x4 rectangle
    for offset_i in range(-2, 3):
        for offset_j in range(-2, 3):
            cluster = []
            for r in range(3):
                for c in range(4):
                    idx = (offset_i + c) * len(range(-4,5)) + (offset_j + r)
                    if 0 <= idx < len(points):
                        cluster.append(points[idx])
            if len(cluster) == N:
                cluster = np.array(cluster)
                center = cluster.mean(axis=0)
                cluster = cluster - center
                x = cluster[:, 0]
                y = cluster[:, 1]
                a = np.zeros(N)
                v = np.concatenate([x, y, a])
                initials.append(v)
    
    # Shape 3: Hexagonal cluster (1 + 6 + 5 from next ring)
    # Central hexagon at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 5 more from the next ring
    def axial_to_cart(i, j):
        return np.array([SQRT3 * (i + 0.5*j), 1.5 * j])
    
    # Central 7
    c7 = [axial_to_cart(0,0), axial_to_cart(1,0), axial_to_cart(1,-1),
          axial_to_cart(0,-1), axial_to_cart(-1,0), axial_to_cart(-1,1),
          axial_to_cart(0,1)]
    
    # Ring 2 points
    ring2 = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            d = 0.5 * (abs(i) + abs(j) + abs(i+j))
            if d == 2:
                ring2.append(axial_to_cart(i, j))
    ring2 = np.array(ring2)
    
    # Pick 5 from ring2 in different ways
    for k in range(15):
        if k < 5:
            # Deterministic picks
            idx = [k*2, k*2+1, k*2+2, k*2+3, k*2+4]
            subset = ring2[[i % len(ring2) for i in idx]]
        else:
            idx = rng.choice(len(ring2), 5, replace=False)
            subset = ring2[idx]
        
        cluster = np.vstack([np.array(c7), subset])
        center = cluster.mean(axis=0)
        cluster = cluster - center
        x = cluster[:, 0]
        y = cluster[:, 1]
        a = np.zeros(N)
        v = np.concatenate([x, y, a])
        initials.append(v)
    
    # Random starts
    for k in range(10):
        x = rng.uniform(-4, 4, N)
        y = rng.uniform(-4, 4, N)
        a = rng.uniform(0, np.pi/3, N)
        v = np.concatenate([x, y, a])
        initials.append(v)
    
    return initials

def construct():
    _T0[0] = time.time()
    
    best = None
    best_S = np.inf
    
    initials = _hex_grid_initials()
    
    # Multi-start SLSQP optimization
    for c0 in initials:
        if _elapsed() > _DEADLINE - 5:
            break
        try:
            r = minimize(_objective, c0, method='SLSQP',
                         options={'maxiter': 200, 'ftol': 1e-8})
            if r.x is not None:
                x = r.x[0:N]; y = r.x[N:2*N]; a = r.x[2*N:3*N]
                ok, S = _is_valid(x, y, a)
                if ok and S < best_S:
                    best_S = S
                    best = r.x
        except Exception:
            pass
    
    # Polish the best candidate with Powell for finer tuning
    if best is not None and _elapsed() < _DEADLINE - 5:
        try:
            r2 = minimize(_objective, best, method='Powell',
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            if r2.x is not None:
                x = r2.x[0:N]; y = r2.x[N:2*N]; a = r2.x[2*N:3*N]
                ok, S = _is_valid(x, y, a)
                if ok and S < best_S:
                    best_S = S
                    best = r2.x
        except Exception:
            pass
    
    if best is None:
        return _fallback()
    
    x = best[0:N]; y = best[N:2*N]; a = best[2*N:3*N]
    ok, S = _is_valid(x, y, a)
    if not ok:
        return _fallback()
    
    inner = [[float(x[i]), float(y[i]), float(np.degrees(a[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
