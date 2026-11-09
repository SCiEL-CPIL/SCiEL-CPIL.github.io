"""Packing 12 unit hexagons in a minimal regular hexagon: Lattice-based LP + Local Refinement."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

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

_DEADLINE = 140.0
_T0 = [0.0]

def _elapsed():
    return time.time() - _T0[0]

def _compute_S_lp(x, y, a):
    """
    Compute minimal outer hexagon side S using Linear Programming.
    Variables: S (scalar)
    Constraints: For each inner hex i, each vertex j, each outer normal k:
    n_k . (R_a_i v_j + t_i) <= S * APO
    """
    # A is the apothem of the outer hexagon. Side length S = 2 * A / sqrt(3)?
    # No, for a regular hexagon, Side = 2 * Apothem / sqrt(3) is incorrect.
    # Apothem = Side * sqrt(3) / 2. So Side = 2 * Apothem / sqrt(3).
    # Wait, the constraint is n . p <= Apothem.
    # If we define S as the side length, then Apothem = S * sqrt(3) / 2.
    # So n . p <= S * sqrt(3) / 2.
    
    # Let's use Apothem A as the variable to simplify, then convert to S.
    # Minimize A
    # Constraints: n_k . p_{i,j} <= A
    
    all_vx = np.zeros((N, 6))
    all_vy = np.zeros((N, 6))
    ca = np.cos(a); sa = np.sin(a)
    vx = _HEX_V[:, 0]; vy = _HEX_V[:, 1]
    
    for i in range(N):
        all_vx[i] = ca[i] * vx - sa[i] * vy + x[i]
        all_vy[i] = sa[i] * vx + ca[i] * vy + y[i]
        
    # Flatten vertices
    p_x = all_vx.flatten()
    p_y = all_vy.flatten()
    
    # Constraints: n_k . p <= A
    # A is the variable. We want to minimize A.
    # c = [1]
    c = np.array([1.0])
    
    A_ub = np.zeros((N * 6 * 6, 1))
    b_ub = np.zeros(N * 6 * 6)
    
    idx = 0
    for i in range(N):
        for j in range(6):
            px = p_x[i * 6 + j]
            py = p_y[i * 6 + j]
            for k in range(6):
                nk_x = _OUTER_N[k, 0]
                nk_y = _OUTER_N[k, 1]
                A_ub[idx, 0] = -(nk_x * px + nk_y * py)
                b_ub[idx] = 0.0
                idx += 1
                
    # Also A > 0
    # linprog requires A_ub x <= b_ub
    # We have - (n . p) <= A  =>  -1 * A <= - (n . p) is wrong sign for minimization?
    # We want n . p - A <= 0 => (n . p) * 1 + (-1) * A <= 0? No, A is variable.
    # Constraint: n_k . p_{i,j} - A <= 0
    # Coeff for A is -1.
    # So A_ub[idx, 0] = -1.0
    # b_ub[idx] = n_k . p_{i,j}
    
    A_ub = np.zeros((N * 6 * 6, 1))
    b_ub = np.zeros(N * 6 * 6)
    idx = 0
    for i in range(N):
        for j in range(6):
            px = p_x[i * 6 + j]
            py = p_y[i * 6 + j]
            for k in range(6):
                val = _OUTER_N[k, 0] * px + _OUTER_N[k, 1] * py
                A_ub[idx, 0] = -1.0
                b_ub[idx] = val
                idx += 1
                
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    if res.status != 0:
        return np.inf
    A_opt = res.x[0]
    if A_opt < 0:
        return np.inf
    S = 2.0 * A_opt / SQRT3
    return float(S)

def _check_overlap(x, y, a, tol=1e-6):
    """Check if any two inner hexagons overlap."""
    ca = np.cos(a); sa = np.sin(a)
    all_vx = np.zeros((N, 6))
    all_vy = np.zeros((N, 6))
    all_nx = np.zeros((N, 6))
    all_ny = np.zeros((N, 6))
    vx = _HEX_V[:, 0]; vy = _HEX_V[:, 1]
    nx = _HEX_N[:, 0]; ny = _HEX_N[:, 1]
    
    for i in range(N):
        all_vx[i] = ca[i] * vx - sa[i] * vy + x[i]
        all_vy[i] = sa[i] * vx + ca[i] * vy + y[i]
        all_nx[i] = ca[i] * nx - sa[i] * ny
        all_ny[i] = sa[i] * nx + ca[i] * ny
        
    for i in range(N):
        for j in range(i + 1, N):
            # SAT check
            # Axes: normals of i and j
            axes = np.vstack([all_nx[i], all_nx[j]])
            proj_i = np.dot(all_vx[i], axes.T)
            proj_j = np.dot(all_vx[j], axes.T)
            min_i = proj_i.min(axis=1)
            max_i = proj_i.max(axis=1)
            min_j = proj_j.min(axis=1)
            max_j = proj_j.max(axis=1)
            
            # Also check y projections? No, SAT works in 2D with 2D axes.
            # We need to project onto 2D axes.
            # Let's do it properly.
            overlap = True
            for ax in axes:
                p_i = all_vx[i] * ax[0] + all_vy[i] * ax[1]
                p_j = all_vx[j] * ax[0] + all_vy[j] * ax[1]
                if p_i.max() < p_j.min() - tol or p_j.max() < p_i.min() - tol:
                    overlap = False
                    break
            if overlap:
                return False
    return True

def _generate_lattice_config(spacing, rot_deg, seed):
    """Generate a lattice of hexagons and pick 12."""
    rng = np.random.default_rng(seed)
    rot = np.radians(rot_deg)
    
    # Hex lattice vectors
    # u = (spacing, 0)
    # v = (spacing/2, spacing * sqrt(3)/2)
    u = np.array([spacing, 0.0])
    v = np.array([spacing / 2.0, spacing * SQRT3 / 2.0])
    
    # Generate a grid of points
    points = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            p = q * u + r * v
            # Rotate
            p_rot = np.array([
                p[0] * np.cos(rot) - p[1] * np.sin(rot),
                p[0] * np.sin(rot) + p[1] * np.cos(rot)
            ])
            points.append(p_rot)
            
    points = np.array(points)
    
    # We want to pick 12 points that are compact.
    # Strategy: Sort by distance to centroid, pick 12 closest?
    # Or pick a random subset and optimize?
    # For now, pick the 12 points closest to the origin (0,0) after centering the grid.
    # But the grid is centered at 0.
    # Let's just pick 12 points from the center of the grid.
    # The points are generated in a loop, so they are roughly ordered.
    # Let's sort by distance to (0,0).
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    chosen_idx = sorted_idx[:12]
    
    x = points[chosen_idx, 0]
    y = points[chosen_idx, 1]
    a = np.zeros(12) # All aligned
    
    # Center the configuration
    cx = x.mean(); cy = y.mean()
    x -= cx; y -= cy
    
    return x, y, a

def _objective(v, lam):
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]
    S = _compute_S_lp(x, y, a)
    if S == np.inf:
        return 1e10
    if lam > 0:
        if not _check_overlap(x, y, a):
            # Calculate penalty based on overlap depth? 
            # For simplicity, large penalty if overlap
            return S + 100.0 * lam
    return S

def _fallback():
    inner = []
    for r in range(2):
        for c in range(6):
            inner.append([SQRT3 * (c - 2.5), 1.5 * r, 0.0])
    x = np.array([v[0] for v in inner])
    y = np.array([v[1] for v in inner])
    a = np.array([np.radians(v[2]) for v in inner])
    S = _compute_S_lp(x, y, a)
    return inner, (0.0, 0.0), float(S), 0.0

def construct():
    _T0[0] = time.time()
    rng = np.random.default_rng(42)
    
    best = None
    best_S = np.inf
    
    # Phase 1: Lattice Search
    # Try different spacings and rotations
    spacings = [SQRT3 * (1.0 + 0.05 * i) for i in range(5)]
    rotations = [0.0, 15.0, 30.0, 45.0]
    
    candidates = []
    
    for s in spacings:
        for rot in rotations:
            if _elapsed() > _DEADLINE - 20:
                break
            x, y, a = _generate_lattice_config(s, rot, 42)
            if not _check_overlap(x, y, a):
                continue
            S = _compute_S_lp(x, y, a)
            if S < best_S:
                best_S = S
                best = np.concatenate([x, y, a])
            candidates.append((S, np.concatenate([x, y, a])))
            
    if best is None:
        return _fallback()
        
    # Sort candidates by S
    candidates.sort(key=lambda t: t[0])
    
    # Phase 2: Local Refinement
    # Use the best 3 candidates as starts for Powell
    for S_init, v_init in candidates[:3]:
        if _elapsed() > _DEADLINE - 10:
            break
        try:
            # Use penalty method to ensure no overlap during refinement
            # Start with high penalty, then reduce?
            # Powell doesn't handle constraints well, so we rely on the penalty.
            # We use a smooth penalty for overlap.
            
            def obj_func(v):
                x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]
                S = _compute_S_lp(x, y, a)
                if S == np.inf:
                    return 1e10
                # Check overlap
                # We need a smooth overlap penalty.
                # For now, use a hard check + large penalty if overlap.
                # This might cause Powell to stall.
                # Let's use the SAT separation distance.
                ca = np.cos(a); sa = np.sin(a)
                all_vx = np.zeros((N, 6))
                all_vy = np.zeros((N, 6))
                all_nx = np.zeros((N, 6))
                all_ny = np.zeros((N, 6))
                vx = _HEX_V[:, 0]; vy = _HEX_V[:, 1]
                nx = _HEX_N[:, 0]; ny = _HEX_N[:, 1]
                
                for i in range(N):
                    all_vx[i] = ca[i] * vx - sa[i] * vy + x[i]
                    all_vy[i] = sa[i] * vx + ca[i] * vy + y[i]
                    all_nx[i] = ca[i] * nx - sa[i] * ny
                    all_ny[i] = sa[i] * nx + ca[i] * ny
                    
                pen = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        axes = np.vstack([all_nx[i], all_nx[j]])
                        overlap = True
                        min_sep = 0.0
                        for ax in axes:
                            p_i = all_vx[i] * ax[0] + all_vy[i] * ax[1]
                            p_j = all_vx[j] * ax[0] + all_vy[j] * ax[1]
                            sep = min(p_i.max(), p_j.max()) - max(p_i.min(), p_j.min())
                            if sep > 1e-6:
                                overlap = False
                                break
                            else:
                                min_sep = min(min_sep, sep)
                        if overlap:
                            # Penalty proportional to penetration depth
                            # min_sep is negative if overlapping
                            pen += (-min_sep) ** 2
                
                return S + 50.0 * pen

            r = minimize(obj_func, v_init, method='Powell',
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if r.x is not None:
                x = r.x[0:N]; y = r.x[N:2*N]; a = r.x[2*N:3*N]
                if _check_overlap(x, y, a):
                    S = _compute_S_lp(x, y, a)
                    if S < best_S:
                        best_S = S
                        best = np.concatenate([x, y, a])
        except Exception:
            pass

    if best is None:
        return _fallback()
        
    x = best[0:N]; y = best[N:2*N]; a = best[2*N:3*N]
    inner = [[float(x[i]), float(y[i]), float(np.degrees(a[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(best_S), 0.0
