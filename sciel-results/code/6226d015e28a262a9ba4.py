"""Structural change: free-form Nelder-Mead optimization over all 3N+3 parameters
(inner hex positions/rotations + outer center/rotation) with smooth objective."""
N = 11
import numpy as np
import math
import time
from scipy.optimize import minimize

# ---- geometry helpers ----------------------------------------------------

def _hex_verts(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    angs = a + np.pi/6 + np.arange(6)*np.pi/3
    return np.column_stack([cx + s*np.cos(angs), cy + s*np.sin(angs)])

def _all_verts(positions, angles, s=1.0):
    n = len(positions)
    v = np.empty((n*6, 2))
    for i in range(n):
        v[i*6:(i+1)*6] = _hex_verts(positions[i][0], positions[i][1], s, angles[i])
    return v

def _required_side(positions, angles, phi_rad, center=(0.0, 0.0)):
    """Exact required side length of outer hex with rotation phi centered at center."""
    verts = _all_verts(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_proj = 0.0
    for t in thetas:
        proj = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (math.sqrt(3)/2)

def _smooth_side(positions, angles, phi_rad, center=(0.0, 0.0), beta=30.0):
    """Smooth surrogate for the side length using log-sum-exp."""
    verts = _all_verts(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    projs = np.zeros((6, len(verts)))
    for k, t in enumerate(thetas):
        projs[k] = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
    m = np.max(projs, axis=1)
    lse_per_dir = m + (1.0/beta) * np.log(np.sum(np.exp(projs - m[:, None]), axis=1))
    m2 = np.max(lse_per_dir)
    lse_final = m2 + (1.0/beta) * math.log(math.exp(beta * (lse_per_dir - m2)).sum())
    return lse_final / (math.sqrt(3)/2)

# ---- SAT overlap check ---------------------------------------------------

def _sat_gap(p1, a1, p2, a2, s=1.0):
    v1 = _hex_verts(p1[0], p1[1], s, a1)
    v2 = _hex_verts(p2[0], p2[1], s, a2)
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        nv = np.array([-e[1], e[0]])
        n = np.linalg.norm(nv)
        if n < 1e-12:
            continue
        nv /= n
        proj1 = v1 @ nv
        proj2 = v2 @ nv
        gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_disjoint(positions, angles, tol=1e-9):
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if _sat_gap(positions[i], angles[i], positions[j], angles[j]) < -tol:
                return False
    return True

# ---- Objective for Nelder-Mead -------------------------------------------

def _objective(x, beta=30.0):
    """
    x = [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, cx, cy, phi]
    Returns smooth side + penalty for overlaps.
    """
    positions = [(x[3*i], x[3*i+1]) for i in range(N)]
    angles = [x[3*i+2] for i in range(N)]
    cx, cy, phi = x[3*N], x[3*N+1], x[3*N+2]
    
    side = _smooth_side(positions, angles, phi, (cx, cy), beta)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            gap = _sat_gap(positions[i], angles[i], positions[j], angles[j])
            if gap < 0:
                penalty += gap * gap * 100.0
    return side + penalty

# ---- Initial configurations ----------------------------------------------

def _init_343():
    s3 = math.sqrt(3)
    def axial_to_xy(q, r):
        return (1.5 * q, s3 * (r + q / 2.0))
    coords = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(1,1),(0,2),(-1,2)]
    x = np.zeros(3*N + 3)
    for i, (q, r) in enumerate(coords):
        px, py = axial_to_xy(q, r)
        x[3*i] = px
        x[3*i+1] = py
        x[3*i+2] = 0.0
    x[3*N] = 0.0
    x[3*N+1] = 0.0
    x[3*N+2] = 0.0
    return x

def _init_alternatives():
    s3 = math.sqrt(3)
    def axial_to_xy(q, r):
        return (1.5 * q, s3 * (r + q / 2.0))
    
    inits = []
    # 3-4-3
    c1 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(1,1),(0,2),(-1,2)]
    # Different ring-2 selections
    c2 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(0,2),(-2,2),(0,-2)]
    c3 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(1,1),(-2,0),(-1,-1)]
    c4 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(0,2),(-2,1),(1,-2)]
    
    for coords in [c1, c2, c3, c4]:
        x = np.zeros(3*N + 3)
        for i, (q, r) in enumerate(coords):
            px, py = axial_to_xy(q, r)
            x[3*i] = px
            x[3*i+1] = py
            x[3*i+2] = 0.0
        inits.append(x)
    
    # Random perturbations of 3-4-3
    base = _init_343()
    rng = np.random.RandomState(42)
    for _ in range(4):
        x = base.copy()
        x[:3*N] += rng.randn(3*N) * 0.3
        inits.append(x)
    
    return inits

# ---- Main construct -------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_side = float('inf')
    best_x = None
    
    inits = _init_alternatives()
    
    for x0 in inits:
        if time.time() > deadline - 20.0:
            break
        
        # Nelder-Mead optimization
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
        except Exception:
            continue
        
        x_opt = res.x
        positions = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
        angles = [x_opt[3*i+2] for i in range(N)]
        cx, cy, phi = x_opt[3*N], x_opt[3*N+1], x_opt[3*N+2]
        
        # Check validity
        if not _check_disjoint(positions, angles, tol=1e-7):
            continue
        
        side = _required_side(positions, angles, phi, (cx, cy))
        
        if side < best_side:
            best_side = side
            best_x = x_opt
    
    # If no valid result, use fallback
    if best_x is None:
        x0 = _init_343()
        positions = [(x0[3*i], x0[3*i+1]) for i in range(N)]
        angles = [x0[3*i+2] for i in range(N)]
        best_side = _required_side(positions, angles, 0.0, (0.0, 0.0))
        best_x = x0
    
    # Extract final result
    positions = [(best_x[3*i], best_x[3*i+1]) for i in range(N)]
    angles = [best_x[3*i+2] for i in range(N)]
    cx, cy, phi = best_x[3*N], best_x[3*N+1], best_x[3*N+2]
    
    # Shift so outer center is at origin
    positions = [(p[0]-cx, p[1]-cy) for p in positions]
    phi = phi  # outer rotation stays the same
    
    # Recompute side in final frame
    final_side = _required_side(positions, angles, phi, (0.0, 0.0))
    
    # Final verification
    if not _check_disjoint(positions, angles, tol=1e-7):
        # Fallback to linear
        positions = [(4.0*k, 0.0) for k in range(-(N//2), N-N//2)]
        angles = [0.0]*N
        phi = 0.0
        final_side = _required_side(positions, angles, phi, (0.0, 0.0))
    
    final_side *= (1.0 + 1e-9)
    
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, math.degrees(phi)
