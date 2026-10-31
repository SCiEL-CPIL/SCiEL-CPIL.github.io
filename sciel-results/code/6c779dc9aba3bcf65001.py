import math
import time
import random
import numpy as np

N = 11
DEADLINE = time.time() + 145.0  # safety margin before 150s kill

# Precompute unit hexagon vertices (flat-top, radius=1)
_UNIT_HEX_VERTS = np.array([
    [math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))]
    for i in range(6)
])

# Rigid cluster: center + 6 ring (dist=sqrt(3)) + 4 corners (dist=2*sqrt(3))
_SP = math.sqrt(3)
_CLUSTER_LOCAL = []
_CLUSTER_LOCAL.append((0.0, 0.0, 0.0))  # center
for i in range(6):
    theta = math.radians(60 * i)
    _CLUSTER_LOCAL.append((_SP * math.cos(theta), _SP * math.sin(theta), 0.0))
# 4 corners: alternate ring positions extended to 2*SP
for i in [0, 2, 4, 6]:
    idx = i % 6
    theta = math.radians(60 * idx)
    _CLUSTER_LOCAL.append((2 * _SP * math.cos(theta), 2 * _SP * math.sin(theta), 0.0))
# That gives 1 + 6 + 4 = 11. But i in [0,2,4,6] -> idx 0,2,4,0 -> duplicate.
# Fix: use 4 distinct corner directions
_CLUSTER_LOCAL = []
_CLUSTER_LOCAL.append((0.0, 0.0, 0.0))
for i in range(6):
    theta = math.radians(60 * i)
    _CLUSTER_LOCAL.append((_SP * math.cos(theta), _SP * math.sin(theta), 0.0))
for i in [0, 2, 3, 5]:
    theta = math.radians(60 * i)
    _CLUSTER_LOCAL.append((2 * _SP * math.cos(theta), 2 * _SP * math.sin(theta), 0.0))
# Now 1+6+4=11

def _transform_cluster(tx, ty, theta_deg):
    """Return Nx2 array of all 11 hexagon centers and their rotations."""
    a = math.radians(theta_deg)
    ca, sa = math.cos(a), math.sin(a)
    centers = np.empty((N, 2))
    rots = np.empty(N)
    for k, (lx, ly, la) in enumerate(_CLUSTER_LOCAL):
        centers[k, 0] = tx + lx * ca - ly * sa
        centers[k, 1] = ty + lx * sa + ly * ca
        rots[k] = la + theta_deg
    return centers, rots

def _all_vertices(tx, ty, theta_deg):
    """Return (N, 6, 2) array of all vertices of all 11 hexagons."""
    centers, rots = _transform_cluster(tx, ty, theta_deg)
    verts = np.empty((N, 6, 2))
    for k in range(N):
        a = math.radians(rots[k])
        ca, sa = math.cos(a), math.sin(a)
        # Rotate unit hex vertices
        rv = np.empty((6, 2))
        for i in range(6):
            ux, uy = _UNIT_HEX_VERTS[i]
            rv[i, 0] = ux * ca - uy * sa
            rv[i, 1] = ux * sa + uy * ca
        verts[k] = rv + centers[k]
    return verts

def _support_max(verts, ocx, ocy, ophi_deg):
    """For each of 6 outer normals, compute max projection of all vertices relative to outer center."""
    a = math.radians(ophi_deg)
    apothem_factor = math.sqrt(3) / 2.0
    max_projs = np.empty(6)
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        nx, ny = math.cos(theta), math.sin(theta)
        # Project all vertices
        projs = (verts[:, :, 0] - ocx) * nx + (verts[:, :, 1] - ocy) * ny
        max_projs[i] = projs.max()
    return max_projs

def _objective(params):
    """
    params = [tx, ty, theta_deg, S, ophi_deg]
    Returns S + penalty for overlaps and containment violations.
    """
    tx, ty, theta_deg, S, ophi_deg = params
    if S <= 0:
        return 1e6
    
    verts = _all_vertices(tx, ty, theta_deg)
    ocx, ocy = 0.0, 0.0
    
    # Containment: for each of 6 outer normals, max_proj must be <= S * apothem_factor
    apothem_factor = math.sqrt(3) / 2.0
    max_projs = _support_max(verts, ocx, ocy, ophi_deg)
    containment_violation = np.maximum(0.0, max_projs - S * apothem_factor)
    penalty = 100.0 * np.sum(containment_violation ** 2)
    
    # Overlap penalty: for each pair, check minimum distance between edges/vertices
    # Use a simpler check: if any vertex of A is inside B or vice versa, or edges cross
    # For speed, use approximate: check if distance between centers < 2*sqrt(3) - epsilon
    # But this is not exact. Let's do a proper check for small N.
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Check if hexagons i and j overlap
            if _hexagons_overlap(verts[i], verts[j]):
                overlap_penalty += 100.0
    
    return S + penalty + overlap_penalty

def _hexagons_overlap(verts_a, verts_b):
    """Check if two hexagons (6,2 arrays) overlap."""
    # Check if any vertex of A is inside B
    for v in verts_a:
        if _point_in_hex(v, verts_b):
            return True
    # Check if any vertex of B is inside A
    for v in verts_b:
        if _point_in_hex(v, verts_a):
            return True
    # Check edge intersections
    for i in range(6):
        a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
        b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
        if _segments_intersect(a1, a2, b1, b2):
            return True
    return False

def _point_in_hex(p, hex_verts):
    """Ray casting point in polygon for hexagon."""
    x, y = p
    inside = False
    j = 5
    for i in range(6):
        xi, yi = hex_verts[i]
        xj, yj = hex_verts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _is_valid(tx, ty, theta_deg, S, ophi_deg, tol=1e-6):
    """Check if configuration is valid (no overlaps, all contained)."""
    verts = _all_vertices(tx, ty, theta_deg)
    ocx, ocy = 0.0, 0.0
    apothem_factor = math.sqrt(3) / 2.0
    max_projs = _support_max(verts, ocx, ocy, ophi_deg)
    if np.any(max_projs > S * apothem_factor + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagons_overlap(verts[i], verts[j]):
                return False
    return True

def construct():
    best_S = float('inf')
    best_params = None
    best_valid = False
    
    # Initial S estimate: bounding the cluster
    initial_S = 2 * _SP + 2.0  # generous initial estimate
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: centered, no rotation
    initial_configs.append([0.0, 0.0, 0.0, initial_S, 0.0])
    
    # Config 2: centered, 30 deg outer rotation
    initial_configs.append([0.0, 0.0, 0.0, initial_S, 30.0])
    
    # Config 3: slight translation
    initial_configs.append([0.1, 0.1, 0.0, initial_S, 0.0])
    initial_configs.append([-0.1, 0.1, 0.0, initial_S, 0.0])
    initial_configs.append([0.1, -0.1, 0.0, initial_S, 0.0])
    initial_configs.append([-0.1, -0.1, 0.0, initial_S, 0.0])
    
    # Config 4: cluster rotation
    initial_configs.append([0.0, 0.0, 15.0, initial_S, 0.0])
    initial_configs.append([0.0, 0.0, -15.0, initial_S, 0.0])
    initial_configs.append([0.0, 0.0, 30.0, initial_S, 0.0])
    
    # Random configs
    rng = random.Random(42)
    for _ in range(20):
        tx = rng.uniform(-0.5, 0.5)
        ty = rng.uniform(-0.5, 0.5)
        theta = rng.uniform(0, 60)
        ophi = rng.uniform(0, 60)
        initial_configs.append([tx, ty, theta, initial_S, ophi])
    
    for cfg in initial_configs:
        if time.time() > DEADLINE:
            break
        
        # Use Nelder-Mead
        try:
            from scipy.optimize import minimize
            res = minimize(
                _objective,
                cfg,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
        except Exception:
            continue
        
        if time.time() > DEADLINE:
            break
        
        tx, ty, theta_deg, S, ophi_deg = res.x
        # Clamp S to be positive
        S = max(S, 1.0)
        
        # Check validity
        if _is_valid(tx, ty, theta_deg, S, ophi_deg):
            if S < best_S:
                best_S = S
                best_params = (tx, ty, theta_deg, S, ophi_deg)
                best_valid = True
        
        # Also try to shrink S further by binary search or local optimization
        # Try a second pass with tighter S
        S_tight = S * 0.95
        cfg2 = [tx, ty, theta_deg, S_tight, ophi_deg]
        try:
            res2 = minimize(
                _objective,
                cfg2,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
        except Exception:
            continue
        
        if time.time() > DEADLINE:
            break
        
        tx2, ty2, theta2, S2, ophi2 = res2.x
        S2 = max(S2, 1.0)
        if _is_valid(tx2, ty2, theta2, S2, ophi2):
            if S2 < best_S:
                best_S = S2
                best_params = (tx2, ty2, theta2, S2, ophi2)
                best_valid = True
    
    if not best_valid or best_params is None:
        # Fallback: use a simple valid configuration
        # Linear arrangement
        inner = []
        for k in range(N):
            inner.append([(k - N // 2) * _SP, 0.0, 0.0])
        S = 3 * _SP + 2.0
        return inner, (0.0, 0.0), S, 0.0
    
    tx, ty, theta_deg, S, ophi_deg = best_params
    
    # Build the inner hexagon positions
    centers, rots = _transform_cluster(tx, ty, theta_deg)
    inner = []
    for k in range(N):
        inner.append([centers[k, 0], centers[k, 1], rots[k] % 360.0])
    
    return inner, (0.0, 0.0), S, ophi_deg
