"""Fresh design: honeycomb start + vectorized SA + Nelder-Mead polish."""
import math
import time
import numpy as np

N = 12

# 6 unit vectors for hexagon vertex directions (pointy-top: vertices at 0,60,...)
HEX_DIRS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

# 6 vertex offsets for a unit hexagon (side 1)
VERTEX_OFFSETS = HEX_DIRS  # distance 1 from center

# Container side length from a set of points (all 72 vertices of 12 hexagons)
def container_side(points):
    # points: (72, 2)
    # For each of 6 axes, compute width; side = max width / 2
    # Actually for a regular hexagon with side s, the width across flats = 2*(sqrt(3)/2)*s = sqrt(3)*s
    # Wait, need to be careful. A regular hexagon of side s: 
    # - distance across opposite vertices = 2s
    # - distance across opposite flats = sqrt(3)*s
    # The 6 vertices are at distance s from center.
    # If we project all points onto the 6 vertex-direction axes, the max extent in each direction
    # from the center must be <= s for the point to be inside.
    # But the container is centered somewhere, not necessarily at origin.
    # 
    # Better approach: The container hexagon (center C, side S, rotation theta) contains a point P
    # iff for all 6 face normals n_i, dot(P - C, n_i) <= S * cos(0) ... 
    # 
    # Let me think differently. A regular hexagon with center C, side S, and one vertex pointing in direction d0
    # has 6 vertices at C + S*exp(i*(d0 + k*60)).
    # The 6 face normals (outward) point in directions d0 + 30 + k*60, and the face is at distance S*sqrt(3)/2 from center.
    # 
    # So P is inside iff for all k: dot(P - C, n_k) <= S*sqrt(3)/2 where n_k are the 6 outward face normals.
    # 
    # To minimize S for a given set of points and container center C and rotation theta:
    # S = max over all points P and all 6 normals n_k of dot(P-C, n_k) / (sqrt(3)/2)
    #   = (2/sqrt(3)) * max_{P, k} dot(P-C, n_k)
    # 
    # But we also need dot(P-C, n_k) >= -S*sqrt(3)/2, i.e., the constraint is |dot(P-C, n_k)| <= S*sqrt(3)/2?
    # No, the hexagon is not symmetric about the origin in the sense that face normals point outward.
    # The constraint is: dot(P-C, n_k) <= S*sqrt(3)/2 for all k.
    # This automatically handles the "other side" because n_{k+3} = -n_k.
    # So the constraint is: for all k=0..5, dot(P-C, n_k) <= S*sqrt(3)/2.
    # Which is equivalent to: max_k dot(P-C, n_k) <= S*sqrt(3)/2 for all P.
    # So S = (2/sqrt(3)) * max_P max_k dot(P-C, n_k)
    # 
    # But wait, this only ensures P is on the correct side of each face. Since the hexagon is the intersection
    # of 6 half-planes, this is correct.

    # n_k: 6 outward face normals. If vertex at angle 0, then face normal at 30, 90, 150, 210, 270, 330.
    FACE_NORMALS = np.array([[math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))] for k in range(6)])
    
    # For given container center C and rotation theta, the normals are rotated by theta.
    # We'll handle rotation in the caller.
    
    # For now, compute with fixed normals (rotation=0), then we can rotate points instead.
    # Actually, let's just compute: for each point P, compute max over k of dot(P, n_k).
    # Then S = (2/sqrt(3)) * max over all P of that value, but shifted by center C.
    # dot(P - C, n_k) = dot(P, n_k) - dot(C, n_k)
    # max_k [dot(P, n_k) - dot(C, n_k)] = max_k dot(P, n_k) - dot(C, n_k) ... no, can't separate max.
    # 
    # Let me just compute directly.
    # points: (72, 2), C: (2,)
    diff = points - C  # (72, 2)
    # (72, 6)
    projections = diff @ FACE_NORMALS.T
    max_proj = projections.max(axis=1)  # (72,)
    max_over_all = max_proj.max()
    S = max_over_all / (math.sqrt(3) / 2)
    # Also need to ensure no negative issues - the hexagon is convex, the constraint is one-sided.
    # But if all projections are very negative, that means points are far in the -n direction,
    # which is handled by the opposite normal. So this should be fine.
    return S

def compute_all_vertices(positions, angles):
    """positions: (12,2), angles: (12,) in radians. Returns (72,2) array of all vertices."""
    # For each hexagon, vertices = center + R(angle) @ VERTEX_OFFSETS
    # VERTEX_OFFSETS: (6,2)
    # R(angle) is 2x2 rotation
    verts = np.zeros((12, 6, 2))
    for i in range(12):
        c, s = math.cos(angles[i]), math.sin(angles[i])
        R = np.array([[c, -s], [s, c]])
        verts[i] = positions[i] + R @ VERTEX_OFFSETS.T  # (6,2)
    return verts.reshape(72, 2)

def objective(params):
    """params: (36,) = [x1,y1,th1, x2,y2,th2, ...]"""
    positions = params[0::3].reshape(12, 2)
    angles = params[1::3]  # Wait, this isn't right.
    # Let me restructure: params = [x1,y1,th1, x2,y2,th2, ..., x12,y12,th12]
    positions = params.reshape(12, 3)[:, :2]
    angles = params.reshape(12, 3)[:, 2]
    
    # Container center is the centroid of all inner hexagon centers
    C = positions.mean(axis=0)
    
    verts = compute_all_vertices(positions, angles)
    return container_side(verts - C + C)  # just container_side(verts) but with C
    
    # Actually container_side takes points and C separately. Let me fix.

def objective_v2(params):
    """params: (36,)"""
    P = params.reshape(12, 3)
    positions = P[:, :2]
    angles = P[:, 2]
    
    C = positions.mean(axis=0)
    
    verts = compute_all_vertices(positions, angles)
    
    # Compute container side with center C
    FACE_NORMALS = np.array([[math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))] for k in range(6)])
    diff = verts - C
    projections = diff @ FACE_NORMALS.T
    max_proj = projections.max(axis=1).max()
    return max_proj / (math.sqrt(3) / 2)

# Honeycomb initial configuration
def honeycomb_init():
    """12 hexagons in a honeycomb pattern."""
    # Honeycomb: hexagons tile the plane with centers at distance 2*sqrt(3) apart in a triangular lattice.
    # For side-1 hexagons, adjacent centers are 2*sqrt(3) apart.
    # Let's place them in a 3x4 or similar arrangement.
    # Actually, for a compact cluster of 12, let's use a hexagonal arrangement:
    # Center + 6 around + 5 more, or a 3-4-5 or 4-4-4 strip.
    
    # Let's try a compact hexagonal cluster: 1 center, 6 ring, 5 of second ring
    # Or simpler: a 3x4 rectangular grid in triangular lattice coords.
    
    # Triangular lattice basis: a1 = (2*sqrt(3), 0), a2 = (sqrt(3), 3)
    a1 = np.array([2*math.sqrt(3), 0])
    a2 = np.array([math.sqrt(3), 3])
    
    # Place 12 in a 3x4 grid
    positions = []
    for i in range(3):
        for j in range(4):
            pos = i * a1 + j * a2
            positions.append(pos)
    
    positions = np.array(positions)
    # Center them
    positions -= positions.mean(axis=0)
    
    # All angles 0 (pointy-top aligned)
    angles = np.zeros(12)
    
    params = np.zeros(36)
    params.reshape(12, 3)[:, :2] = positions
    params.reshape(12, 3)[:, 2] = angles
    return params

def sa_optimize(params0, time_limit):
    """Simulated annealing with adaptive step sizes."""
    params = params0.copy()
    current_obj = objective_v2(params)
    best_params = params.copy()
    best_obj = current_obj
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initial step sizes
    step_pos = np.array([0.5, 0.5, 0.1]) * np.ones(12)  # per-dim steps
    temp = 0.5
    
    n_iter = 0
    while time.time() < deadline - 5.0:  # 5s safety margin
        # Perturb
        delta = np.random.normal(0, 1, 36)
        # Apply different scales to pos vs angle
        delta[0::3] *= step_pos[:, 0]
        delta[1::3] *= step_pos[:, 1]
        delta[2::3] *= step_pos[:, 2]
        
        new_params = params + delta
        new_obj = objective_v2(new_params)
        
        if new_obj < current_obj or np.random.random() < math.exp(-(new_obj - current_obj) / max(temp, 1e-10)):
            params = new_params
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = params.copy()
        
        # Adaptive step size: increase if acceptance high, decrease if low
        # Simple: every 100 iters, adjust
        n_iter += 1
        if n_iter % 100 == 0:
            temp *= 0.995
            # Keep step sizes roughly constant or slowly decrease
            step_pos *= 0.999
    
    return best_params, best_obj

def nelder_mead_polish(params, time_limit):
    """Nelder-Mead on the 36-dim problem."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    def obj(x):
        return objective_v2(x)
    
    res = minimize(obj, params, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if res.success or res.fun < objective_v2(params):
        return res.x, res.fun
    return params, objective_v2(params)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Honeycomb initialization
    params0 = honeycomb_init()
    obj0 = objective_v2(params0)
    
    best_params = params0
    best_obj = obj0
    
    # SA phase: use ~120s
    sa_time = 120.0
    if time.time() > deadline - sa_time - 10:
        sa_time = max(10.0, deadline - time.time() - 10)
    
    params_sa, obj_sa = sa_optimize(params0, sa_time)
    if obj_sa < best_obj:
        best_obj = obj_sa
        best_params = params_sa
    
    # Nelder-Mead polish: use remaining time
    nm_time = max(5.0, deadline - time.time() - 2.0)
    if nm_time > 5.0:
        params_nm, obj_nm = nelder_mead_polish(best_params, nm_time)
        if obj_nm < best_obj:
            best_obj = obj_nm
            best_params = params_nm
    
    # Extract result
    P = best_params.reshape(12, 3)
    positions = P[:, :2]
    angles_rad = P[:, 2]
    
    # Inner hexagons: (x, y, angle_degrees)
    inner = [[float(positions[i, 0]), float(positions[i, 1]), float(math.degrees(angles_rad[i]))] for i in range(12)]
    
    # Container center = centroid
    C = positions.mean(axis=0)
    side = float(best_obj)
    rotation = 0.0  # We assumed rotation 0 for container
    
    return inner, (float(C[0]), float(C[1])), side, rotation
