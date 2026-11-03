"""Packing 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12
DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 + rotation
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    verts = np.zeros((6, 2))
    for i in range(6):
        a = base_angles[i] + angle
        verts[i, 0] = cx + s * np.cos(a)
        verts[i, 1] = cy + s * np.sin(a)
    return verts

def all_inner_vertices(positions, angles, s=1.0):
    """Return (6N, 2) array of all vertices of all inner hexagons."""
    verts_list = []
    for i in range(len(positions)):
        cx, cy = positions[i]
        ang = angles[i]
        verts_list.append(hex_vertices(cx, cy, s, ang))
    return np.vstack(verts_list)

def min_outer_hex_side(vertices):
    """
    Find the minimum side length of a regular hexagon (any rotation, centered at origin WLOG)
    that contains all given vertices.
    We optimize over the outer hexagon's rotation angle.
    For a regular hexagon with side s and rotation phi, a point (x,y) is inside iff
    for all 6 directions k: dot(point, dir_k) <= s where dir_k are unit vectors at angles phi + k*60 deg.
    Actually, the support function: max over 6 directions of dot(p, d_k) <= s.
    So s_min = max_k max_i dot(v_i, d_k) where d_k are the 6 outward normals.
    """
    def objective(phi):
        base = np.radians(phi)
        dirs = np.array([[np.cos(base + k * np.pi / 3), np.sin(base + k * np.pi / 3)] for k in range(6)])
        # For each direction, find max projection
        projs = vertices @ dirs.T  # (Npts, 6)
        max_projs = projs.max(axis=0)  # (6,)
        return max_projs.max()
    
    # Try many rotations
    best_s = np.inf
    best_phi = 0
    for phi in np.linspace(0, 60, 61):
        s_val = objective(phi)
        if s_val < best_s:
            best_s = s_val
            best_phi = phi
    # Refine around best
    if best_s < np.inf:
        res = minimize(objective, [best_phi], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200})
        if res.success or res.fun < best_s:
            best_s = res.fun
            best_phi = res.x[0]
    return best_s, best_phi

def check_overlap(pos1, ang1, pos2, ang2, s1=1.0, s2=1.0, tol=1e-7):
    """Check if two hexagons overlap. Simple: check if any vertex of one is inside the other,
    or if edges intersect. For regular hexagons of same size, we can use a simpler check."""
    # Get vertices
    v1 = hex_vertices(pos1[0], pos1[1], s1, ang1)
    v2 = hex_vertices(pos2[0], pos2[1], s2, ang2)
    
    # Check if any vertex of v1 is strictly inside v2, or vice versa
    # For a regular hexagon, a point is inside if it's within distance <= s from center AND
    # satisfies all 6 half-plane constraints. But simpler: use winding or just check
    # if min distance between edges is < 0.
    
    # Simple check: if centers are farther than 2*s apart, no overlap (for same-size hexes)
    d = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
    if d > 2.0 * max(s1, s2) + tol:
        return False
    
    # Check if any vertex of v1 is inside v2
    def point_in_hex(px, py, cx, cy, s, ang_deg):
        ang = np.radians(ang_deg)
        # Transform point to hex's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -ang
        c, sn = np.cos(-ang), np.sin(-ang)
        lx = c * dx - sn * dy
        ly = sn * dx + c * dy
        # Regular hexagon with flat top: vertices at angles 0,60,...
        # The hexagon is intersection of 6 half-planes: |dot(p, d_k)| <= s * cos(30) = s * sqrt(3)/2
        # where d_k are directions at 0, 60, 120, 180, 240, 300
        # Actually for a regular hexagon with side s, the apothem is s*sqrt(3)/2
        # A point is inside if for all k: |dot(p_local, d_k)| <= apothem
        apothem = s * np.sqrt(3) / 2
        for k in range(6):
            a = k * np.pi / 3
            d = np.array([np.cos(a), np.sin(a)])
            proj = lx * d[0] + ly * d[1]
            if abs(proj) > apothem + tol:
                return False
        return True
    
    for vx, vy in v1:
        if point_in_hex(vx, vy, pos2[0], pos2[1], s2, ang2):
            return True
    for vx, vy in v2:
        if point_in_hex(vx, vy, pos1[0], pos1[1], s1, ang1):
            return True
    return False

def packing_cost(positions, angles):
    """Cost = outer side length + penalty for overlaps."""
    verts = all_inner_vertices(positions, angles)
    # Center all vertices at origin for computing enclosing hex
    center = verts.mean(axis=0)
    verts_centered = verts - center
    s_outer, _ = min_outer_hex_side(verts_centered)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if check_overlap(positions[i], angles[i], positions[j], angles[j]):
                penalty += 10.0
    return s_outer + penalty, s_outer

def initial_lattice():
    """Hexagonal lattice arrangement."""
    positions = []
    angles = []
    # 12 hexagons in a hexagonal pattern: center + 6 around + 5 more
    # Let's do a 3-4-3 or similar arrangement
    # Actually, let's use a simple linear arrangement first
    for i in range(N):
        positions.append([2.0 * i, 0.0])
        angles.append(0.0)
    return positions, angles

def initial_ring():
    """Ring arrangement: 1 center + 11 around."""
    positions = [[0.0, 0.0]]
    angles = [0.0]
    # 11 around in a circle
    r = 2.0  # approximate distance for touching
    for i in range(11):
        a = 2 * np.pi * i / 11
        positions.append([r * np.cos(a), r * np.sin(a)])
        angles.append(0.0)
    return positions, angles

def initial_random(seed=42):
    rng = random.Random(seed)
    positions = []
    angles = []
    for i in range(N):
        positions.append([rng.uniform(-5, 5), rng.uniform(-5, 5)])
        angles.append(rng.uniform(0, 60))
    return positions, angles

def simulated_annealing(positions, angles, n_iter=5000, T_start=1.0, T_end=0.01):
    """Simulated annealing to minimize packing cost."""
    positions = [list(p) for p in positions]
    angles = list(angles)
    
    cost, s_outer = packing_cost(positions, angles)
    best_positions = [list(p) for p in positions]
    best_angles = list(angles)
    best_cost = cost
    best_s = s_outer
    
    for it in range(n_iter):
        if time.time() > DEADLINE:
            break
        T = T_start * (T_end / T_start) ** (it / max(n_iter, 1))
        
        # Perturb a random hexagon
        i = random.randint(0, N-1)
        new_positions = [list(p) for p in positions]
        new_angles = list(angles)
        
        delta_r = 0.5
        new_positions[i][0] += random.gauss(0, delta_r)
        new_positions[i][1] += random.gauss(0, delta_r)
        new_angles[i] = (new_angles[i] + random.gauss(0, 5)) % 60
        
        new_cost, new_s = packing_cost(new_positions, new_angles)
        
        if new_cost < cost or random.random() < np.exp(-(new_cost - cost) / T):
            positions = new_positions
            angles = new_angles
            cost = new_cost
            s_outer = new_s
            
            if cost < best_cost:
                best_cost = cost
                best_s = new_s
                best_positions = [list(p) for p in positions]
                best_angles = list(angles)
    
    return best_positions, best_angles, best_s

def polish(positions, angles):
    """Use scipy minimize to polish the solution."""
    # Flatten: [x0, y0, a0, x1, y1, a1, ...]
    x0 = []
    for i in range(N):
        x0.extend([positions[i][0], positions[i][1], angles[i]])
    
    def obj(x):
        pos = []
        ang = []
        for i in range(N):
            pos.append([x[3*i], x[3*i+1]])
            ang.append(x[3*i+2])
        c, s = packing_cost(pos, ang)
        return c
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success or res.fun < obj(x0):
            pos = []
            ang = []
            for i in range(N):
                pos.append([res.x[3*i], res.x[3*i+1]])
                ang.append(res.x[3*i+2])
            c, s = packing_cost(pos, ang)
            if s < obj(x0):
                return pos, ang, s
    except:
        pass
    return positions, angles, None

def construct():
    global DEADLINE
    DEADLINE = time.time() + 130.0
    
    best_s = np.inf
    best_positions = None
    best_angles = None
    
    # Try multiple initial configurations
    inits = [
        ("lattice", initial_lattice),
        ("ring", initial_ring),
        ("random1", lambda: initial_random(42)),
        ("random2", lambda: initial_random(123)),
        ("random3", lambda: initial_random(999)),
    ]
    
    for name, init_fn in inits:
        if time.time() > DEADLINE - 10:
            break
        try:
            positions, angles = init_fn()
            # Simulated annealing
            pos_sa, ang_sa, s_sa = simulated_annealing(positions, angles, n_iter=3000)
            if s_sa < best_s:
                best_s = s_sa
                best_positions = pos_sa
                best_angles = ang_sa
            
            # Polish
            if time.time() > DEADLINE - 5:
                break
            pos_pol, ang_pol, s_pol = polish(pos_sa, ang_sa)
            if s_pol is not None and s_pol < best_s:
                best_s = s_pol
                best_positions = pos_pol
                best_angles = ang_pol
        except Exception as e:
            pass
    
    if best_positions is None:
        # Fallback
        best_positions, best_angles = initial_lattice()
        best_s = 40.0
    
    # Compute the outer hexagon center and rotation
    verts = all_inner_vertices(best_positions, best_angles)
    center = verts.mean(axis=0)
    verts_centered = verts - center
    s_final, phi_final = min_outer_hex_side(verts_centered)
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (float(center[0]), float(center[1])), float(s_final), float(phi_final)
