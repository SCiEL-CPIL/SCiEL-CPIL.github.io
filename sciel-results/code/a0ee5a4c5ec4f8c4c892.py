"""Packing 12 unit regular hexagons into a regular hexagon, minimizing outer side length."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 115.0  # seconds, safety margin before 120s limit
_START = time.time()

def _check_deadline():
    return (time.time() - _START) > DEADLINE

def _wrap60(a):
    """Wrap angle to [0, 60) degrees."""
    a = a % 60.0
    return a

def _hex_support(direction_deg, rotation_deg):
    """
    Support function of a unit regular hexagon (circumradius 1) in a given direction.
    Returns the distance from center to boundary in that direction.
    For a unit hexagon, the apothem is sqrt(3)/2.
    """
    delta = _wrap60(direction_deg - rotation_deg)
    # The hexagon has vertices at rotation + k*60, and flat normals at rotation + 30 + k*60
    # In the range [0, 60), the boundary distance formula:
    # r(phi) = apothem / cos(phi - 30) where phi is in [-30, 30] relative to a flat normal
    # Equivalently, let d = delta, then the relevant angle is:
    # If delta in [0, 30]: angle from vertex = delta, distance = 1.0 / cos(delta * pi/180) ... no.
    # Let me re-derive:
    # Vertex directions at rotation + k*60. Flat normals at rotation + 30 + k*60.
    # For a direction phi, let delta = wrap60(phi - rotation).
    # delta in [0, 60). The closest vertex or flat to this direction:
    # If delta in [0, 30]: we're between a vertex (at delta=0) and a flat normal (at delta=30).
    # The support is: h(delta) = 1.0 / cos(delta * np.pi / 180) -- NO, that's not right either.
    # 
    # Actually, for a regular hexagon with circumradius R=1:
    # The boundary in polar form (relative to a vertex direction):
    # For angle theta from a vertex direction, in range [-30, 30]:
    # r(theta) = cos(30 deg) / cos(theta) = (sqrt(3)/2) / cos(theta)
    # 
    # So if delta is in [0, 30], theta = delta (from vertex at 0)
    # If delta is in [30, 60], theta = delta - 60 (from vertex at 60)
    if delta <= 30.0:
        theta = delta
    else:
        theta = delta - 60.0
    theta_rad = np.radians(theta)
    return (np.sqrt(3) / 2.0) / np.cos(theta_rad)

def _point_in_hexagon(px, py, cx, cy, rotation_deg):
    """
    Check if point (px, py) is inside a regular hexagon centered at (cx, cy)
    with circumradius 1 and rotation rotation_deg (in degrees).
    The hexagon has vertices at rotation + k*60 for k=0..5.
    """
    dx = px - cx
    dy = py - cy
    # Convert to hexagon's local frame
    rot_rad = np.radians(rotation_deg)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    # Now check if (lx, ly) is inside unit hexagon centered at origin with vertex at angle 0
    # The hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # The sides connect consecutive vertices.
    # A point is inside if it's on the correct side of each edge.
    # Vertices:
    verts = []
    for k in range(6):
        ang = np.radians(k * 60.0)
        verts.append((np.cos(ang), np.sin(ang)))
    # For each edge, check if point is on the inner side
    for k in range(6):
        v1 = verts[k]
        v2 = verts[(k + 1) % 6]
        # Edge from v1 to v2. The interior is to the left (for CCW winding).
        # Normal pointing inward: rotate edge vector by -90 deg? Let's compute.
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        # The inward normal for a CCW polygon edge is (ey, -ex) normalized... 
        # Actually for a regular hexagon centered at origin, the inward normal of edge k
        # points toward the center. The midpoint of edge k is at angle 30 + k*60.
        # The inward normal direction is -midpoint_normalized.
        mx = (v1[0] + v2[0]) / 2.0
        my = (v1[1] + v2[1]) / 2.0
        mnorm = np.sqrt(mx * mx + my * my)
        nx = -mx / mnorm
        ny = -my / mnorm
        # Distance from point to the line through v1 with normal (nx, ny):
        dist = nx * (lx - v1[0]) + ny * (ly - v1[1])
        if dist > 1e-10:
            return False
    return True

def _hexagon_overlap(p1, a1, p2, a2):
    """
    Conservative check: do two unit hexagons at (p1, a1) and (p2, a2) overlap?
    We use the condition that if the distance between centers is less than sqrt(3) (2*apothem),
    they might overlap. For a conservative (safe) check, we say they overlap if dist < sqrt(3).
    But this is only correct for aligned hexagons. For rotated hexagons, the minimum
    distance for non-overlap can be different.
    
    For safety, we'll use a tighter check: the sum of the support functions in the direction
    between centers must be <= the distance.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.sqrt(dx * dx + dy * dy)
    if dist < 1e-12:
        return True
    direction_deg = np.degrees(np.arctan2(dy, dx))
    # Support of hex1 in direction of hex2, and hex2 in direction of hex1
    h1 = _hex_support(direction_deg, a1)
    h2 = _hex_support(direction_deg + 180.0, a2)
    return dist < h1 + h2 - 1e-9

def _outer_side_length(positions, angles):
    """
    Compute the minimum side length S of an outer regular hexagon (circumradius S)
    centered at origin that contains all inner hexagons.
    For each inner hexagon at position p with rotation a, the support in direction phi
    is |p| in that direction + hex_support(phi, a).
    The outer hexagon with side length S has support function S * hex_support(phi, 0)
    (since a regular hexagon of circumradius S has support S * hex_support(phi, 0)).
    We need S * hex_support(phi, 0) >= |p|_phi + hex_support(phi, a) for all phi.
    So S >= (|p|_phi + hex_support(phi, a)) / hex_support(phi, 0) for all p, a, phi.
    We take the max over all inner hexagons and a set of directions.
    """
    max_ratio = 0.0
    # Sample many directions
    for phi_deg in np.arange(0, 360, 5.0):
        phi_rad = np.radians(phi_deg)
        outer_support = _hex_support(phi_deg, 0.0)  # hex_support for unit hexagon
        if outer_support < 1e-10:
            continue
        for i in range(len(positions)):
            p = positions[i]
            a = angles[i]
            # Projection of p in direction phi
            proj = p[0] * np.cos(phi_rad) + p[1] * np.sin(phi_rad)
            abs_proj = abs(proj)
            inner_support = _hex_support(phi_deg, a)
            ratio = (abs_proj + inner_support) / outer_support
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio

def _objective_and_overlap(x, positions_init=None):
    """
    x = [x1, y1, a1, x2, y2, a2, ...] where ai are in radians.
    Returns (objective, max_overlap) where objective is outer side length.
    """
    n = len(x) // 3
    positions = []
    angles_deg = []
    for i in range(n):
        px, py, arad = x[3*i], x[3*i+1], x[3*i+2]
        positions.append([px, py])
        angles_deg.append(np.degrees(arad))
    
    # Check overlaps
    max_overlap = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if _hexagon_overlap(positions[i], angles_deg[i], positions[j], angles_deg[j]):
                # Approximate overlap depth
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                direction_deg = np.degrees(np.arctan2(dy, dx))
                h1 = _hex_support(direction_deg, angles_deg[i])
                h2 = _hex_support(direction_deg + 180.0, angles_deg[j])
                overlap = h1 + h2 - dist
                if overlap > max_overlap:
                    max_overlap = overlap
    
    S = _outer_side_length(positions, angles_deg)
    return S, max_overlap

def construct():
    global _START
    _START = time.time()
    
    best_S = float('inf')
    best_solution = None
    
    # Initial positions: hexagonal lattice cluster
    # 12 hexagons in a compact arrangement
    # A 1-2-3-2-1-1 arrangement or similar
    # Let's use a roughly circular cluster
    base_positions = []
    # Center
    base_positions.append((0.0, 0.0))
    # First ring: 6 hexagons at distance 2*sqrt(3)/2 = sqrt(3) apart (touching)
    for k in range(6):
        ang = np.radians(k * 60.0 + 30.0)
        r = np.sqrt(3.0)  # distance for touching hexagons
        base_positions.append((r * np.cos(ang), r * np.sin(ang)))
    # Second ring: 5 more hexagons
    for k in range(5):
        ang = np.radians(k * 72.0 + 10.0)
        r = 2.0 * np.sqrt(3.0)
        base_positions.append((r * np.cos(ang), r * np.sin(ang)))
    
    rng = np.random.default_rng(42)
    
    n_restarts = 0
    max_restarts = 30
    
    while n_restarts < max_restarts:
        if _check_deadline():
            break
        
        # Initialize with some randomness
        x0 = []
        for (px, py) in base_positions:
            px += rng.uniform(-0.2, 0.2)
            py += rng.uniform(-0.2, 0.2)
            a = rng.uniform(-np.pi/6, np.pi/6)  # small random rotation
            x0.extend([px, py, a])
        x0 = np.array(x0)
        
        # Optimize
        def obj_func(x):
            S, overlap = _objective_and_overlap(x)
            # Penalty for overlap
            penalty = 100.0 * max(overlap, 0.0) ** 2
            return S + penalty
        
        best_local_S = float('inf')
        best_local_x = None
        
        # Try Nelder-Mead first (doesn't need gradient)
        try:
            res = minimize(obj_func, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_local_S:
                S, overlap = _objective_and_overlap(res.x)
                if overlap < 1e-6:
                    best_local_S = S
                    best_local_x = res.x.copy()
                elif S < best_local_S:
                    # Keep as candidate even with small overlap for further polishing
                    best_local_S = S
                    best_local_x = res.x.copy()
        except Exception:
            pass
        
        # Try L-BFGS-B as well
        if not _check_deadline():
            try:
                def grad_func(x):
                    eps = 1e-7
                    grad = np.zeros_like(x)
                    f0 = obj_func(x)
                    for i in range(len(x)):
                        xp = x.copy()
                        xp[i] += eps
                        grad[i] = (obj_func(xp) - f0) / eps
                    return grad
                
                res2 = minimize(obj_func, x0, method='L-BFGS-B', jac=grad_func,
                               options={'maxiter': 1000, 'ftol': 1e-12})
                if res2.fun < best_local_S:
                    S, overlap = _objective_and_overlap(res2.x)
                    if overlap < 1e-6:
                        best_local_S = S
                        best_local_x = res2.x.copy()
                    elif S < best_local_S:
                        best_local_S = S
                        best_local_x = res2.x.copy()
            except Exception:
                pass
        
        if best_local_x is not None:
            S, overlap = _objective_and_overlap(best_local_x)
            if S < best_S:
                best_S = S
                best_solution = best_local_x.copy()
        
        n_restarts += 1
    
    # If no valid solution found, return a valid but large solution
    if best_solution is None or _check_deadline():
        # Return the initial lattice positions with a large outer hexagon
        positions = []
        angles = []
        for (px, py) in base_positions:
            positions.append([px, py])
            angles.append(0.0)
        S = _outer_side_length(positions, angles)
        # Make sure it's valid by scaling up if needed
        inner = []
        for p, a in zip(positions, angles):
            inner.append([p[0], p[1], a])
        return inner, (0.0, 0.0), max(S + 0.1, 40.0), 0.0
    
    # Extract solution
    n = len(best_solution) // 3
    inner = []
    for i in range(n):
        px = best_solution[3*i]
        py = best_solution[3*i+1]
        a_deg = np.degrees(best_solution[3*i+2])
        inner.append([px, py, a_deg])
    
    # Final verification and polishing
    positions = [[inner[i][0], inner[i][1]] for i in range(n)]
    angles = [inner[i][2] for i in range(n)]
    
    # Check for overlaps and adjust S
    S = _outer_side_length(positions, angles)
    
    # Small polish: try to reduce S slightly by adjusting positions
    if not _check_deadline():
        try:
            x_polish = np.array(best_solution)
            for _ in range(50):
                if _check_deadline():
                    break
                res_p = minimize(obj_func, x_polish, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                S_p, overlap_p = _objective_and_overlap(res_p.x)
                if overlap_p < 1e-8 and S_p < S:
                    S = S_p
                    best_solution = res_p.x.copy()
                    x_polish = res_p.x.copy()
                else:
                    break
        except Exception:
            pass
    
    # Re-extract if polished
    if best_solution is not None:
        n = len(best_solution) // 3
        inner = []
        for i in range(n):
            px = best_solution[3*i]
            py = best_solution[3*i+1]
            a_deg = np.degrees(best_solution[3*i+2])
            inner.append([px, py, a_deg])
        S = _outer_side_length([[inner[i][0], inner[i][1]] for i in range(n)],
                              [inner[i][2] for i in range(n)])
    
    return inner, (0.0, 0.0), S, 0.0
