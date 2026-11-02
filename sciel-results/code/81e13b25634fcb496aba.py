import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 142.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _hex_support(cx, cy, angle_deg, theta):
    """Support function of unit hexagon: max dot product with direction theta."""
    a0 = math.radians(angle_deg)
    # For a regular hexagon with circumradius 1, the support function in direction theta
    # is 1 * max over vertices of cos(vertex_angle - theta)
    # But more efficiently: for a regular hexagon, support = cos(min angular distance to a vertex normal)
    # The 6 vertex directions are a0 + k*pi/3
    # Support in direction theta = max_k cos(a0 + k*pi/3 - theta)
    # This equals cos(min_k |a0 + k*pi/3 - theta| mod 2pi, wrapped)
    diff = (theta - a0) % (2 * math.pi)
    # The 6 directions are at 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3 relative to a0
    # The minimum angular distance to any of these:
    k = round(diff / (math.pi / 3.0))
    d1 = abs(diff - k * math.pi / 3.0)
    d2 = abs(diff - (k - 1) * math.pi / 3.0)
    d3 = abs(diff - (k + 1) * math.pi / 3.0)
    min_d = min(d1, d2, d3)
    return math.cos(min_d)


def _hex_distance(d, angle1, angle2):
    """Minimum distance between two unit hexagons with centers separated by vector d, 
    rotations angle1, angle2. Returns 0 if overlapping, positive if separated."""
    # Two convex polygons are disjoint iff there exists a separating axis.
    # For two regular hexagons, the separating axes are the normals to all edges.
    # Edge normals of hexagon with rotation a: a + pi/6 + k*pi/3 for k=0..5
    # Distance along axis n: |d·n| - support1(n) - support2(n)
    # If for all axes this is >= 0, they're disjoint.
    # The minimum distance is the max over axes of (support1 + support2 - |d·n|) ... no wait.
    # Actually: they overlap iff for ALL axes n: |d·n| < support1(n) + support2(n)
    # They are disjoint iff there EXISTS an axis n: |d·n| >= support1(n) + support2(n)
    # The minimum separation = min over axes of (|d·n| - support1(n) - support2(n))
    # If this min > 0, they're separated by that amount.
    # If this min < 0, they overlap (penetration depth = -min).
    
    a1 = math.radians(angle1)
    a2 = math.radians(angle2)
    d = np.array(d)
    
    # Try all 12 edge normals (6 from each hexagon)
    min_sep = float('inf')
    for k in range(6):
        # Normals from hexagon 1
        theta1 = a1 + math.pi/6.0 + k * math.pi/3.0
        proj = abs(np.dot(d, [math.cos(theta1), math.sin(theta1)]))
        s1 = _hex_support(0, 0, angle1, theta1)
        s2 = _hex_support(0, 0, angle2, theta1)
        sep = proj - s1 - s2
        if sep < min_sep:
            min_sep = sep
        
        # Normals from hexagon 2
        theta2 = a2 + math.pi/6.0 + k * math.pi/3.0
        proj = abs(np.dot(d, [math.cos(theta2), math.sin(theta2)]))
        s1 = _hex_support(0, 0, angle1, theta2)
        s2 = _hex_support(0, 0, angle2, theta2)
        sep = proj - s1 - s2
        if sep < min_sep:
            min_sep = sep
    
    return min_sep


def _enclosing_hex_side(verts, outer_angle_deg):
    """Find minimum side length of regular hexagon with given orientation containing all vertices."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return side
    except Exception:
        pass
    
    # Fallback: centroid-based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return side


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3-4-4 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # 2-4-2-3 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        # Center + 6 ring + 4 more
        positions = []
        positions.append((0, 0))
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        # 2-3-3-3 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        # 4-4-3 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        # 5-6 pattern
        positions = []
        y1 = -0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        # 6-5 pattern
        positions = []
        y1 = -0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        # Diamond-ish
        positions = []
        y0 = -3.0*scale
        positions.append((0, y0))
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        y4 = 3.0*scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        # 3-3-5 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: compact grid
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]


def _objective(params, inner_angle):
    """Objective: enclosing hex side length + penalty for overlaps."""
    # params: [x0,y0, x1,y1, ..., x10,y10, outer_angle_deg]
    # Or we fix outer angle and only optimize positions
    n_pos = N * 2
    positions = [(params[i], params[i+1]) for i in range(0, n_pos, 2)]
    outer_angle = params[-1]
    
    # Get all vertices
    all_vertices = []
    for px, py in positions:
        verts = _hex_vertices(px, py, inner_angle)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Enclosing hex side
    side = _enclosing_hex_side(all_vertices, outer_angle)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            di = np.array([positions[j][0] - positions[i][0], positions[j][1] - positions[i][1]])
            dist = _hex_distance(di, inner_angle, inner_angle)
            if dist < 0:
                penalty += dist * dist * 100.0
    
    return side + penalty


def _optimize_layout(positions, inner_angle, outer_angles):
    """Optimize a layout by trying different outer angles and refining positions."""
    best_side = float('inf')
    best_pos = None
    best_outer = 0.0
    
    for outer_angle in outer_angles:
        # Compute enclosing hex for this angle
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        side = _enclosing_hex_side(all_vertices, outer_angle)
        
        if side < best_side:
            best_side = side
            best_pos = positions
            best_outer = outer_angle
    
    # Try to optimize positions with Nelder-Mead
    if time.time() < _DEADLINE - 15.0:
        # Fix the best outer angle and optimize positions
        x0 = np.array([positions[i][0] for i in range(N)] + 
                      [positions[i][1] for i in range(N)] + [best_outer])
        
        def obj(params):
            pos = [(params[i], params[i+1]) for i in range(0, N*2, 2)]
            oa = params[-1]
            all_v = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in pos])
            s = _enclosing_hex_side(all_v, oa)
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    di = np.array([pos[j][0]-pos[i][0], pos[j][1]-pos[i][1]])
                    d = _hex_distance(di, inner_angle, inner_angle)
                    if d < 0:
                        pen += d*d*100.0
            return s + pen
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side:
                best_side = res.fun
                best_pos = [(res.x[i], res.x[i+1]) for i in range(0, N*2, 2)]
                best_outer = res.x[-1]
        except Exception:
            pass
    
    return best_side, best_pos, best_outer


def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Outer angles to try
    outer_angles = list(np.arange(0, 30.0, 1.0))
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0
        
        # First pass: try discrete outer angles
        best_local_side = float('inf')
        best_local_pos = positions
        best_local_angle = 0.0
        
        for outer_angle_deg in outer_angles:
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            side = _enclosing_hex_side(all_vertices, outer_angle_deg)
            if side < best_local_side:
                best_local_side = side
                best_local_pos = positions
                best_local_angle = outer_angle_deg
        
        # Second pass: refine with local optimization
        if time.time() < _DEADLINE - 15.0:
            refined_side, refined_pos, refined_angle = _optimize_layout(
                best_local_pos, inner_angle, 
                list(np.arange(best_local_angle - 5, best_local_angle + 5, 0.5)))
            
            if refined_side < best_local_side:
                best_local_side = refined_side
                best_local_pos = refined_pos
                best_local_angle = refined_angle
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in best_local_pos]
            best_center = (0.0, 0.0)  # Will be recomputed
            best_outer_angle = best_local_angle
    
    # Compute proper center for the best configuration
    if best_inner is not None:
        all_vertices = np.vstack([_hex_vertices(p[0], p[1], p[2]) for p in best_inner])
        # Find center via LP
        base_angle = math.radians(best_outer_angle + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        M = len(all_vertices)
        A_rows = []
        b_vals = []
        for i in range(M):
            vi = all_vertices[i]
            for k in range(6):
                nk = normals[k]
                A_rows.append([-nk[0], -nk[1], -1.0])
                b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
                A_rows.append([nk[0], nk[1], -1.0])
                b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.status == 0:
                best_center = (res.x[0], res.x[1])
                best_side = res.x[2] * 2.0 / _SQRT3
        except Exception:
            pass
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
