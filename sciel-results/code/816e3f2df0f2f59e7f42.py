import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    # Normals to the 6 sides of the outer hexagon
    # Vertices at outer_angle + 60k => edge normals at outer_angle + 30 + 60k
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    # Variables: [cx, cy, R], minimize R
    # Constraints for each vertex v and normal n:
    #   (v - c).n <= R  =>  -n.c - R <= -v.n
    #   (c - v).n <= R  =>   n.c - R <=  v.n
    
    A = []
    b = []
    for i in range(M):
        vx, vy = verts[i]
        for nx, ny in normals:
            vd = vx*nx + vy*ny
            A.append([-nx, -ny, -1.0])
            b.append(-vd)
            A.append([nx, ny, -1.0])
            b.append(vd)
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3/2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _polish_enclosing_hex(verts, outer_angle_deg, cx0, cy0, R0):
    """
    High-precision polish using SLSQP to refine (cx, cy, R) for a given outer angle.
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def objective(x):
        return x[2]
    
    def constraints(x):
        cx, cy, R = x
        cons = []
        for i in range(len(verts)):
            vx, vy = verts[i]
            for j in range(6):
                nx, ny = normals[j]
                # (v - c).n <= R  =>  -(v-c).n - R <= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, vx=vx, vy=vy, nx=nx, ny=ny: -((vx-x[0])*nx + (vy-x[1])*ny) - x[2]
                })
                # (c - v).n <= R  =>  (c-v).n - R <= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, vx=vx, vy=vy, nx=nx, ny=ny: ((x[0]-vx)*nx + (x[1]-vy)*ny) - x[2]
                })
        return cons
    
    try:
        res = minimize(objective, [cx0, cy0, R0], method='SLSQP',
                       constraints=constraints,
                       bounds=[(None, None), (None, None), (0, None)],
                       options={'maxiter': 100, 'ftol': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3/2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = math.sqrt(3)
    scale = 1.0000001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    # Pre-compute all (i,j) pairs in a reasonable range
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i*u[0] + j*v[0]
            y = i*u[1] + j*v[1]
            coords.append((x, y, i, j))
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        # Center (0,0), ring1: 6 neighbors, ring2: pick 4 to stay compact
        target = [(0,0)]
        # Ring 1
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        # Ring 2 - pick 4 that keep it compact
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        # Pick 4 closest to center
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        # Find these in coords
        pos_set = set((x,y) for x,y,i,j in coords)
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3 rows of 4,1,4 or similar
        # Use 2-3-3-3 or 4-3-4 pattern
        # Try: row at y=-1.5: 3 hexes, y=0: 4 hexes, y=1.5: 4 hexes
        positions = []
        # y = -1.5*scale, x offset
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale  # offset for interlocking
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
        # 4-3-4 but shifted
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
    else:
        # Fallback: sorted by distance
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return [(x,y) for x,y,i,j in coords_sorted[:N]]

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    layouts = [_build_positions(i) for i in range(4)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            all_vertices = np.array(all_vertices)
            
            # Coarse search
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            best_coarse_R = 0.0
            
            for outer_angle_deg in np.arange(0, 30.1, 1.0):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    R = side * (_SQRT3/2)
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
                        best_coarse_R = R
            
            # Fine search
            if best_coarse_angle > 0:
                for outer_angle_deg in np.arange(max(0, best_coarse_angle - 1.0),
                                                 min(30.1, best_coarse_angle + 1.0), 0.05):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        R = side * (_SQRT3/2)
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle_deg
                            best_coarse_center = (cx, cy)
                            best_coarse_R = R
            
            # High-precision polish
            if best_coarse_side < float('inf') and time.time() < _DEADLINE - 15.0:
                polished = _polish_enclosing_hex(all_vertices, best_coarse_angle,
                                                  best_coarse_center[0], best_coarse_center[1],
                                                  best_coarse_R)
                if polished:
                    cx, cy, side = polished
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_center = (cx, cy)
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
                best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
