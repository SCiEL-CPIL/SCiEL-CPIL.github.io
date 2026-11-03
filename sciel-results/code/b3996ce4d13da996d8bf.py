import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return verts

def min_hex_side(points, outer_angle_deg):
    """Minimal side of regular hexagon with given orientation containing all points."""
    a = np.radians(outer_angle_deg)
    normals = [np.array([np.cos(a + k * np.pi/3), np.sin(a + k * np.pi/3)]) for k in range(6)]
    P = np.array(points)
    max_span = 0
    for n in normals:
        proj = P @ n
        span = proj.max() - proj.min()
        if span > max_span:
            max_span = span
    return max_span / np.sqrt(3)

def construct():
    # Triangular lattice for hexagonal tiling
    # Adjacent centers distance = sqrt(3)
    # Basis: e1 = (sqrt(3), 0), e2 = (sqrt(3)/2, 3/2)
    e1 = np.array([np.sqrt(3), 0.0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Candidate arrangements as (i,j) lattice coords
    arrangements = []
    
    # 3-4-3-2
    a1 = []
    for row, cnt in enumerate([3,4,3,2]):
        off = (4 - cnt) / 2.0
        for col in range(cnt):
            a1.append((col + off, row))
    arrangements.append(a1)
    
    # 4-4-4
    a2 = [(col, row) for row in range(3) for col in range(4)]
    arrangements.append(a2)
    
    # 3-3-3-3
    a3 = [(col, row) for row in range(4) for col in range(3)]
    arrangements.append(a3)
    
    # 2-3-4-3
    a4 = []
    for row, cnt in enumerate([2,3,4,3]):
        off = (4 - cnt) / 2.0
        for col in range(cnt):
            a4.append((col + off, row))
    arrangements.append(a4)
    
    # 4-3-4-1
    a5 = []
    for row, cnt in enumerate([4,3,4,1]):
        off = (4 - cnt) / 2.0
        for col in range(cnt):
            a5.append((col + off, row))
    arrangements.append(a5)
    
    # Hexagonal cluster: center + 6 ring + 5 outer
    # Lattice coords for a hexagon of radius 1 (7 hexagons) plus 5 more
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # That's 7. Need 5 more from ring 2.
    # Ring 2 positions: (2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)
    # Pick 5 symmetric ones
    a6 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
          (2,0), (0,2), (-2,1), (-2,0), (0,-2)]
    arrangements.append(a6)
    
    # Another: 4-4-4 but staggered
    a7 = []
    for row in range(3):
        for col in range(4):
            a7.append((col + (row % 2) * 0.5, row))
    arrangements.append(a7)
    
    best_side = float('inf')
    best_inner = None
    best_angle = 0.0
    
    for arr in arrangements:
        # Compute positions
        positions = []
        for (i, j) in arr:
            x = i * e1[0] + j * e2[0]
            y = i * e1[1] + j * e2[1]
            positions.append((x, y))
        
        # Center
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        centered = [(x - cx, y - cy) for (x, y) in positions]
        
        # All vertices of inner hexagons (all at angle 0)
        all_verts = []
        for (x, y) in centered:
            all_verts.extend(hex_vertices(x, y, 1.0, 0.0))
        
        # Optimize outer hexagon orientation
        def obj(a):
            return min_hex_side(all_verts, a)
        
        res = minimize_scalar(obj, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
        side = res.fun
        ang = res.x
        
        if side < best_side:
            best_side = side
            best_angle = ang
            best_inner = [[x, y, 0.0] for (x, y) in centered]
    
    return best_inner, (0.0, 0.0), best_side, best_angle
