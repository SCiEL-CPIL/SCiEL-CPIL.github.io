"""Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices_arr(cx, cy, side, angle_deg):
    """Return (6,2) array of vertices of regular hexagon."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    base = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    rot = np.array([[ca, -sa], [sa, ca]])
    verts = base @ rot.T
    return np.column_stack([cx + side*verts[:,0], cy + side*verts[:,1]])

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    if dx*dx + dy*dy < 1e-14:
        return np.sqrt((px-ax)**2 + (py-ay)**2)
    t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t*dx, ay + t*dy
    return np.sqrt((px-cx)**2 + (py-cy)**2)

def _separation(x, i, j):
    """Minimum distance between hex i and hex j (positive if separated)."""
    cx1, cy1, a1 = x[3*i], x[3*i+1], x[3*i+2]
    cx2, cy2, a2 = x[3*j], x[3*j+1], x[3*j+2]
    v1 = _hex_vertices_arr(cx1, cy1, 1.0, a1)
    v2 = _hex_vertices_arr(cx2, cy2, 1.0, a2)
    min_sep = float('inf')
    for ie in range(6):
        ax, ay = v1[ie, 0], v1[ie, 1]
        bx, by = v1[(ie+1)%6, 0], v1[(ie+1)%6, 1]
        for jv in range(6):
            d = _point_to_segment_dist(v2[jv,0], v2[jv,1], ax, ay, bx, by)
            min_sep = min(min_sep, d)
    for je in range(6):
        ax, ay = v2[je, 0], v2[je, 1]
        bx, by = v2[(je+1)%6, 0], v2[(je+1)%6, 1]
        for iv in range(6):
            d = _point_to_segment_dist(v1[iv,0], v1[iv,1], ax, ay, bx, by)
            min_sep = min(min_sep, d)
    return min_sep

def _containment_violation(x, outer_side):
    """Max distance any inner vertex is outside the outer hexagon (0 if all inside)."""
    apothem = outer_side * np.sqrt(3)/2
    max_viol = 0.0
    for i in range(N):
        cx, cy, a = x[3*i], x[3*i+1], x[3*i+2]
        verts = _hex_vertices_arr(cx, cy, 1.0, a)
        for v in verts:
            for k in range(6):
                theta = np.radians(30 + 60*k)
                nx, ny = np.cos(theta), np.sin(theta)
                dist = v[0]*nx + v[1]*ny - apothem
                if dist > max_viol:
                    max_viol = dist
    return max_viol

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    outer_angle = 0.0
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice initial positions
    positions = [(0, 0)]
    for i in range(6):
        ang = np.radians(60*i)
        positions.append((sqrt3*np.cos(ang), sqrt3*np.sin(ang)))
    for i in range(6):
        ang = np.radians(60*i + 30)
        positions.append((sqrt3*np.cos(ang)*2, sqrt3*np.sin(ang)*2))
    
    x0 = np.zeros(3*N)
    for i in range(N):
        x0[3*i] = positions[i][0]
        x0[3*i+1] = positions[i][1]
        x0[3*i+2] = 0.0
    
    def penalty(x, outer_side):
        pen = 0.0
        cv = _containment_violation(x, outer_side)
        if cv > 0:
            pen += cv**2 * 10
        for i in range(N):
            for j in range(i+1, N):
                sep = _separation(x, i, j)
                if sep < 0:
                    pen += sep**2 * 10
        return pen
    
    # Binary search on outer_side
    lower, upper = 3.0, 10.0
    best_solution = None
    best_side = upper
    
    for _ in range(12):
        if time.time() > deadline - 10:
            break
        mid = (lower + upper) / 2
        best_pen = float('inf')
        best_x = None
        
        for r in range(4):
            if time.time() > deadline - 5:
                break
            if r == 0:
                x_init = x0.copy()
            else:
                x_init = x0 + np.random.normal(0, 0.3, size=3*N)
                x_init[2::3] = 0
            
            try:
                res = minimize(lambda xx: penalty(xx, mid), x_init, method='L-BFGS-B',
                               options={'maxiter': 100, 'gtol': 1e-5})
                if res.fun < best_pen:
                    best_pen = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        if best_pen < 1e-4:
            upper = mid
            best_solution = best_x
            best_side = mid
        else:
            lower = mid
    
    if best_solution is not None:
        inner = [[best_solution[3*i], best_solution[3*i+1], best_solution[3*i+2]] for i in range(N)]
        return inner, (0.0, 0.0), best_side, outer_angle
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
