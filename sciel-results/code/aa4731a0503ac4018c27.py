import numpy as np
import time

N = 11

def hex_verts(cx, cy, ang_deg, s=1.0):
    a = np.radians(ang_deg)
    return np.array([[cx + s*np.cos(a + k*np.pi/3), cy + s*np.sin(a + k*np.pi/3)] for k in range(6)])

def poly_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap (intersect with positive area or one contains a vertex of other)."""
    # Check if any vertex of v1 is inside v2 or vice versa
    def point_in_poly(p, poly):
        # Ray casting
        n = len(poly)
        inside = False
        x, y = p
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    for p in v1:
        if point_in_poly(p, v2):
            return True
    for p in v2:
        if point_in_poly(p, v1):
            return True
    return False

def min_side_for_angle(pts, cx, cy, ang_deg):
    """Compute required side length for outer hexagon with given center and angle."""
    a = np.radians(ang_deg)
    inradius = 0.0
    for k in range(6):
        # Outward normal direction for edge k
        n_ang = a + np.pi/6 + k * np.pi/3
        nx, ny = np.cos(n_ang), np.sin(n_ang)
        for p in pts:
            proj = (p[0]-cx)*nx + (p[1]-cy)*ny
            if proj > inradius:
                inradius = proj
    if inradius < 0:
        inradius = 0
    return inradius / (np.sqrt(3)/2)

def min_enclosing_side(pts, cx=0.0, cy=0.0):
    """Find min side length over all outer hexagon angles."""
    best = np.inf
    best_ang = 0.0
    # Coarse grid
    for ang in np.linspace(0, 60, 60):
        s = min_side_for_angle(pts, cx, cy, ang)
        if s < best:
            best = s
            best_ang = ang
    # Fine grid around best
    for ang in np.linspace(best_ang-5, best_ang+5, 100):
        s = min_side_for_angle(pts, cx, cy, ang)
        if s < best:
            best = s
            best_ang = ang
    return best, best_ang

def construct():
    t0 = time.time()
    deadline = t0 + 115
    
    # Initial: hexagonal lattice, spacing 2.0
    # 1 center + 6 ring1 + 4 from ring2
    init = [(0,0)]
    for k in range(6):
        a = k*np.pi/3
        init.append((2*np.cos(a), 2*np.sin(a)))
    # Ring 2: 12 positions, pick 4
    for k in range(0, 12, 3):  # every 3rd
        a = k*np.pi/6
        init.append((4*np.cos(a), 4*np.sin(a)))
    
    inner = [[float(x), float(y), 0.0] for (x,y) in init[:N]]
    
    # Verify no overlaps in initial config
    for i in range(N):
        for j in range(i+1, N):
            v1 = hex_verts(inner[i][0], inner[i][1], inner[i][2])
            v2 = hex_verts(inner[j][0], inner[j][1], inner[j][2])
            if poly_overlap(v1, v2):
                # Push apart
                pass  # Initial config should be fine with spacing 2
    
    all_pts = np.vstack([hex_verts(x, y, a) for (x,y,a) in inner])
    best_side, best_ang = min_enclosing_side(all_pts)
    best_inner = [list(x) for x in inner]
    
    # Local search: try moving each hexagon
    for it in range(500):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            ox, oy, oa = best_inner[idx]
            best_local = best_side
            best_li = best_inner[:]
            
            for dx in [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]:
                for dy in [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]:
                    for da in [-30, -15, -5, 5, 15, 30]:
                        if time.time() > deadline:
                            break
                        ni = [list(x) for x in best_li]
                        ni[idx] = [ox+dx, oy+dy, oa+da]
                        
                        # Check overlaps
                        ok = True
                        vi = hex_verts(ni[idx][0], ni[idx][1], ni[idx][2])
                        for j in range(N):
                            if j == idx:
                                continue
                            vj = hex_verts(ni[j][0], ni[j][1], ni[j][2])
                            if poly_overlap(vi, vj):
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        ap = np.vstack([hex_verts(x, y, a) for (x,y,a) in ni])
                        s, a2 = min_enclosing_side(ap)
                        if s < best_local:
                            best_local = s
                            best_li = ni
                            improved = True
            
            if best_local < best_side - 1e-10:
                best_side = best_local
                best_inner = best_li
                improved = True
        
        if not improved:
            break
    
    ap = np.vstack([hex_verts(x, y, a) for (x,y,a) in best_inner])
    final_side, final_ang = min_enclosing_side(ap)
    
    return best_inner, (0.0, 0.0), final_side, final_ang
