import numpy as np
import time
from scipy.optimize import linprog

N = 11
DEADLINE = time.time() + 140.0

def get_hex_vertices(cx, cy, angle_deg):
    rad = np.deg2rad(angle_deg)
    verts = []
    for i in range(6):
        theta = rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def calc_min_enclosing_hex(all_verts):
    """
    Finds the minimal side length, center, and angle of a regular hexagon
    enclosing the given set of vertices.
    """
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sweep angle
    # 0 to 60 degrees
    # Use 1 degree steps for speed, then refine best
    for phi_deg in np.arange(0, 60, 1.0):
        if time.time() > DEADLINE:
            break
        
        phi_rad = np.deg2rad(phi_deg)
        normals = []
        for k in range(6):
            theta = phi_rad + np.pi/6 + k * np.pi/3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        c = [0, 0, 1]
        A = []
        b = []
        
        for p in all_verts:
            px, py = p
            for n in normals:
                nx, ny = n
                A.append([-nx, ny, -np.sqrt(3)/2])
                b.append(-(nx*px - ny*py))
        
        bounds = [(-100, 100), (-100, 100), (0, None)]
        
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                xc, yc, L = res.x
                if L < best_side:
                    best_side = L
                    best_center = (xc, yc)
                    best_angle = phi_deg
        except Exception:
            continue
    
    # Refine around best angle
    if best_side < float('inf'):
        start = max(0, best_angle - 1.0)
        end = min(60, best_angle + 1.0)
        for phi_deg in np.arange(start, end, 0.1):
            if time.time() > DEADLINE:
                break
            phi_rad = np.deg2rad(phi_deg)
            normals = []
            for k in range(6):
                theta = phi_rad + np.pi/6 + k * np.pi/3
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            c = [0, 0, 1]
            A = []
            b = []
            
            for p in all_verts:
                px, py = p
                for n in normals:
                    nx, ny = n
                    A.append([-nx, ny, -np.sqrt(3)/2])
                    b.append(-(nx*px - ny*py))
            
            bounds = [(-100, 100), (-100, 100), (0, None)]
            
            try:
                res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if res.success:
                    xc, yc, L = res.x
                    if L < best_side:
                        best_side = L
                        best_center = (xc, yc)
                        best_angle = phi_deg
            except Exception:
                continue

    return best_side, best_center, best_angle

def construct():
    # Candidate Topologies
    candidates = []
    
    # 1. 4-4-3 Block
    pts1 = []
    for q in range(4): pts1.append((q, 0))
    for q in range(4): pts1.append((q, 1))
    for q in range(3): pts1.append((q, 2))
    candidates.append(pts1)
    
    # 2. 3-4-4 Block (Shifted)
    pts2 = []
    for q in range(3): pts2.append((q, 0))
    for q in range(4): pts2.append((q, 1))
    for q in range(4): pts2.append((q, 2))
    candidates.append(pts2)
    
    # 3. 1-6-4 Star (Center + 1 Ring + 4 from 2nd Ring)
    # Axial coords for hex grid
    # Center (0,0)
    # 1 Ring: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # 2 Ring: Pick 4 symmetric ones.
    # (2,0), (2,-2), (-2,0), (-2,2) ?
    # Let's use (2,0), (0,2), (-2,0), (0,-2) ? No, those are not 2nd ring in axial?
    # Axial distance: max(|q|, |r|, |q+r|)
    # (2,0) dist 2. (0,2) dist 2. (-2,0) dist 2. (0,-2) dist 2.
    # These are "opposite" vertices of the 2nd ring hexagon.
    pts3 = [(0,0)]
    pts3 += [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    pts3 += [(2,0), (0,2), (-2,0), (0,-2)]
    candidates.append(pts3)
    
    best_overall_side = float('inf')
    best_overall_inner = None
    best_overall_center = (0,0)
    best_overall_angle = 0
    
    for axial_pts in candidates:
        if time.time() > DEADLINE:
            break
            
        # Convert to Cartesian
        cart = []
        for q, r in axial_pts:
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            cart.append([x, y])
        cart = np.array(cart)
        
        # Center
        centroid = cart.mean(axis=0)
        cart -= centroid
        
        # Generate Vertices
        all_verts = []
        for x, y in cart:
            verts = get_hex_vertices(x, y, 0.0)
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        # Find Min Enclosing Hex
        side, center, angle = calc_min_enclosing_hex(all_verts)
        
        if side < best_overall_side:
            best_overall_side = side
            best_overall_inner = [[x, y, 0.0] for x, y in cart]
            best_overall_center = center
            best_overall_angle = angle
            
    # Fallback
    if best_overall_side == float('inf'):
        # Return a valid but large configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_overall_inner, best_overall_center, best_overall_side, best_overall_angle
