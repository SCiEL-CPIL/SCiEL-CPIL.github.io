import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define a compact cluster of 12 unit hexagons in honeycomb axial coordinates
    positions_axial = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (2, 0),
        (1, 1),
        (0, 2),
        (-1, 2),
        (-2, 1),
    ]
    
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        centers.append((x, y))
    
    def hexagon_vertices(cx, cy, side, rotation_deg):
        verts = []
        for k in range(6):
            angle = np.radians(rotation_deg) + k * np.pi / 3
            vx = cx + side * np.cos(angle)
            vy = cy + side * np.sin(angle)
            verts.append((vx, vy))
        return verts
    
    def point_in_hexagon(px, py, hx, hy, side, rotation_deg):
        inradius = side * np.sqrt(3) / 2.0
        for k in range(6):
            normal_angle = np.radians(rotation_deg) + np.pi / 6 + k * np.pi / 3
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            dist = (px - hx) * nx + (py - hy) * ny
            if dist > inradius + 1e-9:
                return False
        return True
    
    def cluster_fits(phi, theta, S):
        cos_phi = np.cos(np.radians(phi))
        sin_phi = np.sin(np.radians(phi))
        
        for i in range(N):
            cx0, cy0 = centers[i]
            cx = cx0 * cos_phi - cy0 * sin_phi
            cy = cx0 * sin_phi + cy0 * cos_phi
            
            inner_rot = phi
            verts = hexagon_vertices(cx, cy, 1.0, inner_rot)
            
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, 0.0, 0.0, S, theta):
                    return False
        return True
    
    def min_side_for_phi(phi, theta=0.0):
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if cluster_fits(phi, theta, mid):
                hi = mid
            else:
                lo = mid
        return hi
    
    lo_phi, hi_phi = 0.0, 60.0
    for _ in range(80):
        m1 = lo_phi + (hi_phi - lo_phi) / 3.0
        m2 = hi_phi - (hi_phi - lo_phi) / 3.0
        f1 = min_side_for_phi(m1)
        f2 = min_side_for_phi(m2)
        if f1 < f2:
            hi_phi = m2
        else:
            lo_phi = m1
    
    best_phi = (lo_phi + hi_phi) / 2.0
    best_S = min_side_for_phi(best_phi)
    
    cos_phi = np.cos(np.radians(best_phi))
    sin_phi = np.sin(np.radians(best_phi))
    
    inner = []
    for i in range(N):
        cx0, cy0 = centers[i]
        cx = cx0 * cos_phi - cy0 * sin_phi
        cy = cx0 * sin_phi + cy0 * cos_phi
        inner.append([cx, cy, best_phi])
    
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
