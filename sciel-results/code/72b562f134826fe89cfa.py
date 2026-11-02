import numpy as np
from scipy.optimize import minimize
import math

N = 12

def construct():
    sqrt3 = math.sqrt(3)
    
    # Compact hexagonal cluster of 12 unit hexagons
    # Axial coordinates (q, r) -> Cartesian for flat-top hex tiling
    positions_axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2),
    ]
    
    pts = np.array([
        [sqrt3 * (q + r / 2.0), 1.5 * r] for q, r in positions_axial
    ])
    
    def min_inradius(cx, cy, angle_deg):
        a = math.radians(angle_deg)
        mr = 0.0
        for i in range(6):
            na = a + math.radians(60 * i + 30)
            nx, ny = math.cos(na), math.sin(na)
            d = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            v = np.max(d)
            if v > mr:
                mr = v
        return mr
    
    def obj(p):
        cx, cy, ang = p
        return 2.0 * min_inradius(cx, cy, ang) / sqrt3
    
    cx0, cy0 = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    
    best, bp = float('inf'), None
    for a0 in [0, 30, 60, 90, 120, 150]:
        r = minimize(obj, [cx0, cy0, float(a0)], method='Nelder-Mead',
                     options={'maxiter': 30000, 'xatol': 1e-9, 'fatol': 1e-11})
        if r.fun < best:
            best, bp = r.fun, r.x
    
    cx, cy, ang = bp
    inner = [[float(x), float(y), 0.0] for x, y in pts]
    return inner, (float(cx), float(cy)), float(best), float(ang)
