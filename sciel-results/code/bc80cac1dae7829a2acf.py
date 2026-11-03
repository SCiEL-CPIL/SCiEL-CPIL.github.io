"""
Packing 11 unit regular hexagons inside a regular hexagon.
"""

import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place 11 unit hexagons in a hex lattice arrangement (4-3-4 pattern)
    axial_coords = [
        (-2, 0), (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2)
    ]
    
    # Convert to Cartesian for flat-top hexagons
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        centers.append((x, y))
    
    # All inner hexagons have angle 0
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Find the minimum enclosing hexagon
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    cx0 = (max(xs) + min(xs)) / 2
    cy0 = (max(ys) + min(ys)) / 2
    S0 = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 2
    
    def max_violation(cx, cy, S, angle_deg, points):
        max_v = 0.0
        for px, py in points:
            dx = px - cx
            dy = py - cy
            theta = np.radians(angle_deg)
            rx = dx * np.cos(theta) + dy * np.sin(theta)
            ry = -dx * np.sin(theta) + dy * np.cos(theta)
            apothem = S * np.sqrt(3) / 2
            for k in range(6):
                angle = np.radians(60 * k)
                nx = np.cos(angle)
                ny = np.sin(angle)
                proj = rx * nx + ry * ny
                v = proj - apothem
                if v > max_v:
                    max_v = v
        return max_v
    
    def objective(params):
        cx, cy, S, angle = params
        if S <= 0:
            return 1e10
        return S + 10 * max(max_violation(cx, cy, S, angle, centers), 0)
    
    x0 = [cx0, cy0, S0, 0.0]
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    cx, cy, S, angle = result.x
    if S <= 0:
        S = S0
    
    return inner, (cx, cy), S, angle
# EVOLVE-BLOCK-END
