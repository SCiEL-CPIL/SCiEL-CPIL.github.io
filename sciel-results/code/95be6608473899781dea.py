import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    sqrt3 = np.sqrt(3)
    
    best_S = float('inf')
    best_centers = None
    best_center = None
    
    # Generate hexagonal lattice positions
    positions = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 1.5
            positions.append((x, y))
    
    # Sort by distance from origin and take 12 most central
    positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    centers = positions[:N]
    
    # Compute all vertices of the 12 unit hexagons
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = np.pi / 3 * k
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
    
    # Optimize the outer hexagon center to minimize side length
    def max_constraint(params):
        hx, hy = params
        max_val = 0.0
        for vx, vy in all_vertices:
            dx = vx - hx
            dy = vy - hy
            c1 = abs(dy) / (sqrt3 / 2)
            c2 = abs(dx)
            c3 = abs(dx * sqrt3/2 + dy/2) / (sqrt3/2)
            c4 = abs(dx * sqrt3/2 - dy/2) / (sqrt3/2)
            m = c1 if c1 > c2 else c2
            m = m if m > c3 else c3
            m = m if m > c4 else c4
            if m > max_val:
                max_val = m
        return max_val
    
    res = minimize(max_constraint, [0.0, 0.0], method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    hx, hy = res.x
    S = res.fun
    
    inner = [[c[0], c[1], 0.0] for c in centers]
    return inner, (hx, hy), S, 0.0
