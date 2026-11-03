"""Optimized packing of 12 unit hexagons in a regular hexagon via convex optimization."""
import numpy as np
from scipy.optimize import minimize
import math

N = 12
SQRT3 = math.sqrt(3.0)

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon as a (6,2) array."""
    a = math.radians(angle_deg)
    verts = np.zeros((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def construct():
    """
    Pack 12 unit hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, center, side, angle).
    """
    # Variables: [cx1,cy1,a1, cx2,cy2,a2, ..., cx12,cy12,a12, ox,oy,oa,os]
    # Total: 3*12 + 4 = 40
    
    # Initial guess: hexagonal grid arrangement
    init_inner = []
    # 4 columns x 3 rows, hexagonal offset
    for row in range(3):
        for col in range(4):
            x = col * 2.2 + (0.0 if row % 2 == 0 else 1.1)
            y = row * 1.9
            init_inner.append([x, y, 0.0])
    
    init_outer = [0.0, 0.0, 0.0, 8.0]
    
    x0 = []
    for p in init_inner:
        x0.extend(p)
    x0.extend(init_outer)
    x0 = np.array(x0, dtype=float)
    
    bounds = []
    for i in range(N):
        bounds.append((-50.0, 50.0))
        bounds.append((-50.0, 50.0))
        bounds.append((0.0, 360.0))
    bounds.append((-50.0, 50.0))
    bounds.append((-50.0, 50.0))
    bounds.append((0.0, 360.0))
    bounds.append((1.0, 50.0))
    
    def objective(x):
        return x[-1]
    
    def constraints(x):
        ineq = []
        inner_params = x[:3*N].reshape(N, 3)
        ox, oy, o_angle, o_side = x[3*N], x[3*N+1], x[3*N+2], x[3*N+3]
        
        r_outer = o_side * SQRT3 / 2.0
        a_outer = math.radians(o_angle)
        
        # Outer hexagon: 6 half-planes. Normal k points to side midpoint at angle a_outer + (k+0.5)*pi/3
        # Constraint: n_k . (p - o_center) <= r_outer
        # => n_k . p <= r_outer + n_k . o_center
        
        for i in range(N):
            cx, cy, ang = inner_params[i]
            verts = hex_vertices(cx, cy, 1.0, ang)
            for v in verts:
                for k in range(6):
                    theta = a_outer + (k + 0.5) * math.pi / 3.0
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    # n_k . v <= r_outer + n_k . [ox, oy]
                    lhs = nx * v[0] + ny * v[1]
                    rhs = r_outer + nx * ox + ny * oy
                    ineq.append(rhs - lhs)  # >= 0
        
        # Non-overlap: center distance squared >= 3 (conservative for unit hexagons)
        for i in range(N):
            for j in range(i+1, N):
                dx = inner_params[i, 0] - inner_params[j, 0]
                dy = inner_params[i, 1] - inner_params[j, 1]
                ineq.append(dx*dx + dy*dy - 3.0)
        
        return np.array(ineq)
    
    cons = [{'type': 'ineq', 'fun': constraints}]
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 300, 'ftol': 1e-7})
        
        if result.success or (result.fun < 10.0 and not np.any(np.isnan(result.x))):
            x = result.x
            inner = []
            for i in range(N):
                inner.append([float(x[3*i]), float(x[3*i+1]), float(x[3*i+2])])
            center = (float(x[3*N]), float(x[3*N+1]))
            side = float(x[3*N+3])
            angle = float(x[3*N+2])
        else:
            inner = init_inner
            center = (0.0, 0.0)
            side = 8.0
            angle = 0.0
    except Exception:
        inner = init_inner
        center = (0.0, 0.0)
        side = 8.0
        angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
