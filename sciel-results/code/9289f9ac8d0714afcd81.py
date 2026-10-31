"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Helper to get vertices of a regular hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def support_function_hex(verts, u):
        # u is a unit vector
        # h(u) = max dot(v, u) for v in verts
        dots = np.dot(verts, u)
        return np.max(dots)

    def overlap_penalty(params, i, j):
        # Extract params for i and j
        cx_i, cy_i, ang_i = params[3*i], params[3*i+1], params[3*i+2]
        cx_j, cy_j, ang_j = params[3*j], params[3*j+1], params[3*j+2]
        
        # Vertices
        v_i = get_hex_vertices(cx_i, cy_i, 1.0, ang_i)
        v_j = get_hex_vertices(cx_j, cy_j, 1.0, ang_j)
        
        # Center difference
        dx = cx_i - cx_j
        dy = cy_i - cy_j
        
        # Get normals for both hexagons
        # Normals are perpendicular to edges.
        # For a regular hexagon, edge normals are at angles: angle_deg + 30 + 60*k
        penalty = 0.0
        
        # Axes from hex i
        for k in range(6):
            theta_i = math.radians(ang_i + 30 + 60*k)
            u = np.array([math.cos(theta_i), math.sin(theta_i)])
            
            h_i = support_function_hex(v_i, u)
            h_j = support_function_hex(v_j, -u)
            
            dist = dx * u[0] + dy * u[1]
            
            # If dist < h_i + h_j, they overlap in this direction
            violation = (h_i + h_j) - dist
            if violation > 0:
                penalty += violation * violation
        
        # Axes from hex j
        for k in range(6):
            theta_j = math.radians(ang_j + 30 + 60*k)
            u = np.array([math.cos(theta_j), math.sin(theta_j)])
            
            h_i = support_function_hex(v_i, u)
            h_j = support_function_hex(v_j, -u)
            
            dist = dx * u[0] + dy * u[1]
            
            violation = (h_i + h_j) - dist
            if violation > 0:
                penalty += violation * violation
                
        return penalty

    def boundary_penalty(params, cx_out, cy_out, s_out):
        # Outer hexagon fixed rotation 0
        # Inradius of outer hexagon
        inrad = s_out * sqrt3 / 2
        
        # Outer hexagon normals (for rotation 0, flat top? or pointy top?)
        # Let's assume pointy top (vertex at 90 deg? No, standard math angle 0 is right).
        # If vertex at 0, 60, 120... then normals are at 30, 90, 150...
        # Let's use the same convention as get_hex_vertices: vertices at angle + 60k.
        # If angle=0, vertex at 0.
        # Edge between vertex 0 and 1 is perpendicular to direction 30 deg.
        # So normals are at 30, 90, 150, 210, 270, 330.
        
        penalty = 0.0
        for i in range(N):
            cx_i, cy_i, ang_i = params[3*i], params[3*i+1], params[3*i+2]
            v_i = get_hex_vertices(cx_i, cy_i, 1.0, ang_i)
            
            for m in range(6):
                theta_m = math.radians(30 + 60*m)
                u = np.array([math.cos(theta_m), math.sin(theta_m)])
                
                # Project outer center to get boundary offset
                # The boundary is at distance inrad from center in direction u
                # So for a point p, (p - c_out) . u <= inrad
                # Violation = (p - c_out) . u - inrad
                
                # Vector from outer center to inner center
                dx = cx_i - cx_out
                dy = cy_i - cy_out
                
                # We need to check all vertices of inner hex i
                # Or we can use support function of inner hex in direction u
                # h_inner(u) + (c_i - c_out) . u <= inrad
                # violation = h_inner(u) + (c_i - c_out) . u - inrad
                
                h_inner = support_function_hex(v_i, u)
                proj_center = dx * u[0] + dy * u[1]
                
                violation = h_inner + proj_center - inrad
                if violation > 0:
                    penalty += violation * violation
                    
        return penalty

    def objective(params):
        # params structure:
        # 0..2N-1: cx, cy, angle for each inner hex
        # 2N: cx_out
        # 2N+1: cy_out
        # 2N+2: s_out
        
        cx_out = params[2*N]
        cy_out = params[2*N+1]
        s_out = params[2*N+2]
        
        if s_out < 2.0:
            return 1e10
        
        # Overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen += overlap_penalty(params, i, j)
        
        # Boundary penalty
        pen += 10.0 * boundary_penalty(params, cx_out, cy_out, s_out)
        
        # Objective: minimize side length + penalties
        # If penalties are large, return large value
        if pen > 1e-3:
            return s_out + pen * 1000
        else:
            return s_out

    # Initial guess: Hexagonal packing
    # Rows: 3, 4, 4
    rows = [3, 4, 4]
    positions = []
    angles = []
    y_offset = 0.0
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            positions.append([x, y])
            angles.append(0.0)
        y_offset += 1.5
    
    # Center
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_init = (min(xs) + max(xs)) / 2
    cy_init = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx_init
        p[1] -= cy_init
        
    # Estimate s_init
    max_dist = 0
    for p in positions:
        max_dist = max(max_dist, math.hypot(p[0], p[1]))
    s_init = max_dist + 1.5

    # Build initial params
    x0 = []
    for i in range(N):
        x0.append(positions[i][0])
        x0.append(positions[i][1])
        x0.append(angles[i])
    x0.append(0.0)
    x0.append(0.0)
    x0.append(s_init)

    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-3, 3)) # cx
        bounds.append((-3, 3)) # cy
        bounds.append((-30, 30)) # angle
    bounds.append((-2, 2)) # cx_out
    bounds.append((-2, 2)) # cy_out
    bounds.append((2.5, 8.0)) # s_out

    best_result = None
    best_score = 1e10
    
    # Multi-start DE
    for seed in range(5):
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0 if seed == 0 else None,
                maxiter=100,
                popsize=20,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=seed,
                polish=True
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to initial
        best_result = x0

    # Extract final solution
    pos = []
    angs = []
    for i in range(N):
        pos.append([best_result[3*i], best_result[3*i+1]])
        angs.append(best_result[3*i+2])
        
    cx_out = best_result[2*N]
    cy_out = best_result[2*N+1]
    s_out = best_result[2*N+2]
    
    # Ensure s_out is reasonable
    if s_out < 2.0:
        s_out = s_init
        
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (cx_out, cy_out), s_out, 0.0
# EVOLVE-BLOCK-END
