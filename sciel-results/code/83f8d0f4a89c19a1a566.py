"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Helper: Get vertices of a regular hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        verts = []
        for k in range(6):
            ang = math.radians(angle_deg + 60 * k)
            vx = cx + side * math.cos(ang)
            vy = cy + side * math.sin(ang)
            verts.append((vx, vy))
        return verts

    # Helper: Support function of a hexagon in direction u (unit vector)
    # h(u) = max(v dot u) for v in vertices
    def hex_support(cx, cy, side, angle_deg, ux, uy):
        max_proj = -1e9
        for k in range(6):
            ang = math.radians(angle_deg + 60 * k)
            vx = cx + side * math.cos(ang)
            vy = cy + side * math.sin(ang)
            proj = vx * ux + vy * uy
            if proj > max_proj:
                max_proj = proj
        return max_proj

    # Helper: Calculate overlap penalty between two hexagons
    # Returns 0 if disjoint, >0 if overlapping
    def overlap_penalty(c1x, c1y, a1, c2x, c2y, a2):
        # Critical directions are edge normals of both hexagons
        # Edge normals for hex at angle a are at a + 30 + 60k
        dirs = []
        for k in range(6):
            dirs.append(math.radians(a1 + 30 + 60 * k))
            dirs.append(math.radians(a2 + 30 + 60 * k))
        
        max_violation = 0.0
        for ang in dirs:
            ux = math.cos(ang)
            uy = math.sin(ang)
            
            # h1(u) + h2(-u) <= u . (c2 - c1)
            # h2(-u) is support of H2 in direction -u
            # Note: support is defined from origin? No, it's a function of the shape relative to its center?
            # Standard support function h_K(u) = sup { x . u : x in K }.
            # If K is centered at c, h_K(u) = c . u + h_{K-c}(u).
            # So h_{H1}(u) = c1 . u + h_{rel1}(u)
            # h_{H2}(-u) = c2 . (-u) + h_{rel2}(-u) = -c2 . u + h_{rel2}(-u)
            # Condition: c1 . u + h_{rel1}(u) - c2 . u + h_{rel2}(-u) <= u . (c2 - c1)
            # (c1 - c2) . u + h_{rel1}(u) + h_{rel2}(-u) <= 0
            # h_{rel1}(u) + h_{rel2}(-u) <= (c2 - c1) . u
            
            # Let's compute h_rel for unit hexagon (side 1, center 0)
            h_rel1 = 0
            for k in range(6):
                ang_v = math.radians(a1 + 60 * k)
                vx = math.cos(ang_v)
                vy = math.sin(ang_v)
                proj = vx * ux + vy * uy
                if proj > h_rel1:
                    h_rel1 = proj
            
            h_rel2_neg = 0
            for k in range(6):
                ang_v = math.radians(a2 + 60 * k)
                vx = math.cos(ang_v)
                vy = math.sin(ang_v)
                # Direction is -u
                proj = -vx * ux - vy * uy
                if proj > h_rel2_neg:
                    h_rel2_neg = proj
            
            c2_minus_c1 = (c2x - c1x, c2y - c1y)
            dot_sep = c2_minus_c1[0] * ux + c2_minus_c1[1] * uy
            
            violation = h_rel1 + h_rel2_neg - dot_sep
            if violation > max_violation:
                max_violation = violation
                
        return max_violation

    # Helper: Boundary penalty for one inner hexagon vs outer hexagon
    # Outer: center (ox, oy), side S, rotation 0
    # Inner: center (ix, iy), side 1, rotation a
    def boundary_penalty(ix, iy, a, ox, oy, S):
        # Outer hexagon edge normals (for rotation 0, pointy top? or flat top?)
        # Let's assume standard orientation: vertices at 0, 60, ...
        # Then edge normals are at 30, 90, ...
        # Inradius of outer = S * sqrt(3) / 2
        inradius = S * sqrt3 / 2
        
        # Vertices of inner hexagon
        verts = get_hex_vertices(ix, iy, 1.0, a)
        
        max_violation = 0.0
        for vx, vy in verts:
            # Check against all 6 outer edges
            for k in range(6):
                ang_n = math.radians(30 + 60 * k)
                nx = math.cos(ang_n)
                ny = math.sin(ang_n)
                
                # Projection of vertex relative to outer center onto normal
                proj = (vx - ox) * nx + (vy - oy) * ny
                # Must be <= inradius
                violation = proj - inradius
                if violation > max_violation:
                    max_violation = violation
                    
        return max_violation

    def objective(params):
        n = N
        # Extract params
        pos = []
        for i in range(n):
            pos.append((params[2*i], params[2*i+1]))
        angs = [params[2*n + i] for i in range(n)]
        ox = params[3*n]
        oy = params[3*n + 1]
        S = params[3*n + 2]
        
        if S < 2.0:
            return 1e6
            
        total_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                v = overlap_penalty(pos[i][0], pos[i][1], angs[i], 
                                    pos[j][0], pos[j][1], angs[j])
                if v > 0:
                    total_overlap += v * v # Quadratic penalty for smoothness
        
        total_boundary = 0.0
        for i in range(n):
            v = boundary_penalty(pos[i][0], pos[i][1], angs[i], ox, oy, S)
            if v > 0:
                total_boundary += v * v
        
        # We want to minimize S, but penalize violations heavily
        # If violations are non-zero, the solution is invalid.
        # The evaluator checks validity. If invalid, score 0.
        # So we must drive violations to 0.
        # Return S + large_penalty * (total_overlap + total_boundary)
        return S + 100.0 * (total_overlap + total_boundary)

    # Bounds
    bounds = []
    # Inner positions: roughly within [-5, 5]
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
    # Inner angles: [0, 360)
    for i in range(N):
        bounds.append((0.0, 360.0))
    # Outer center: [-5, 5]
    bounds.append((-5.0, 5.0))
    bounds.append((-5.0, 5.0))
    # Outer side: [3.0, 15.0]
    bounds.append((3.0, 15.0))
    
    # Initial guess: Hexagonal packing
    # Rows: 3, 4, 4
    rows = [3, 4, 4]
    init_pos = []
    init_angs = []
    y_offset = 0.0
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            init_pos.append((x, y))
            init_angs.append(0.0)
        y_offset += 1.5
        
    # Center init
    xs = [p[0] for p in init_pos]
    ys = [p[1] for p in init_pos]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for i in range(N):
        init_pos[i] = (init_pos[i][0] - cx, init_pos[i][1] - cy)
        
    x0 = []
    for p in init_pos:
        x0.append(p[0])
        x0.append(p[1])
    for a in init_angs:
        x0.append(a)
    x0.append(0.0)
    x0.append(0.0)
    
    # Estimate S
    max_dist = max(math.hypot(p[0], p[1]) for p in init_pos)
    S_init = max_dist + 1.5
    x0.append(S_init)
    
    # Run DE
    # Use a robust seed
    result = differential_evolution(
        objective,
        bounds,
        x0=x0,
        maxiter=100,
        popsize=20,
        tol=1e-5,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True # Use L-BFGS-B at the end for fine tuning
    )
    
    best_params = result.x
    pos = []
    for i in range(N):
        pos.append([best_params[2*i], best_params[2*i+1]])
    angs = [best_params[2*N + i] for i in range(N)]
    ox = best_params[3*N]
    oy = best_params[3*N + 1]
    S = best_params[3*N + 2]
    
    # Verify validity manually to be safe?
    # If result.fun is high, it might be invalid.
    # The objective includes penalties, so if fun is close to S, it's valid.
    # If fun >> S, it's invalid.
    # We return the best found.
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (ox, oy), S, 0.0
# EVOLVE-BLOCK-END
