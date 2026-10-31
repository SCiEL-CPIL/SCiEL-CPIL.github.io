"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Heuristic initial arrangement: hexagonal packing
    # Try a 3-4-4 row arrangement
    sqrt3 = math.sqrt(3)
    
    # Initial positions in a hexagonal grid pattern
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons
    positions = []
    angles = []
    
    # Generate hexagonal packing positions
    # Each row offset by sqrt3/2 horizontally, rows separated by 1.5 vertically
    rows = [3, 4, 4]
    y_offset = 0.0
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            positions.append([x, y])
            angles.append(0.0)
        y_offset += 1.5
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    # Initial outer hexagon side length estimate
    # The bounding box of the inner hexagons
    max_dist = 0
    for p in positions:
        max_dist = max(max_dist, math.hypot(p[0], p[1]))
    # Outer hexagon circumradius must be at least max_dist + 1 (inner hex circumradius)
    s_init = max_dist + 1.0 + 0.5
    
    # Use differential evolution to optimize
    def objective(params):
        # params: 2*N positions + N angles + 2 center + 1 side = 2N + N + 3
        n = len(positions)
        pos = []
        for i in range(n):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*n + i] for i in range(n)]
        cx = params[3*n]
        cy = params[3*n + 1]
        s = params[3*n + 2]
        
        if s < 2.0:
            return 1e6
        
        # Check overlap between all pairs of inner hexagons
        # For two hexagons with centers c1, c2, angles a1, a2, side length 1
        # They overlap if the distance between centers is less than the sum of their
        # "effective radii" in the direction of separation.
        # Use support function: for a regular hexagon, the support function in direction u is:
        # h(u) = max over vertices of dot(vertex, u)
        # Two convex bodies A and B are disjoint if for all directions u, 
        # h_A(u) + h_B(-u) < dot(c2 - c1, u)
        # This is expensive, so use a simpler distance-based check with margin
        
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.hypot(dx, dy)
                # Minimum distance for non-overlap of two unit hexagons
                # The minimum distance between centers of two non-overlapping unit hexagons
                # depends on relative rotation. Worst case is ~2.0 (vertex to vertex)
                # Best case (flat to flat) is ~1.732
                # Use a conservative minimum of 1.7 to allow some overlap penalty
                min_dist = 1.7
                if dist < min_dist:
                    return 1e6 + (min_dist - dist) * 100
        
        # Check containment: each inner hexagon must be inside outer hexagon
        # Outer hexagon: center (cx, cy), side s, rotation 0 (we can fix outer rotation to 0)
        # A point (x, y) is inside a regular hexagon centered at (cx, cy) with side s if:
        # For a hexagon with flat top (rotation 0), the conditions are:
        # |x - cx| <= s
        # |y - cy| <= s * sqrt(3) / 2
        # And the corner constraints... actually let me use the proper half-plane test
        
        # Outer hexagon vertices for side s, center (cx, cy), rotation 0 (pointy top):
        # Actually, let's define outer hexagon with flat top for simplicity
        # Vertices at angles 0, 60, 120, 180, 240, 300 from center, distance s
        # For flat-top: vertices at 30, 90, 150, 210, 270, 330 degrees? 
        # Let me use: vertex k at angle (60*k) degrees, distance s from center
        # So vertices at 0, 60, 120, 180, 240, 300 degrees
        
        # To check if a point is inside, we can check it's on the correct side of each edge
        # Edge k connects vertex k and vertex k+1
        # The inward normal to edge k points toward the center
        
        # For a regular hexagon with circumradius s, the inradius is s * sqrt(3)/2
        # A point is inside if its distance from center in each of the 6 "flat" directions 
        # is <= inradius
        
        # The 6 directions from center to midpoints of edges (for pointy-top hex):
        # At angles 30, 90, 150, 210, 270, 330 degrees
        # Distance from center to each edge midpoint = inradius = s * sqrt(3)/2
        
        inradius_outer = s * sqrt3 / 2
        
        for i in range(n):
            # Check all 6 vertices of inner hexagon i are inside outer hexagon
            # Inner hexagon i: center pos[i], angle angs[i], side 1
            # Vertices of inner hexagon at angles angs[i] + 60*k, distance 1 from center
            for k in range(6):
                vx = pos[i][0] + math.cos(math.radians(angs[i] + 60*k))
                vy = pos[i][1] + math.sin(math.radians(angs[i] + 60*k))
                # Check if (vx, vy) is inside outer hexagon centered at (cx, cy) with inradius inradius_outer
                # The outer hexagon has edges perpendicular to directions at 30, 90, 150, 210, 270, 330
                # For each direction d_k, the projection of (vx-cx, vy-cy) onto d_k must be <= inradius_outer
                for m in range(6):
                    ang_m = math.radians(30 + 60*m)
                    dx = vx - cx
                    dy = vy - cy
                    proj = dx * math.cos(ang_m) + dy * math.sin(ang_m)
                    if proj > inradius_outer + 1e-6:
                        return 1e6 + (proj - inradius_outer) * 100
        
        return s
    
    # Define bounds for differential evolution
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
    for i in range(N):
        bounds.append((-30, 30))  # angle
    bounds.append((-5, 5))  # cx
    bounds.append((-5, 5))  # cy
    bounds.append((3.0, 15.0))  # s
    
    # Initial guess
    x0 = []
    for i in range(N):
        x0.append(positions[i][0])
        x0.append(positions[i][1])
    for i in range(N):
        x0.append(angles[i])
    x0.append(0.0)
    x0.append(0.0)
    x0.append(s_init)
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        x0=x0,
        maxiter=50,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42
    )
    
    # Extract solution
    best_params = result.x
    pos = []
    for i in range(N):
        pos.append([best_params[2*i], best_params[2*i+1]])
    angs = [best_params[2*N + i] for i in range(N)]
    cx = best_params[3*N]
    cy = best_params[3*N + 1]
    s = best_params[3*N + 2]
    
    # If DE didn't find a good solution, fall back to heuristic
    if result.fun > 100:
        # Use heuristic with estimated outer size
        max_dist = 0
        for p in positions:
            max_dist = max(max_dist, math.hypot(p[0], p[1]))
        s = max_dist + 1.5
        cx = 0.0
        cy = 0.0
        pos = positions
        angs = angles
    else:
        # Use the DE solution
        pass
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
    
    return inner, (cx, cy), s, 0.0
# EVOLVE-BLOCK-END
