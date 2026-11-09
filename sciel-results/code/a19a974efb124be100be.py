"""Hexagon packing: place 12 unit hexagons in a minimal enclosing hexagon."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon (flat-top orientation by default)."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(-angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = cos_a * dx - sin_a * dy
        ly = sin_a * dx + cos_a * dy

        # For a regular hexagon with vertices at angles 0, 60, 120, ...
        # The hexagon can be described as intersection of 6 half-planes
        # Distance from center in each of 6 directions must be <= side
        for i in range(6):
            theta = i * math.pi / 3.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            dist = lx * nx + ly * ny
            if dist > side:
                return False
        return True

    def hex_overlap(i, j, centers, angles, eps=1e-9):
        """Check if two unit hexagons overlap. Returns True if they do."""
        cx1, cy1 = centers[i]
        cx2, cy2 = centers[j]
        a1 = angles[i]
        a2 = angles[j]

        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = math.sqrt(dx * dx + dy * dy)

        # Quick rejection: if distance > 4 (diameter), no overlap
        if dist > 4.0 - eps:
            return False
        # If distance < 2*sqrt(3) ≈ 3.464, definitely overlapping (distance between parallel sides)
        if dist < 2.0 * sqrt3 - eps:
            return True

        # Check all vertices of both hexagons
        verts1 = get_hex_vertices(cx1, cy1, a1)
        verts2 = get_hex_vertices(cx2, cy2, a2)

        for v in verts1:
            if point_in_hex(v[0], v[1], cx2, cy2, a2):
                return True
        for v in verts2:
            if point_in_hex(v[0], v[1], cx1, cy1, a1):
                return True

        # Check edge intersections
        for k in range(6):
            e1 = (verts1[k], verts1[(k + 1) % 6])
            for l in range(6):
                e2 = (verts2[l], verts2[(l + 1) % 6])
                if segments_intersect(e1, e2):
                    return True

        return False

    def segments_intersect(p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        return False

    def required_L(centers, angles, outer_center=(0.0, 0.0)):
        """Compute minimum side length of enclosing hexagon centered at outer_center
        with a given orientation (we'll optimize outer orientation separately).
        Actually, let's compute the enclosing hexagon side for a given outer orientation.
        
        The enclosing hexagon is axis-aligned to some orientation. Let's fix the outer
        hexagon orientation to 0 degrees (flat-top) and just find the minimum side.
        
        For a regular hexagon of side L centered at (0,0) with flat-top:
        Vertices at angles 0, 60, 120, 180, 240, 300 degrees, distance L from center.
        A point (x,y) is inside if for each direction i, the projection onto the
        normal of side i is <= L.
        
        The 6 outward normals for a flat-top hexagon (vertices at 0, 60, ...) are
        at angles 30, 90, 150, 210, 270, 330 degrees.
        
        Actually, let me think more carefully. A regular hexagon with vertices at
        (L,0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L,0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2).
        
        The sides have outward normals at angles: 30, 90, 150, 210, 270, 330 degrees.
        For a point to be inside, its projection onto each outward normal must be <= L.
        Wait, that's not right either. Let me use the support function approach.
        
        For a convex polygon, the minimum enclosing hexagon of a given orientation
        has side length = max over all vertices of the inner shapes of (support_function / 1)
        where support_function in direction n is the max dot product with n.
        
        For a regular hexagon of side L, the support function in direction at angle phi is:
        L * (cos(phi - k*pi/3) for the appropriate k) ... 
        
        Actually, simpler: a regular hexagon of side L centered at origin with flat-top
        has the property that a point (x,y) is inside iff:
        |y| <= L*sqrt(3)/2
        |x| + |y|/sqrt(3) <= L  (for the appropriate quadrant)
        
        Let me just use the vertex-based approach: the enclosing hexagon of side L
        (flat-top, centered at origin) contains all points whose maximum projection
        onto the 6 face normals is <= L.
        
        Face normals (outward) for flat-top hexagon:
        n0 = (cos(30°), sin(30°)) = (sqrt(3)/2, 1/2)
        n1 = (cos(90°), sin(90°)) = (0, 1)
        n2 = (cos(150°), sin(150°)) = (-sqrt(3)/2, 1/2)
        n3 = (cos(210°), sin(210°)) = (-sqrt(3)/2, -1/2)
        n4 = (cos(270°), sin(270°)) = (0, -1)
        n5 = (cos(330°), sin(330°)) = (sqrt(3)/2, -1/2)
        
        For a point p to be inside the hexagon of side L:
        dot(p, n_i) <= L for all i = 0..5
        
        So L must be >= max over all inner hexagon vertices of max_i dot(v, n_i)
        """
        # Outer hexagon orientation: we'll try a few or fix it
        # For now, fix outer orientation to 0 (flat-top)
        L = 0.0
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for (vx, vy) in verts:
                dx = vx - outer_center[0]
                dy = vy - outer_center[1]
                for k in range(6):
                    theta = math.radians(30 + k * 60)
                    proj = dx * math.cos(theta) + dy * math.sin(theta)
                    if proj > L:
                        L = proj
        return L

    def objective(params):
        """Objective: minimize the required enclosing hexagon side length.
        params: 24 values = 12*(x,y) + 12*angle
        Returns: required_L if no overlap, else penalty + required_L
        """
        centers = [(params[2*i], params[2*i+1]) for i in range(N)]
        angles = list(params[2*N:2*N+N])
        
        # Check for overlaps
        for i in range(N):
            for j in range(i+1, N):
                if hex_overlap(i, j, centers, angles):
                    # Penalty for overlap
                    cx1, cy1 = centers[i]
                    cx2, cy2 = centers[j]
                    dx = cx2 - cx1
                    dy = cy2 - cy1
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = 2.0 * sqrt3  # minimum distance for non-overlap
                    penalty = max(0, min_dist - dist) * 100.0
                    L = required_L(centers, angles)
                    return L + penalty
        
        return required_L(centers, angles)

    # Initial guess: a reasonable packing
    # Place hexagons in a honeycomb-like pattern
    def initial_guess():
        centers = []
        angles = []
        # A 3-4-3 arrangement (approximate)
        positions = [
            (-2.0*sqrt3, -1.5), (-0.0, -1.5), (2.0*sqrt3, -1.5),
            (-3.0*sqrt3, 0.0), (-1.0*sqrt3, 0.0), (1.0*sqrt3, 0.0), (3.0*sqrt3, 0.0),
            (-2.0*sqrt3, 1.5), (-0.0, 1.5), (2.0*sqrt3, 1.5),
            (-1.0*sqrt3, -3.0), (1.0*sqrt3, -3.0),
        ]
        for (px, py) in positions:
            centers.append((px, py))
            angles.append(0.0)
        return centers, angles

    centers0, angles0 = initial_guess()
    params0 = []
    for (cx, cy) in centers0:
        params0.append(cx)
        params0.append(cy)
    for a in angles0:
        params0.append(a)

    # Bounds for differential evolution
    bounds = [(-8, 8)] * (2*N) + [(0, 360)] * N

    # Run global optimization
    best_result = None
    best_value = float('inf')

    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=True,
            init='latinhypercube',
            x0=params0,
            callback=None,
            updating='immediate',
            disp=False,
        )
        
        if time.time() > deadline - 20:
            # Time is running out, return what we have
            if de_result is not None and de_result.fun < best_value:
                best_value = de_result.fun
                best_result = de_result.x
            if best_result is not None:
                centers = [(best_result[2*i], best_result[2*i+1]) for i in range(N)]
                angles = list(best_result[2*N:2*N+N])
                L = required_L(centers, angles)
                inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
                return inner, (0.0, 0.0), L, 0.0

        if de_result.fun < best_value:
            best_value = de_result.fun
            best_result = de_result.x

    except Exception:
        pass

    # Local refinement
    if best_result is not None and time.time() < deadline - 30:
        try:
            local_result = minimize(
                objective,
                best_result,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8},
            )
            if local_result.fun < best_value:
                best_value = local_result.fun
                best_result = local_result.x
        except Exception:
            pass

    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    centers = [(best_result[2*i], best_result[2*i+1]) for i in range(N)]
    angles = list(best_result[2*N:2*N+N])
    L = required_L(centers, angles)
    
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), L, 0.0

# EVOLVE-BLOCK-END
