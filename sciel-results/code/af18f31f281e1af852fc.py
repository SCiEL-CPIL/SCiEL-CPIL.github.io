import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        # Transform point to hexagon's local frame
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a

        # In local frame, hexagon with flat top has vertices at angles 0,60,...,300
        # The hexagon is the intersection of 6 half-planes
        # For a hexagon with vertices at (side*cos(k*pi/3), side*sin(k*pi/3)),
        # the supporting lines are at distance side*sqrt(3)/2 from center
        # in directions perpendicular to the edges

        # Actually, let's use the fact that a regular hexagon with "radius" r (center to vertex)
        # has inradius r*sqrt(3)/2. Here side = r for a regular hexagon.
        # The hexagon is defined by |x'| <= r*sqrt(3)/2 in three directions 60 deg apart

        r = side
        inradius = r * math.sqrt(3) / 2.0

        # Three pairs of parallel supporting lines
        for k in range(3):
            a = k * math.pi / 3.0  # 0, 60, 120 degrees
            # Distance from center to supporting line in direction a
            dist = abs(lx * math.cos(a) + ly * math.sin(a))
            if dist > inradius + 1e-12:
                return False
        return True

    def hexagon_containment_penalty(verts, cx, cy, side, angle_deg):
        """Penalty for hexagon vertices not being inside the outer hexagon."""
        penalty = 0.0
        for (vx, vy) in verts:
            if not point_in_hexagon(vx, vy, cx, cy, side, angle_deg):
                # Compute how far outside
                # Transform to local frame
                dx = vx - cx
                dy = vy - cy
                a_rad = math.radians(angle_deg)
                cos_a = math.cos(a_rad)
                sin_a = math.sin(a_rad)
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a

                r = side
                inradius = r * math.sqrt(3) / 2.0

                max_violation = 0.0
                for k in range(3):
                    a = k * math.pi / 3.0
                    dist = abs(lx * math.cos(a) + ly * math.sin(a))
                    violation = dist - inradius
                    if violation > max_violation:
                        max_violation = violation
                penalty += max_violation ** 2
        return penalty

    def hexagons_overlap(v1, v2):
        """Check if two hexagons overlap using SAT."""
        axes = []
        for i in range(6):
            # Edge normals
            for j in range(6):
                pass
        # Collect all edge normals from both hexagons
        for hex_verts in [v1, v2]:
            for i in range(6):
                p1 = hex_verts[i]
                p2 = hex_verts[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal /= norm
                    axes.append(normal)

        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False  # Separating axis found, no overlap
        return True  # No separating axis, overlap

    def overlap_penalty(verts_list):
        """Compute penalty for overlapping inner hexagons."""
        penalty = 0.0
        n = len(verts_list)
        for i in range(n):
            for j in range(i + 1, n):
                if hexagons_overlap(verts_list[i], verts_list[j]):
                    # Compute minimum translation distance approximation
                    # Use the SAT gap
                    v1, v2 = verts_list[i], verts_list[j]
                    axes = []
                    for hex_verts in [v1, v2]:
                        for k in range(6):
                            p1 = hex_verts[k]
                            p2 = hex_verts[(k + 1) % 6]
                            edge = p2 - p1
                            normal = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm > 1e-12:
                                normal /= norm
                                axes.append(normal)

                    min_gap = float('inf')
                    for axis in axes:
                        proj1 = np.dot(v1, axis)
                        proj2 = np.dot(v2, axis)
                        min1, max1 = proj1.min(), proj1.max()
                        min2, max2 = proj2.min(), proj2.max()
                        gap = max(min1 - max2, min2 - max1)
                        if gap < min_gap:
                            min_gap = gap
                    # min_gap is negative if overlapping
                    if min_gap < 0:
                        penalty += (-min_gap) ** 2
        return penalty

    def objective(params):
        """Objective: minimize outer side length + penalties."""
        # params: [x0,y0,theta0, x1,y1,theta1, ..., x11,y11,theta11, cx, cy, side, oangle]
        inner_params = params[:N * 3]
        cx = params[N * 3]
        cy = params[N * 3 + 1]
        side = params[N * 3 + 2]
        oangle = params[N * 3 + 3]

        if side < 1.0:
            return 1e6

        # Compute vertices for each inner hexagon
        verts_list = []
        for i in range(N):
            xi = inner_params[i * 3]
            yi = inner_params[i * 3 + 1]
            th = inner_params[i * 3 + 2]
            verts = hex_vertices(xi, yi, th)
            verts_list.append(verts)

        # Penalty for overlap
        overlap_pen = overlap_penalty(verts_list)

        # Penalty for containment
        contain_pen = 0.0
        for verts in verts_list:
            contain_pen += hexagon_containment_penalty(verts, cx, cy, side, oangle)

        # Objective: minimize side length, penalize violations
        return side + 100.0 * overlap_pen + 100.0 * contain_pen

    def gradient_free_optimize(x0, method='Nelder-Mead', maxiter=5000):
        """Run optimization with time check."""
        result = None
        best_x = x0.copy()
        best_f = objective(x0)

        def callback(xk):
            nonlocal best_x, best_f
            f = objective(xk)
            if f < best_f:
                best_f = f
                best_x = xk.copy()
            return False  # Continue

        try:
            if method == 'Nelder-Mead':
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6},
                               callback=callback)
                result = res
            elif method == 'L-BFGS-B':
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6},
                               callback=callback)
                result = res
        except Exception:
            pass

        # Return best found
        if result is not None and result.fun < best_f:
            best_x = result.x
            best_f = result.fun
        return best_x, best_f

    # Initial configuration: honeycomb arrangement
    # For unit hexagons (side=1), center-to-center distance for touching is sqrt(3)
    # Honeycomb lattice vectors: a1 = (3/2, sqrt(3)/2), a2 = (0, sqrt(3))
    # Wait, for flat-top hexagons with side 1:
    # The 6 neighbors are at angles 0, 60, 120, 180, 240, 300 degrees at distance sqrt(3)
    # No, let me think more carefully.

    # A regular hexagon with side length s has:
    # - Circumradius (center to vertex) = s
    # - Inradius (center to edge midpoint) = s * sqrt(3)/2
    # Two such hexagons touching edge-to-edge have center-to-center distance = 2 * inradius = s * sqrt(3)

    # So for unit hexagons, touching distance = sqrt(3)

    # Honeycomb lattice: basis vectors
    # d = sqrt(3)
    # a1 = (d, 0) = (sqrt(3), 0)
    # a2 = (d/2, d*sqrt(3)/2) = (sqrt(3)/2, 3/2)
    # Wait, that's not right either. Let me use the standard hexagonal lattice.

    # For a honeycomb of hexagons with flat-top orientation:
    # The centers form a triangular lattice with spacing sqrt(3)
    # Lattice vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)

    # Let me just place 12 hexagons in a compact hexagonal cluster.
    # A perfect hexagonal cluster of 19 has rows 1-2-3-4-3-2-1.
    # For 12, I'll use a subset.

    # Alternative: use a 4x3 rectangular arrangement in the honeycomb lattice
    # or a hexagonal cluster of 12.

    # Let me use a hexagonal cluster: center + 6 neighbors + 5 more from the next ring
    # Positions in the triangular lattice (in units of sqrt(3)):
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  [in lattice coords]
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1)

    # Convert lattice coords to Cartesian:
    # (i, j) -> (sqrt(3)*(i + j/2), 3*j/2)  ... hmm, let me use a cleaner mapping.

    # Triangular lattice with spacing a = sqrt(3):
    # (i, j) -> (a*(i + j/2), a*sqrt(3)*j/2) = (sqrt(3)*(i+j/2), 3*j/2)

    d = math.sqrt(3)

    def lattice_to_cart(i, j):
        return (d * (i + j / 2.0), 3.0 * j / 2.0)

    # 12 hexagons in a compact arrangement:
    # Use rows: row -1: 3, row 0: 4, row 1: 3, row 2: 2
    # Or a hexagonal cluster: center(1) + ring1(6) + 5 from ring2 = 12

    # Let me use: center + 6 neighbors + 5 more
    # Lattice positions for a hexagonal cluster of 12:
    positions_lattice = [
        (0, 0),      # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1 (6)
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),  # 5 from ring 2
    ]

    inner_positions = []
    for (i, j) in positions_lattice:
        x, y = lattice_to_cart(i, j)
        inner_positions.append((x, y, 0.0))  # angle = 0

    # Compute bounding region to estimate outer hexagon size
    all_points = []
    for (x, y, th) in inner_positions:
        verts = hex_vertices(x, y, th)
        all_points.extend(verts)
    all_points = np.array(all_points)

    # Estimate outer hexagon: center at centroid, side large enough
    cx0 = float(np.mean(all_points[:, 0]))
    cy0 = float(np.mean(all_points[:, 1]))

    # Compute required side length: max distance from center to any vertex, then convert
    max_dist = 0.0
    for (vx, vy) in all_points:
        dist = math.hypot(vx - cx0, vy - cy0)
        if dist > max_dist:
            max_dist = dist

    # For a hexagon with center at origin and flat-top, the maximum distance to a vertex
    # in any direction depends on the angle. The "radius" function is:
    # r(theta) = inradius / cos((theta mod 60) - 30)
    # To contain a point at distance d from center at angle phi, we need:
    # side >= d * cos((phi mod 60) - 30) / (sqrt(3)/2) ... this is getting complex.
    # Let me just use a generous initial estimate.
    side0 = max_dist * 1.2 + 1.0
    oangle0 = 0.0

    # Build initial parameter vector
    x0 = []
    for (x, y, th) in inner_positions:
        x0.extend([x, y, th])
    x0.extend([cx0, cy0, side0, oangle0])
    x0 = np.array(x0)

    # Run optimization
    best_x = x0.copy()
    best_f = objective(x0)

    # Try multiple restarts and methods
    rng = np.random.default_rng(42)

    for restart in range(5):
        if time_left() < 20.0:
            break

        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb initial solution
            x_init = x0.copy()
            x_init[:N * 3] += rng.normal(0, 0.5, size=N * 3)
            x_init[N * 3:N * 3 + 2] += rng.normal(0, 0.5, size=2)
            x_init[N * 3 + 2] = max(1.5, x0[N * 3 + 2] * (1 + rng.normal(0, 0.1)))
            x_init[N * 3 + 3] = rng.uniform(-30, 30)

        # Try Nelder-Mead
        x_best, f_best = gradient_free_optimize(x_init, method='Nelder-Mead', maxiter=3000)
        if f_best < best_f:
            best_f = f_best
            best_x = x_best.copy()

        if time_left() < 20.0:
            break

        # Try L-BFGS-B
        x_best2, f_best2 = gradient_free_optimize(x_init, method='L-BFGS-B', maxiter=2000)
        if f_best2 < best_f:
            best_f = f_best2
            best_x = x_best2.copy()

    # Extract best solution
    inner = []
    for i in range(N):
        x = float(best_x[i * 3])
        y = float(best_x[i * 3 + 1])
        th = float(best_x[i * 3 + 2])
        inner.append([x, y, th])

    cx = float(best_x[N * 3])
    cy = float(best_x[N * 3 + 1])
    side = float(best_x[N * 3 + 2])
    oangle = float(best_x[N * 3 + 3])

    # Ensure side is reasonable
    if side < 1.0:
        side = 1.0

    return inner, (cx, cy), side, oangle

# EVOLVE-BLOCK-END
