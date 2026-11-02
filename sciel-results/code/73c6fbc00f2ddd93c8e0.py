import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # --- Hexagon geometry helpers ---
    def hex_vertices(cx, cy, side, angle):
        """Return 6 vertices of a regular hexagon with given center, side, rotation."""
        verts = []
        for k in range(6):
            theta = angle + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def point_in_hex(px, py, cx, cy, side, angle):
        """Check if point (px,py) is inside or on boundary of hexagon."""
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        ca = math.cos(-angle)
        sa = math.sin(-angle)
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        # For a unit hexagon (side=1), the apothem is sqrt(3)/2
        # The hexagon in local frame has vertices at angles k*pi/3
        # Check against all 6 half-planes
        apothem = side * math.sqrt(3) / 2
        for k in range(6):
            theta = k * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            if lx * nx + ly * ny > apothem:
                return False
        return True

    def hexagons_disjoint(v1, v2, eps=1e-7):
        """
        Check if two hexagons (as lists of 6 vertices) are disjoint (no interior overlap).
        They may touch at edges/vertices.
        Uses: hexagons are disjoint iff no vertex of one is strictly inside the other.
        """
        # Check if any vertex of v1 is strictly inside v2
        for i in range(6):
            px, py = v1[i]
            # Use strict interior test: point must be inside with margin
            # For hex2, we need its center and angle. But we only have vertices.
            # Alternative: use edge intersection + point-in-polygon.
            pass
        # Simpler: check if any vertex of v1 is inside v2 (strictly)
        # We need center and angle of v2 for point_in_hex. Let's pass those.
        # Actually, let's restructure: pass centers and angles.
        return True  # placeholder

    def hexagons_disjoint_cc(c1x, c1y, a1, c2x, c2y, a2, eps=1e-7):
        """
        Check if two unit hexagons with given centers/angles are disjoint.
        Two unit hexagons are disjoint iff no vertex of one is strictly inside the other.
        """
        v1 = hex_vertices(c1x, c1y, 1.0, a1)
        v2 = hex_vertices(c2x, c2y, 1.0, a2)
        # Check v1 vertices against v2
        for px, py in v1:
            if point_in_hex(px, py, c2x, c2y, 1.0, a2):
                # Check if strictly inside (not on boundary)
                # For simplicity, if point_in_hex returns True, we consider it overlapping
                # unless it's very close to boundary. For safety, treat as overlap.
                return False
        # Check v2 vertices against v1
        for px, py in v2:
            if point_in_hex(px, py, c1x, c1y, 1.0, a1):
                return False
        return True

    def all_disjoint(positions, angles, eps=1e-7):
        """Check all pairs of inner hexagons are disjoint."""
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if not hexagons_disjoint_cc(positions[i][0], positions[i][1], angles[i],
                                            positions[j][0], positions[j][1], angles[j], eps):
                    return False
        return True

    def inner_in_outer(positions, angles, outer_cx, outer_cy, outer_side, outer_angle):
        """Check all inner hexagons are inside the outer hexagon."""
        for i in range(len(positions)):
            verts = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for px, py in verts:
                if not point_in_hex(px, py, outer_cx, outer_cy, outer_side, outer_angle):
                    return False
        return True

    def min_outer_side(positions, angles):
        """
        Compute the minimum side length of a regular hexagon (centered at origin,
        rotatable) that contains all inner hexagons.
        """
        # The outer hexagon's rotation can be varied. For a given outer rotation alpha,
        # the constraint is that all inner vertices must satisfy:
        #   v . n_k <= S * sqrt(3)/2  for k=0..5
        # where n_k are the outward normals at angles alpha + k*pi/3.
        # So S >= (2/sqrt(3)) * max_{i, vertex, k} v . n_k
        # We minimize over alpha in [0, pi/3] (6-fold symmetry).
        
        best_S = float('inf')
        # Sample alpha
        for alpha_deg in range(0, 60, 1):
            alpha = alpha_deg * math.pi / 180.0
            outer_normals = [alpha + k * math.pi / 3 for k in range(6)]
            max_dot = 0.0
            for i in range(len(positions)):
                cx, cy = positions[i]
                verts = hex_vertices(cx, cy, 1.0, angles[i])
                for vx, vy in verts:
                    for n_angle in outer_normals:
                        dot = vx * math.cos(n_angle) + vy * math.sin(n_angle)
                        if dot > max_dot:
                            max_dot = dot
            S = 2.0 * max_dot / math.sqrt(3)
            if S < best_S:
                best_S = S
        return best_S

    def objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
        Returns outer side length + penalty for overlap.
        """
        positions = []
        angles = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            positions.append((x, y))
            angles.append(a)
        
        # Check disjointness
        if not all_disjoint(positions, angles):
            return 1e6  # heavy penalty
        
        return min_outer_side(positions, angles)

    # --- Initial guess: hexagonal lattice arrangement ---
    # Use axial coordinates for a compact cluster of 11 hexagons.
    # Hex grid: distance between centers of touching unit hexagons is 2 (since side=1, 
    # the distance from center to vertex is 1, so center-to-center for edge-sharing is 2).
    # Wait: for regular hexagons with side s, the distance between centers of two 
    # edge-adjacent hexagons is 2*s*sqrt(3)/2 = s*sqrt(3). Let me verify.
    # A regular hexagon with side s has circumradius s and inradius s*sqrt(3)/2.
    # Two hexagons sharing an edge: the distance between centers is 2 * inradius = s*sqrt(3).
    # So center-to-center distance for touching hexagons is sqrt(3) ≈ 1.732.
    
    # Axial to cartesian for flat-top hexagons with side 1:
    # x = 1.5 * q
    # y = sqrt(3)/2 * q + sqrt(3) * r
    # But this gives center-to-center distance of sqrt(3) for neighbors? Let's check.
    # Neighbors in axial: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (0,0) -> (0,0)
    # (1,0) -> (1.5, sqrt(3)/2)
    # Distance: sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25 + 0.75) = sqrt(3) ✓
    
    hex_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-2, 0)
    ]
    
    init_positions = []
    for q, r in hex_coords:
        x = 1.5 * q
        y = math.sqrt(3) / 2 * q + math.sqrt(3) * r
        init_positions.append((x, y))
    
    init_angles = [0.0] * N
    
    x0 = []
    for p, a in zip(init_positions, init_angles):
        x0.append(p[0])
        x0.append(p[1])
        x0.append(a)
    
    x0 = np.array(x0)
    
    # --- Optimization ---
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 2 * math.pi))
    
    best_result = None
    best_value = float('inf')
    
    # Nelder-Mead from initial guess
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_value and res.fun < 100:
                best_value = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # Random restarts
    restarts = 8
    for r in range(restarts):
        if time.time() > deadline - 10:
            break
        x_pert = x0.copy()
        for i in range(N):
            x_pert[i*3] += random.gauss(0, 0.5)
            x_pert[i*3+1] += random.gauss(0, 0.5)
            x_pert[i*3+2] = random.uniform(0, 2*math.pi)
        try:
            res = minimize(objective, x_pert, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_value and res.fun < 100:
                best_value = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # If we found a valid solution, use it
    if best_result is not None and best_value < 50.0:
        inner = []
        for i in range(N):
            x = best_result[i * 3]
            y = best_result[i * 3 + 1]
            a = best_result[i * 3 + 2]
            a_deg = math.degrees(a) % 360.0
            inner.append([x, y, a_deg])
        
        # Compute the actual outer side by optimizing outer rotation more finely
        positions = []
        angles = []
        for i in range(N):
            positions.append((best_result[i*3], best_result[i*3+1]))
            angles.append(best_result[i*3+2])
        
        # Fine-tune outer rotation
        best_S = float('inf')
        best_alpha = 0.0
        for alpha_deg in np.linspace(0, 60, 601):
            alpha = alpha_deg * math.pi / 180.0
            outer_normals = [alpha + k * math.pi / 3 for k in range(6)]
            max_dot = 0.0
            for i in range(N):
                cx, cy = positions[i]
                verts = hex_vertices(cx, cy, 1.0, angles[i])
                for vx, vy in verts:
                    for n_angle in outer_normals:
                        dot = vx * math.cos(n_angle) + vy * math.sin(n_angle)
                        if dot > max_dot:
                            max_dot = dot
            S = 2.0 * max_dot / math.sqrt(3)
            if S < best_S:
                best_S = S
                best_alpha = alpha
        
        outer_center = (0.0, 0.0)
        outer_side = best_S
        outer_angle_deg = math.degrees(best_alpha) % 360.0
        
        return inner, outer_center, outer_side, outer_angle_deg
    
    # Fallback: linear row
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
