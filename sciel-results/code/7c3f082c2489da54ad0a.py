"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import random

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi * np.arange(6) / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_to_segment_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    ax, ay = float(ax), float(ay)
    bx, by = float(bx), float(by)
    px, py = float(px), float(py)
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return np.sqrt((px - ax)**2 + (py - ay)**2)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return np.sqrt((px - qx)**2 + (py - qy)**2)

def min_dist_to_hex_boundary(points, hex_verts):
    """Minimum distance from any point to the boundary of the hexagon defined by hex_verts (6 points)."""
    min_d = np.inf
    n = len(hex_verts)
    for i in range(n):
        ax, ay = hex_verts[i]
        bx, by = hex_verts[(i+1) % n]
        for px, py in points:
            d = point_to_segment_dist(px, py, ax, ay, bx, by)
            if d < min_d:
                min_d = d
    return min_d

def min_dist_between_hexagons(verts1, verts2):
    """Minimum distance between two convex polygons (hexagons)."""
    min_d = np.inf
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        ax, ay = verts1[i]
        bx, by = verts1[(i+1) % n1]
        for j in range(n2):
            # Check distance from edge of poly1 to vertices of poly2
            for px, py in verts2:
                d = point_to_segment_dist(px, py, ax, ay, bx, by)
                if d < min_d:
                    min_d = d
            # Check distance from edge of poly2 to vertices of poly1
            for px, py in verts1:
                cx, cy = verts2[j]
                dx, dy = verts2[(j+1) % n2]
                d = point_to_segment_dist(px, py, cx, cy, dx, dy)
                if d < min_d:
                    min_d = d
    return min_d

def check_overlap(verts1, verts2):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    edges = []
    for i in range(len(verts1)):
        ax, ay = verts1[i]
        bx, by = verts1[(i+1) % len(verts1)]
        edges.append((bx - ax, by - ay))
    for i in range(len(verts2)):
        ax, ay = verts2[i]
        bx, by = verts2[(i+1) % len(verts2)]
        edges.append((bx - ax, by - ay))
    
    for ex, ey in edges:
        # Normal
        nx, ny = -ey, ex
        # Project both polygons onto this normal
        proj1 = [ax * nx + ay * ny for ax, ay in verts1]
        proj2 = [ax * nx + ay * ny for ax, ay in verts2]
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        if max1 < min2 or max2 < min1:
            return False
    return True

def construct():
    """
    Construct a packing of N unit regular hexagons inside a regular hexagon,
    minimizing the outer side length.
    """
    # Initial configuration: hexagons in a honeycomb-like arrangement
    # For N hexagons, arrange in a compact cluster
    # Start with a reasonable guess
    
    # Generate initial positions in a roughly hexagonal cluster
    positions = []
    angles = []
    
    # Hexagonal lattice arrangement
    # For a hexagonal cluster of side k, we have 3k^2 - 3k + 1 hexagons
    # Let's just place them in a grid-like pattern initially
    side = 1.0
    # Distance between centers of adjacent hexagons in honeycomb: 2 * side * cos(30) = sqrt(3) * side
    dx = np.sqrt(3) * side
    dy = 1.5 * side
    
    count = 0
    for row in range(-5, 6):
        for col in range(-5, 6):
            if count >= N:
                break
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y])
            angles.append(0.0)
            count += 1
        if count >= N:
            break
    
    # Ensure we have exactly N positions
    while len(positions) < N:
        positions.append([random.uniform(-5, 5), random.uniform(-5, 5)])
        angles.append(0.0)
    positions = positions[:N]
    angles = angles[:N]
    
    def evaluate(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_side]
        Returns: outer_side (to minimize) subject to constraints
        """
        outer_side = params[-1]
        if outer_side < 0.1:
            return 1e6
        
        inner_hexes = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            verts = hexagon_vertices(x, y, 1.0, a)
            inner_hexes.append(verts)
        
        outer_verts = hexagon_vertices(0.0, 0.0, outer_side, 0.0)
        
        # Check if all inner hexagons are inside the outer hexagon
        # and don't overlap each other
        penalty = 0.0
        
        for i in range(N):
            # Check if inner hexagon i is inside outer hexagon
            # All vertices must be inside
            for vx, vy in inner_hexes[i]:
                # Check if point is inside outer hexagon using winding number or ray casting
                # Simpler: check distance to each edge
                inside = True
                for j in range(6):
                    ax, ay = outer_verts[j]
                    bx, by = outer_verts[(j+1) % 6]
                    # Cross product to check side
                    cross = (bx - ax) * (vy - ay) - (by - ay) * (vx - ax)
                    if cross > 1e-9:  # Outside (assuming CCW ordering)
                        inside = False
                        break
                if not inside:
                    # Penalty: distance from point to outer boundary
                    d = min_dist_to_hex_boundary([(vx, vy)], outer_verts)
                    penalty += max(0, -d) ** 2
                    break
            
            # Check overlap with other inner hexagons
            for j in range(i + 1, N):
                if check_overlap(inner_hexes[i], inner_hexes[j]):
                    # Overlap penalty
                    d = min_dist_between_hexagons(inner_hexes[i], inner_hexes[j])
                    penalty += max(0, -d) ** 2
        
        return outer_side + 100.0 * penalty
    
    # Multi-start optimization
    best_result = None
    best_score = np.inf
    
    n_starts = 5
    for start in range(n_starts):
        if start == 0:
            # Use initial positions
            x0 = []
            for i in range(N):
                x0.extend([positions[i][0], positions[i][1], angles[i]])
            # Estimate outer side
            max_dist = max(np.sqrt(p[0]**2 + p[1]**2) for p in positions)
            x0.append(max_dist + 3.0)
        else:
            # Random perturbation
            x0 = []
            for i in range(N):
                x = positions[i][0] + random.gauss(0, 2)
                y = positions[i][1] + random.gauss(0, 2)
                a = random.uniform(0, 60)
                x0.extend([x, y, a])
            max_dist = max(np.sqrt(p[0]**2 + p[1]**2) for p in positions)
            x0.append(max_dist + 3.0 + random.uniform(0, 2))
        
        try:
            result = minimize(evaluate, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: return the initial configuration with a large outer hexagon
        return [[positions[i][0], positions[i][1], angles[i]] for i in range(N)], (0.0, 0.0), 40.0, 0.0
    
    # Extract final parameters
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i + 1]
        a = best_result[3*i + 2]
        inner.append([float(x), float(y), float(a)])
    
    outer_side = float(best_result[-1])
    
    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is for compatibility)
# EVOLVE-BLOCK-END
