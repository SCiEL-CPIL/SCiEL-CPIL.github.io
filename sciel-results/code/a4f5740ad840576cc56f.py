"""Rewritten program for unit regular hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    # Precompute outer hexagon edge normals for rotation 0 (pointy-top)
    # Edges are perpendicular to directions at 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([
        [math.cos(math.radians(30 + 60 * m)), math.sin(math.radians(30 + 60 * m))]
        for m in range(6)
    ])
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Return 6 vertices of a regular hexagon."""
        verts = []
        for k in range(6):
            ang = math.radians(angle_deg + 60 * k)
            verts.append([cx + side * math.cos(ang), cy + side * math.sin(ang)])
        return np.array(verts)
    
    def support_function(verts, direction):
        """Support function: max dot product of vertices with direction."""
        return np.max(verts @ direction)
    
    def check_overlap(verts_i, verts_j):
        """
        Check if two hexagons overlap using support function distance.
        Returns 0 if disjoint, positive value if overlapping.
        """
        # Compute all 12 edge normals (6 from each hexagon, both directions)
        # Actually, we need to check: for all directions u, h_i(u) + h_j(-u) <= 0
        # where h is the support function relative to each hexagon's center
        # But we can use the vertices directly
        
        # The minimum distance between two convex polygons can be found by
        # checking all pairs of edges. For two hexagons, we check 12 edge directions.
        
        # Simpler approach: compute the minimum distance between any vertex of one
        # and any edge of the other, and vice versa.
        
        # Actually, for the penalty, we can use a simpler metric:
        # The overlap is positive if the minimum distance between the two sets is negative.
        # We approximate by checking if any vertex of one is inside the other.
        
        # For a robust check, let's use the fact that two convex polygons are disjoint
        # iff there exists a separating axis. We check all 12 edge normals.
        
        # Get edge normals for hexagon i
        normals_i = []
        for k in range(6):
            v1 = verts_i[k]
            v2 = verts_i[(k + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            normals_i.append(normal)
        
        normals_j = []
        for k in range(6):
            v1 = verts_j[k]
            v2 = verts_j[(k + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            normals_j.append(normal)
        
        all_normals = normals_i + normals_j
        
        max_overlap = 0
        for normal in all_normals:
            # Project both hexagons onto this normal
            proj_i = verts_i @ normal
            proj_j = verts_j @ normal
            
            min_i = np.min(proj_i)
            max_i = np.max(proj_i)
            min_j = np.min(proj_j)
            max_j = np.max(proj_j)
            
            # If intervals don't overlap, we found a separating axis
            if max_i < min_j or max_j < min_i:
                return 0  # Disjoint
            
            # Overlap in this direction
            overlap = min(max_i, max_j) - max(min_i, min_j)
            if overlap > max_overlap:
                max_overlap = overlap
        
        return max_overlap
    
    def objective(params):
        n = len(params) // 3
        pos = np.array(params[:2 * n]).reshape(n, 2)
        angs = params[2 * n:3 * n]
        cx = params[3 * n]
        cy = params[3 * n + 1]
        s = params[3 * n + 2]
        outer_ang = params[3 * n + 3]
        
        if s < 2.0:
            return 1e6
        
        # Compute vertices for all inner hexagons
        all_verts = []
        for i in range(n):
            verts = hex_vertices(pos[i][0], pos[i][1], angs[i], 1.0)
            all_verts.append(verts)
        
        # Check pairwise overlaps
        total_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                overlap = check_overlap(all_verts[i], all_verts[j])
                if overlap > 0:
                    total_overlap += overlap * 10
        
        if total_overlap > 0.01:
            return 1e6 + total_overlap * 100
        
        # Check containment in outer hexagon
        # Outer hexagon: center (cx, cy), side s, rotation outer_ang
        # Inradius = s * sqrt(3) / 2
        inradius_outer = s * SQRT3 / 2
        
        # Rotate outer normals by outer_ang
        cos_a = math.cos(math.radians(outer_ang))
        sin_a = math.sin(math.radians(outer_ang))
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_normals = outer_normals @ rot_matrix.T
        
        containment_violation = 0
        for i in range(n):
            for vertex in all_verts[i]:
                dx = vertex[0] - cx
                dy = vertex[1] - cy
                for normal in rotated_normals:
                    proj = dx * normal[0] + dy * normal[1]
                    if proj > inradius_outer:
                        containment_violation += (proj - inradius_outer) * 10
        
        if containment_violation > 0.01:
            return 1e6 + containment_violation * 100
        
        return s
    
    # Initial heuristic: hexagonal packing
    positions = []
    angles = []
    rows = [3, 4, 4]
    y_offset = 0.0
    for i, count in enumerate(rows):
        x_start = -(count - 1) * SQRT3 / 2
        for j in range(count):
            x = x_start + j * SQRT3
            y = y_offset
            positions.append([x, y])
            angles.append(0.0)
        y_offset += 1.5
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx0
        p[1] -= cy0
    
    # Initial side length estimate
    max_dist = max(math.hypot(p[0], p[1]) for p in positions)
    s_init = (max_dist + 1.0) * 2 / SQRT3 + 0.5
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
    for i in range(N):
        bounds.append((-30, 30))  # angle (degrees)
    bounds.append((-5, 5))  # cx
    bounds.append((-5, 5))  # cy
    bounds.append((3.0, 12.0))  # s
    bounds.append((-30, 30))  # outer_ang
    
    # Initial guess
    x0 = []
    for i in range(N):
        x0.append(positions[i][0])
        x0.append(positions[i][1])
    for i in range(N):
        x0.append(angles[i])
    x0.append(0.0)  # cx
    x0.append(0.0)  # cy
    x0.append(s_init)
    x0.append(0.0)  # outer_ang
    
    # Multi-start differential evolution
    best_result = None
    best_fun = float('inf')
    
    for restart in range(3):
        seed = 42 + restart * 100
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=seed,
            polish=True
        )
        
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
    
    if best_result is None or best_result.fun > 100:
        # Fallback to heuristic
        max_dist = max(math.hypot(p[0], p[1]) for p in positions)
        s = (max_dist + 1.0) * 2 / SQRT3 + 0.3
        cx = 0.0
        cy = 0.0
        pos = positions
        angs = angles
        outer_ang = 0.0
    else:
        best_params = best_result.x
        pos = []
        for i in range(N):
            pos.append([best_params[2 * i], best_params[2 * i + 1]])
        angs = [best_params[2 * N + i] for i in range(N)]
        cx = best_params[3 * N]
        cy = best_params[3 * N + 1]
        s = best_params[3 * N + 2]
        outer_ang = best_params[3 * N + 3]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
    
    return inner, (cx, cy), s, outer_ang
# EVOLVE-BLOCK-END
