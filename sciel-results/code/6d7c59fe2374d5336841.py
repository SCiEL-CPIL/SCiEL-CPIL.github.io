"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """
        Compute the minimum side length S of the outer hexagon.
        Vectorized implementation.
        """
        # Transform all vertices to container's local coordinate system
        # vertices_list is a list of N arrays, each 6x2
        all_verts = np.vstack(vertices_list) # Shape (6*N, 2)
        
        lx = all_verts[:, 0] - ocx
        ly = all_verts[:, 1] - ocy
        
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        
        # Rotate by -oangle
        rx = lx * cos_a + ly * sin_a
        ry = -lx * sin_a + ly * cos_a
        
        # Project onto 6 normals
        # Normals are at pi/6 + k*pi/3
        thetas = np.pi / 6 + np.arange(6) * np.pi / 3
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        
        # Projections shape (6*N, 6)
        projs = np.outer(rx, cos_thetas) + np.outer(ry, sin_thetas)
        
        # Max absolute projection for each normal
        max_abs_projs = np.max(np.abs(projs), axis=0)
        
        # The side length S is determined by the max projection / apothem
        apothem = np.sqrt(3) / 2
        S = np.max(max_abs_projs) / apothem
        
        return S

    def overlap_penalty(v1, v2):
        """
        Calculate a smooth-ish penalty for overlap between two hexagons.
        Returns 0 if disjoint, positive value if overlapping.
        Uses SAT to find the minimum separation (negative if overlapping).
        """
        # Axes are normals to the edges of both polygons
        # For regular hexagons, the normals are at angles 0, 60, 120, 180, 240, 300 degrees?
        # No, vertices are at 0, 60... Edges are between them.
        # Normal to edge between vertex 0 and 1 (angles 0 and 60) is at 30 degrees.
        # So normals are at 30, 90, 150, 210, 270, 330.
        # Which is pi/6 + k*pi/3.
        
        # We can precompute the normal vectors for unit hexagons, but they rotate.
        # v1 and v2 are 6x2 arrays.
        
        # Get edge vectors for v1
        edges1 = np.roll(v1, -1, axis=0) - v1
        # Normals are perpendicular to edges.
        # For a CCW polygon, normal is (edge_y, -edge_x) or (-edge_y, edge_x)?
        # Let's use (-edge_y, edge_x) and normalize?
        # Actually, for regular hexagons, the normals are fixed relative to the rotation angle.
        # It is faster to just use the 6 normals of v1 and 6 normals of v2.
        
        # Normals for v1:
        # v1 is generated from angle a1. The normals are at a1 + pi/6 + k*pi/3.
        # Instead of recalculating from vertices, we can just use the vertex differences.
        
        axes = []
        for v in [v1, v2]:
            edges = np.roll(v, -1, axis=0) - v
            # Normal vector
            normals = np.column_stack([-edges[:, 1], edges[:, 0]])
            # Normalize
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / norms
            axes.append(normals)
        
        all_axes = np.vstack(axes) # 12 x 2
        
        # Project v1 and v2 onto all axes
        # proj1 = v1 @ all_axes.T -> 6x12
        # proj2 = v2 @ all_axes.T -> 6x12
        
        proj1 = v1 @ all_axes.T
        proj2 = v2 @ all_axes.T
        
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # Overlap on each axis:
        # If max1 < min2 or max2 < min1, no overlap on this axis.
        # Overlap amount = min(max1, max2) - max(min1, min2)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        # If any axis has overlap <= 0, they are disjoint (separated on that axis)
        # Wait, SAT says: if there is ANY separating axis, they are disjoint.
        # So, if min(overlaps) <= 0, they are disjoint?
        # No. If they are disjoint, there is at least one axis where the intervals do NOT overlap.
        # That means for that axis, overlap <= 0.
        # So, if np.min(overlaps) <= 0, they are disjoint.
        
        # If they are overlapping, ALL axes must have overlap > 0.
        # The "depth" of overlap is often defined as the minimum of the overlaps?
        # Or the sum?
        # For a penalty, we want a positive value if overlapping.
        # If min(overlaps) > 0, they overlap.
        # The magnitude of overlap can be approximated by the minimum overlap (narrowest gap) or sum.
        # Let's use the sum of overlaps as a robust penalty, or just the min overlap.
        # Min overlap is the "bottleneck" axis.
        
        min_overlap = np.min(overlaps)
        
        if min_overlap <= 0:
            return 0.0
        else:
            # Smooth penalty: square of the penetration depth
            return min_overlap ** 2

    def objective(params):
        """
        Compute the outer side length S and penalty for overlaps.
        params layout:
        [0:3]   outer center x, y, angle
        [3:3+3N] inner hexagons (cx, cy, angle)
        """
        ocx = params[0]
        ocy = params[1]
        oangle = params[2]
        
        vertices_list = []
        for i in range(N):
            cx = params[3 + 3 * i]
            cy = params[3 + 3 * i + 1]
            angle = params[3 + 3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Quick bounding box check to skip SAT if far apart
                # Bounding box of hexagon is roughly 2x2
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                if np.linalg.norm(c1 - c2) > 2.2:
                    continue
                penalty += overlap_penalty(vertices_list[i], vertices_list[j])
        
        return S + 10.0 * penalty # Scale penalty

    def get_initial_params():
        p = np.zeros(3 + 3 * N)
        # Outer
        p[0:3] = [0, 0, 0]
        
        # Inner
        idx = 0
        # Center
        p[3 + 3*idx : 3 + 3*idx + 3] = [0, 0, 0]
        idx += 1
        # Ring 1
        for k in range(6):
            theta = k * np.pi / 3
            cx = np.sqrt(3) * np.cos(theta)
            cy = np.sqrt(3) * np.sin(theta)
            p[3 + 3*idx : 3 + 3*idx + 3] = [cx, cy, 0]
            idx += 1
        # Ring 2 (4 corners of the hexagon shape)
        # Standard hex packing of 19 hexes has rings 1, 2, 3?
        # 1 + 6 + 12 = 19.
        # We only have 11.
        # 1 + 6 + 4 = 11.
        # The 4 in ring 2 should be placed to minimize S.
        # In a hex lattice, the neighbors of the center are at distance sqrt(3).
        # The next ring is at distance 2*sqrt(3).
        # Angles for ring 2 neighbors in a hex grid (aligned with ring 1):
        # Ring 1 is at 0, 60, 120, 180, 240, 300.
        # Ring 2 positions that are "corners" of the hex shape:
        # 0, 120, 240 are aligned with ring 1 vertices?
        # Let's check coordinates.
        # Center (0,0).
        # Ring 1: (sqrt(3), 0), (sqrt(3)/2, 1.5), ...
        # Ring 2: (2*sqrt(3), 0), (sqrt(3), 3), ...
        # The 6 ring 2 positions are at distance 2*sqrt(3) and angles 0, 60, 120, 180, 240, 300.
        # Wait, in a hex lattice, the "corners" of the 7-hex cluster are the ring 1 vertices.
        # The 19-hex cluster has ring 2.
        # If we pick 4 from ring 2, which 4?
        # To make a compact shape, maybe pick 0, 120, 240 and one more?
        # Or 0, 60, 120, 180?
        # Let's try 0, 120, 240 (3) and one more, say 0? No, distinct.
        # Let's try 0, 120, 240 and 180?
        # Actually, the current code uses [0, 120, 240, 180].
        angles_r2 = [0, 120, 240, 180]
        for deg in angles_r2:
            theta = np.radians(deg)
            cx = 2 * np.sqrt(3) * np.cos(theta)
            cy = 2 * np.sqrt(3) * np.sin(theta)
            p[3 + 3*idx : 3 + 3*idx + 3] = [cx, cy, 0]
            idx += 1
        return p

    best_params = get_initial_params()
    best_score = objective(best_params)
    
    # Multi-start optimization with Gradient-Based Solver (L-BFGS-B)
    np.random.seed(42)
    
    # Bounds for inner hexagons to keep them in a reasonable area
    # Outer side length is likely around 4-5.
    # Center of outer is 0,0.
    # Inner hexes should be within radius ~ 4.
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (-np.pi, np.pi)] # Outer
    for i in range(N):
        bounds.append((-4, 4)) # cx
        bounds.append((-4, 4)) # cy
        bounds.append((-np.pi/2, np.pi/2)) # angle (symmetry)

    for i in range(20):
        p = best_params.copy()
        if i > 0:
            # Perturb inner positions and rotations
            noise_inner = np.random.normal(0, 0.5, size=3 * N)
            noise_outer = np.random.normal(0, 0.1, size=3)
            p[0:3] += noise_outer
            p[3:] += noise_inner
            
        res = minimize(objective, p, method='L-BFGS-B', 
                      bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x

    # Extract results
    ocx, ocy, oangle = best_params[0], best_params[1], best_params[2]
    inner = []
    for i in range(N):
        cx = best_params[3 + 3 * i]
        cy = best_params[3 + 3 * i + 1]
        angle = best_params[3 + 3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # Recalculate S for the best params to be safe
    vertices_list = []
    for i in range(N):
        cx = best_params[3 + 3 * i]
        cy = best_params[3 + 3 * i + 1]
        angle = best_params[3 + 3 * i + 2]
        verts = hex_vertices(cx, cy, angle)
        vertices_list.append(verts)
        
    S_final = compute_S_for_container(vertices_list, ocx, ocy, oangle)
    
    return inner, (ocx, ocy), S_final, np.degrees(oangle)
# EVOLVE-BLOCK-END
