"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 12
TIME_LIMIT = 115.0  # safety margin below 120s (evaluator kills at 150s)

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def construct():
    t0 = time.time()
    random.seed(42)
    
    def generate_rigid_clusters():
        """Generate candidate rigid clusters of 12 unit hexagons."""
        clusters = []
        
        # Cluster 1: 3x4 Hexagonal Lattice
        # Rows of 4, 4, 4. Offset odd rows.
        row_h = math.sqrt(3)
        col_w = 2.0
        pts = []
        for r in range(3):
            for c in range(4):
                x = c * col_w + (col_w / 2 if r % 2 else 0.0)
                y = r * row_h
                # Center the cluster
                pts.append((x - 3.0, y - row_h, 0.0))
        clusters.append(pts)
        
        # Cluster 2: 2x6 Hexagonal Lattice
        # Rows of 6, 6.
        pts = []
        for r in range(2):
            for c in range(6):
                x = c * col_w + (col_w / 2 if r % 2 else 0.0)
                y = r * row_h
                pts.append((x - 5.0, y - row_h/2, 0.0))
        clusters.append(pts)
        
        # Cluster 3: "Flower" or Compact
        # Center + 6 neighbors + 5 more? 
        # Let's try a 2-3-3-2-2 or similar compact shape.
        # Actually, 3x4 is likely best for hexagons due to aspect ratio matching.
        # Let's add a slightly perturbed 3x4 to allow for "breathing".
        pts_perturbed = [p[:] for p in pts]
        # Don't actually perturb for rigid search, but keep as distinct candidate if we want.
        
        return clusters

    def cluster_side_length(cluster, global_rotation_deg, container_rotation_deg):
        """
        Calculate the minimum side length of the container for a given cluster
        and rotations.
        cluster: list of (x, y, angle) relative to cluster center.
        global_rotation_deg: rotation of the entire cluster.
        container_rotation_deg: rotation of the container.
        """
        # Rotate cluster points
        g_rot = math.radians(global_rotation_deg)
        cg, sg = math.cos(g_rot), math.sin(g_rot)
        
        rotated_points = []
        for x, y, a in cluster:
            rx = cg * x - sg * y
            ry = sg * x + cg * y
            rotated_points.append((rx, ry, a + global_rotation_deg))
        
        # Center of mass of the rotated cluster (should be 0 if cluster is centered)
        # We assume cluster is centered at (0,0).
        
        # Container normals
        c_rot = math.radians(container_rotation_deg)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = c_rot + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        max_need = 0.0
        for px, py, pangle in rotated_points:
            verts = _hex_vertices(px, py, pangle)
            for (vx, vy) in verts:
                for (nx, ny) in normals:
                    val = (nx * vx + ny * vy) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    def objective_relative_angle(rel_angle_deg):
        """
        Objective function for 1D search.
        We fix cluster rotation to 0 and vary container rotation,
        or vice versa. Due to symmetry, it's equivalent to varying relative angle.
        Let's fix cluster at 0 deg, and vary container rotation.
        """
        # rel_angle_deg is the rotation of the container relative to the cluster.
        # Since cluster is fixed at 0, container is at rel_angle_deg.
        return cluster_side_length(best_cluster, 0.0, rel_angle_deg)

    # 1. Generate and evaluate rigid clusters
    best_cluster = None
    best_score = float('inf')
    best_config = None
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    clusters = generate_rigid_clusters()
    
    for cluster in clusters:
        # Perform 1D search over relative angle.
        # The function is periodic with period 30 degrees (6-fold symmetry of hex).
        # We search in [0, 30].
        
        def obj(x):
            return cluster_side_length(cluster, 0.0, x)
        
        res = minimize_scalar(obj, bounds=(0, 30), method='bounded', options={'xatol': 1e-6})
        
        if res.success:
            best_rel_angle = res.x
            score = res.fun
            
            # Reconstruct the full configuration
            # Cluster is rotated by 0, container by best_rel_angle.
            # We need to output inner hexagons and container params.
            # Inner hexagons are in global coordinates.
            # Since we fixed cluster rotation to 0, their positions are just the cluster points.
            # However, we might want to rotate the whole system to a "canonical" orientation?
            # The score is invariant to global rotation.
            # Let's keep container at best_rel_angle and cluster at 0.
            
            inner = [[float(x), float(y), float(a)] for x, y, a in cluster]
            center = [0.0, 0.0]
            angle = best_rel_angle
            
            if score < best_score:
                best_score = score
                best_cluster = cluster
                best_config = inner
                best_center = center
                best_angle = angle

    # 2. Local Polish with Nelder-Mead
    # We take the best rigid configuration and allow small adjustments to positions and angles.
    # This can help if the rigid assumption is slightly too restrictive.
    
    if best_config is not None:
        x0 = []
        for x, y, a in best_config:
            x0.extend([x, y, a])
        x0.extend([best_center[0], best_center[1], best_angle])
        
        def nm_objective(x):
            cfg = []
            for i in range(N):
                cfg.append([x[3*i], x[3*i+1], x[3*i+2]])
            center = (x[3*N], x[3*N+1])
            angle = x[3*N+2]
            
            # Check validity
            if not _is_valid(cfg):
                return 100.0 + _outer_hex_side(cfg, center, angle)
            
            return _outer_hex_side(cfg, center, angle)
        
        # Run Nelder-Mead
        try:
            res_nm = minimize(nm_objective, x0, method='Nelder-Mead', 
                              options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000, 'maxfev': 1000})
            
            if res_nm.fun < best_score:
                # Update best
                best_score = res_nm.fun
                cfg = []
                for i in range(N):
                    cfg.append([res_nm.x[3*i], res_nm.x[3*i+1], res_nm.x[3*i+2]])
                best_config = cfg
                best_center = [res_nm.x[3*N], res_nm.x[3*N+1]]
                best_angle = res_nm.x[3*N+2]
        except Exception as e:
            pass # Fallback to rigid solution

    # 3. Final Check and Return
    # Ensure we have a valid solution
    if best_config is None:
        # Fallback to a simple grid
        row_h = math.sqrt(3)
        col_w = 2.0
        pts = []
        for r in range(3):
            for c in range(4):
                x = c * col_w + (col_w / 2 if r % 2 else 0.0)
                y = r * row_h
                pts.append([x - 3.0, y - row_h, 0.0])
        best_config = pts
        best_center = [0.0, 0.0]
        best_angle = 0.0
        best_score = _outer_hex_side(best_config, best_center, best_angle)
        
    # Convert to required format
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_config]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
