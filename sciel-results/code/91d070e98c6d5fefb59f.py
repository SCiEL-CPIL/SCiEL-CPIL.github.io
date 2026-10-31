"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i+1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False, axis
    return True, None

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    # Normals for the 6 sides of the outer hexagon
    # The distance from center to side is S * sqrt(3) / 2
    # So S = 2 * d / sqrt(3) where d is max projection onto normal
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                d = np.dot(v - np.array([ox, oy]), n)
                if d > 0:
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
            if overlap:
                return False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            n = len(outer_verts)
            inside = True
            for i in range(n):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                # Check if point is on the same side of edge as the center
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, np.array([ox, oy]) - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def generate_rigid_clusters():
    """Generate several rigid cluster configurations for the 11 hexagons."""
    clusters = []
    
    # Cluster 1: Center + Ring 1 (6) + Ring 2 (4 corners)
    # This is the standard hexagonal lattice packing for 11.
    # Positions are in local coordinates, angles are 0 relative to cluster.
    c1 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        c1.append((x, y, 0.0))
    # Add 4 from ring 2: e.g., at 0, 60, 120, 180 degrees
    for k in range(4):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        c1.append((x, y, 0.0))
    clusters.append(c1)
    
    # Cluster 2: Center + Ring 1 (6) + Ring 2 (4 alternating corners: 30, 90, 150, 210)
    # This creates a different symmetry.
    c2 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        c2.append((x, y, 0.0))
    for k in range(4):
        angle = np.radians(30 + 60 * k)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        c2.append((x, y, 0.0))
    clusters.append(c2)
    
    # Cluster 3: Slightly rotated inner hexagons? No, rigid body means relative angles fixed.
    # Let's try a cluster where the outer 4 are rotated 30 degrees relative to center.
    c3 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        c3.append((x, y, 0.0))
    for k in range(4):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        # Rotate these 4 hexagons by 30 degrees
        c3.append((x, y, 30.0))
    clusters.append(c3)
    
    return clusters

def rigid_objective(params, cluster, deadline):
    """
    Objective function for the rigid body approach.
    params: [tx, ty, rot, ox, oy, outer_rot]
    tx, ty: translation of cluster center
    rot: rotation of cluster
    ox, oy: center of outer hexagon
    outer_rot: rotation of outer hexagon
    """
    if time.time() > deadline:
        return float('inf')
        
    tx, ty, rot, ox, oy, outer_rot = params
    rot_deg = np.degrees(rot) % 360
    outer_rot_deg = np.degrees(outer_rot) % 360
    
    # Construct absolute positions of inner hexagons
    inner_hexagons = []
    c, s = np.cos(rot), np.sin(rot)
    for lx, ly, l_ang in cluster:
        # Rotate local position
        nx = c * lx - s * ly + tx
        ny = s * lx + c * ly + ty
        # Rotate local angle
        n_ang = (l_ang + rot_deg) % 360
        inner_hexagons.append((nx, ny, n_ang))
        
    S = compute_outer_side_length(inner_hexagons, (ox, oy), outer_rot_deg)
    
    # Check for overlaps to add penalty if any (should be rare for rigid clusters)
    penalty = 0.0
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
                if overlap:
                    penalty += (2.0 - d) ** 2 * 1000
                    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    clusters = generate_rigid_clusters()
    
    best_result = None
    best_score = float('inf')
    
    # Define bounds for Differential Evolution
    # tx, ty: -5 to 5
    # rot: 0 to 2*pi
    # ox, oy: -5 to 5
    # outer_rot: 0 to 2*pi
    bounds = [(-5, 5), (-5, 5), (0, 2*np.pi), (-5, 5), (-5, 5), (0, 2*np.pi)]
    
    for cluster in clusters:
        if time.time() > deadline - 30:
            break
            
        # 1. Global search with Differential Evolution
        try:
            res_de = differential_evolution(rigid_objective, bounds, args=(cluster, deadline),
                                            maxiter=50, popsize=15, tol=1e-6, 
                                            seed=42, polish=True)
            
            if res_de.fun < best_score:
                params = res_de.x
                tx, ty, rot, ox, oy, outer_rot = params
                rot_deg = np.degrees(rot) % 360
                outer_rot_deg = np.degrees(outer_rot) % 360
                
                c, s = np.cos(rot), np.sin(rot)
                inner_hexagons = []
                for lx, ly, l_ang in cluster:
                    nx = c * lx - s * ly + tx
                    ny = s * lx + c * ly + ty
                    n_ang = (l_ang + rot_deg) % 360
                    inner_hexagons.append([nx, ny, n_ang])
                    
                S = compute_outer_side_length(inner_hexagons, (ox, oy), outer_rot_deg)
                
                if check_validity(inner_hexagons, (ox, oy), S, outer_rot_deg):
                    best_score = S
                    best_result = (inner_hexagons, (ox, oy), S, outer_rot_deg)
        except Exception:
            pass
            
        # 2. Local refinement with L-BFGS-B starting from DE result
        if time.time() > deadline - 20:
            break
            
        try:
            # If DE failed, use a reasonable initial guess
            if res_de.fun == float('inf'):
                ic = np.array([0, 0, 0, 0, 0, 0])
            else:
                ic = res_de.x
                
            res_lbfgs = minimize(rigid_objective, ic, args=(cluster, deadline), method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res_lbfgs.fun < best_score:
                params = res_lbfgs.x
                tx, ty, rot, ox, oy, outer_rot = params
                rot_deg = np.degrees(rot) % 360
                outer_rot_deg = np.degrees(outer_rot) % 360
                
                c, s = np.cos(rot), np.sin(rot)
                inner_hexagons = []
                for lx, ly, l_ang in cluster:
                    nx = c * lx - s * ly + tx
                    ny = s * lx + c * ly + ty
                    n_ang = (l_ang + rot_deg) % 360
                    inner_hexagons.append([nx, ny, n_ang])
                    
                S = compute_outer_side_length(inner_hexagons, (ox, oy), outer_rot_deg)
                
                if check_validity(inner_hexagons, (ox, oy), S, outer_rot_deg):
                    best_score = S
                    best_result = (inner_hexagons, (ox, oy), S, outer_rot_deg)
        except Exception:
            pass

    # Fallback if all rigid body attempts failed (should not happen)
    if best_result is None:
        # Simple lattice fallback
        hex_positions = [(0.0, 0.0)]
        for k in range(6):
            angle = np.radians(60 * k)
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            hex_positions.append((x, y))
        for k in range(4):
            angle = np.radians(60 * k)
            x = 2 * np.sqrt(3) * np.cos(angle)
            y = 2 * np.sqrt(3) * np.sin(angle)
            hex_positions.append((x, y))
            
        inner = [[px, py, 0.0] for px, py in hex_positions[:N]]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)
        
    return best_result

# EVOLVE-BLOCK-END
