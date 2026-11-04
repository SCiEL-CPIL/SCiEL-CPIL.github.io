import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # Safety margin below 150s kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with rotation rot_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    Uses Separating Axis Theorem for convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # negative = overlapping

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    # Apothem = (sqrt(3)/2) * side
    # max_proj is the max distance from center to a side (apothem)
    # So side = 2 * apothem / sqrt(3)
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def get_L_from_centers(centers, rots, outer_rot):
        """Compute outer side length from centers, rotations, and outer rotation."""
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in zip(centers[:,0], centers[:,1], rots)])
        return outer_hex_side(all_points, outer_rot)
    
    def is_valid_config(centers, rots):
        """Check if configuration has no overlaps."""
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in zip(centers[:,0], centers[:,1], rots)]
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < -1e-6:
                    return False
        return True
    
    # Generate a honeycomb cluster of 12 hexagons
    # This is a rigid body: 1 center + 6 in first ring + 5 in second ring
    sqrt3 = np.sqrt(3)
    
    def make_honeycomb_12():
        """Create 12 hexagons in a honeycomb pattern: 1 center + 6 ring + 5 outer."""
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        
        # First ring: 6 hexagons at distance sqrt(3)
        for k in range(6):
            ang = np.radians(60 * k)
            x = sqrt3 * np.cos(ang)
            y = sqrt3 * np.sin(ang)
            pts.append([x, y, 0.0])
        
        # Second ring: 5 hexagons (positions 1,3,5,7,9 in the 60-degree grid, skipping to make 5)
        # Actually for 12 total, we need 1 + 6 + 5. The standard honeycomb has 1+6+12=19.
        # For exactly 12, let's use a different arrangement: a 3x4 rectangular grid of hexagons.
        # Hex grid: rows offset by sqrt(3)/2, spacing sqrt(3) horizontally, 1.5 vertically.
        # 3 rows x 4 cols = 12 hexagons
        pts = []
        for r in range(3):
            y = r * 1.5
            for c in range(4):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2
                pts.append([x, y, 0.0])
        
        pts = np.array(pts)
        # Center it
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_honeycomb_3x4():
        """3 rows x 4 cols hex grid."""
        pts = []
        for r in range(3):
            y = r * 1.5
            for c in range(4):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_honeycomb_4x3():
        """4 rows x 3 cols hex grid."""
        pts = []
        for r in range(4):
            y = r * 1.5
            for c in range(3):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring_12():
        """12 hexagons arranged in a ring around center."""
        pts = []
        R = 2.5  # radius for center of ring hexagons
        for i in range(12):
            theta = 2 * np.pi * i / 12
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    # Generate base cluster configurations
    base_configs = []
    base_configs.append(make_honeycomb_3x4())
    base_configs.append(make_honeycomb_4x3())
    base_configs.append(make_ring_12())
    
    # Add rotated variants of each
    for base in base_configs:
        rotated = base.copy()
        rotated[:, 2] = 30.0  # Rotate all hexagons by 30 degrees
        base_configs.append(rotated)
    
    best_L = float('inf')
    best_inner = None
    best_outer_rot = 0.0
    
    # Phase 1: For each base config, optimize global rotation and outer hex rotation
    for cfg_idx, base_cfg in enumerate(base_configs):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        base_centers = base_cfg[:, :2].copy()
        base_rots = base_cfg[:, 2].copy()
        
        # Try a grid of global rotation angles and outer rotation angles
        # Global rotation rotates the entire cluster
        # Outer rotation rotates the enclosing hexagon
        
        best_local_L = float('inf')
        best_cluster_rot = 0.0
        best_outer_rot_local = 0.0
        
        # Grid search over cluster rotation (0 to 60 degrees, step 5) and outer rotation (0 to 60, step 5)
        for cluster_rot_deg in np.arange(0, 60, 5.0):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            for outer_rot_deg in np.arange(0, 60, 5.0):
                # Rotate all centers around origin by cluster_rot_deg
                angle_rad = np.radians(cluster_rot_deg)
                c, s = np.cos(angle_rad), np.sin(angle_rad)
                R = np.array([[c, -s], [s, c]])
                
                new_centers = (R @ base_centers.T).T
                new_rots = base_rots + cluster_rot_deg
                
                # Check validity (rotations preserve relative positions, so if base was valid, this is too)
                # But just to be safe, we can skip this check for rigid bodies
                # Compute outer side
                L = get_L_from_centers(new_centers, new_rots, outer_rot_deg)
                
                if L < best_local_L:
                    best_local_L = L
                    best_cluster_rot = cluster_rot_deg
                    best_outer_rot_local = outer_rot_deg
        
        if best_local_L < best_L:
            best_L = best_local_L
            best_outer_rot = best_outer_rot_local
            
            # Store the best configuration
            angle_rad = np.radians(best_cluster_rot)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[c, -s], [s, c]])
            new_centers = (R @ base_centers.T).T
            new_rots = base_rots + best_cluster_rot
            
            best_inner = np.column_stack((new_centers, new_rots))
    
    # Phase 2: Fine-tune with Nelder-Mead for the best configuration
    if best_inner is not None and time.time() - start_time < DEADLINE - 15:
        # Refine the global rotation and outer rotation
        base_centers = best_inner[:, :2].copy()
        base_rots = best_inner[:, 2].copy()
        
        # Extract current best rotations
        # Find the cluster rotation that was used
        # We need to recover it, so let's just optimize from scratch with Nelder-Mead
        # The state is: cluster_rot (global rotation of the rigid body) and outer_rot
        
        # But we need to know which base config was used. Let's track it.
        # Actually, let's just do a local optimization on the full parameter set.
        # Parameters: 12*(x,y,rot) + outer_rot = 37 parameters
        # That's a lot. Let's instead optimize just the global rotation and outer rotation
        # for the best rigid body configuration.
        
        # We need to know which base config gave the best result.
        # Let's recompute by trying each base config with Nelder-Mead.
        
        for cfg_idx, base_cfg in enumerate(base_configs[:3]):  # Only original 3, not rotated variants
            if time.time() - start_time > DEADLINE - 15:
                break
            
            base_centers = base_cfg[:, :2].copy()
            base_rots = base_cfg[:, 2].copy()
            
            def rigid_objective(params):
                cluster_rot = params[0]
                outer_rot = params[1]
                
                angle_rad = np.radians(cluster_rot)
                c, s = np.cos(angle_rad), np.sin(angle_rad)
                R = np.array([[c, -s], [s, c]])
                
                new_centers = (R @ base_centers.T).T
                new_rots = base_rots + cluster_rot
                
                return get_L_from_centers(new_centers, new_rots, outer_rot)
            
            # Initial guess from grid search
            x0 = np.array([best_cluster_rot if cfg_idx == 0 else 0.0, best_outer_rot])
            
            try:
                res = minimize(rigid_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                
                if res.fun < best_L:
                    best_L = res.fun
                    cluster_rot_opt = res.x[0]
                    outer_rot_opt = res.x[1]
                    
                    angle_rad = np.radians(cluster_rot_opt)
                    c, s = np.cos(angle_rad), np.sin(angle_rad)
                    R = np.array([[c, -s], [s, c]])
                    new_centers = (R @ base_centers.T).T
                    new_rots = base_rots + cluster_rot_opt
                    
                    best_inner = np.column_stack((new_centers, new_rots))
                    best_outer_rot = outer_rot_opt
            except Exception:
                pass
    
    # Phase 3: Try individual hexagon optimization for the best rigid body result
    if best_inner is not None and time.time() - start_time < DEADLINE - 10:
        # Now allow individual hexagons to move slightly to break the rigid constraint
        # This is a local refinement
        centers = best_inner[:, :2].copy()
        rots = best_inner[:, 2].copy()
        
        def full_objective(params):
            if time.time() - start_time > DEADLINE - 5:
                return 1e6
            
            c = params[:2*N].reshape(N, 2)
            r = params[2*N:3*N]
            outer_rot = params[3*N]
            
            # Compute overlap penalty
            vertices_list = [get_hex_vertices(x, y, 1.0, rot) for x, y, rot in zip(c[:,0], c[:,1], r)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i] - c[j])
                    if d < 2.0:
                        d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                        if d_sat < 0:
                            penalty += (-d_sat) ** 2
            
            all_points = np.vstack(vertices_list)
            L = outer_hex_side(all_points, outer_rot)
            
            return L + 50.0 * penalty
        
        x0 = np.concatenate((centers.flatten(), rots.flatten(), [best_outer_rot]))
        
        try:
            res = minimize(full_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            if res.fun < best_L:
                # Verify validity
                c = res.x[:2*N].reshape(N, 2)
                r = res.x[2*N:3*N]
                outer_rot = res.x[3*N]
                
                vertices_list = [get_hex_vertices(x, y, 1.0, rot) for x, y, rot in zip(c[:,0], c[:,1], r)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = sat_overlap(vertices_list[i], vertices_list[j])
                        if d < -1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    L = get_L_from_centers(c, r, outer_rot)
                    if L < best_L:
                        best_L = L
                        best_inner = np.column_stack((c, r))
                        best_outer_rot = outer_rot
        except Exception:
            pass
    
    # Fallback
    if best_inner is None:
        # Simple linear arrangement
        inner_list = []
        for i in range(N):
            x = i * 2.0
            y = 0.0
            inner_list.append([x, y, 0.0])
        return inner_list, (0.0, 0.0), 40.0, 0.0
    
    # Format output
    inner_list = []
    for i in range(N):
        x, y, rot = best_inner[i]
        inner_list.append([float(x), float(y), float(rot)])
    
    return inner_list, (0.0, 0.0), float(best_L), float(best_outer_rot)

# EVOLVE-BLOCK-END
