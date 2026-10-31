import numpy as np
import time

N = 11
HEX_DIST = np.sqrt(3)

def get_hex_vertices(center, angle_deg=0.0, side=1.0):
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.deg2rad(np.arange(0, 360, 60))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def min_bounding_hex_side(points, center, angle_deg=0.0):
    # points: Nx2 array of vertices
    # We want to find the minimum side length S of a regular hexagon
    # centered at 'center' with rotation 'angle_deg' that contains all points.
    # A regular hexagon of side S has inradius r = S * sqrt(3) / 2.
    # The condition for a point p to be inside is:
    # | (p - center) dot n_i | <= r for i in 0, 1, 2 (normals at 0, 60, 120 deg)
    # where n_i are the normals to the sides.
    # Actually, the normals are at 30, 90, 150 deg if the flat side is at 0?
    # Let's define the hexagon such that its sides are perpendicular to vectors at angles:
    # alpha, alpha + 60, alpha + 120.
    # For a "flat-top" hexagon (vertex at 0, 60...), the sides are at 30, 90, 150.
    # For a "pointy-top" hexagon (vertex at 90...), sides are at 0, 60, 120.
    # The problem says "regular hexagon". Orientation is free via 'angle'.
    # We can fix the inner hexagons to angle 0 (flat top? or pointy top?).
    # Let's assume inner hexagons are "pointy top" (vertex at 90 deg) or "flat top" (vertex at 0 deg).
    # Standard hex grid often uses pointy top.
    # Let's use vertex at 0 deg (flat top) for simplicity? 
    # No, let's use the property that the bounding box orientation can be anything.
    # To simplify, we can fix the bounding hexagon orientation to 0 and rotate the cluster?
    # No, we return the bounding hexagon angle.
    # So we can fix the bounding hexagon angle to 0 and optimize the cluster orientation?
    # Or fix cluster orientation and optimize bounding hexagon angle.
    # Let's fix inner hexagons to angle 0.
    # Then we optimize the bounding hexagon center, side, and angle.
    # For a given set of points, the min bounding hexagon can be found by checking 3 axes.
    # But the angle is also a variable.
    # However, for a compact cluster, the optimal bounding hexagon will likely be aligned with the cluster's symmetry.
    # If the cluster is hexagonally symmetric, the bounding hexagon should be aligned.
    # So we can fix the bounding hexagon angle to 0 (or 30) and just optimize center and side.
    # Let's fix bounding hexagon angle to 0.
    # Then we just need to find the min S and center.
    # The min center is the centroid? No, but close.
    # We can just calculate the required S for a given center and take the min over centers?
    # Actually, for a fixed orientation, the min bounding box center is unique.
    # We can calculate the projection range for each of the 3 axes.
    # S is determined by the max range.
    # We can optimize the center to minimize the max range.
    
    # For simplicity in the local search, we can just fix the center to the centroid of the cluster
    # and calculate the required S. Then optimize the cluster positions to minimize S.
    # This is a valid approximation and often optimal for symmetric shapes.
    
    cx, cy = center
    # Normals for a hexagon with "flat" sides at 0, 60, 120?
    # If vertices are at 0, 60..., sides are at 30, 90, 150.
    # Let's use normals at 0, 60, 120. This corresponds to a hexagon with vertices at 30, 90...
    # It doesn't matter as long as we are consistent.
    # Let's use normals at 0, 60, 120.
    normals = []
    for a in [0, 60, 120]:
        rad = np.deg2rad(a)
        normals.append(np.array([np.cos(rad), np.sin(rad)]))
    normals = np.array(normals)
    
    # Translate points to center
    rel_points = points - np.array(center)
    projections = rel_points @ normals.T
    # projections is Nx3
    min_proj = np.min(projections, axis=0)
    max_proj = np.max(projections, axis=0)
    widths = max_proj - min_proj
    # The required inradius r must be >= max(|min_proj|, |max_proj|) for each axis?
    # No, the hexagon is centered at 'center'.
    # So we need |proj| <= r for all points.
    # So r >= max(abs(min_proj), abs(max_proj))
    r_required = np.max(np.abs(np.vstack([min_proj, max_proj])))
    S = 2 * r_required / np.sqrt(3)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # 10 seconds budget
    
    # Initial configuration: Hexagonal cluster
    # Axial to Cartesian (pointy top, side 1)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Distance between neighbors is sqrt(3).
    
    # Generate 11 positions
    # Center (0,0)
    # Ring 1 (6)
    # Ring 2 (4)
    
    # Let's just use a list of axial coords
    coords = [(0,0)]
    # Ring 1
    for i in range(6):
        q = np.cos(np.deg2rad(60*i))
        r = np.sin(np.deg2rad(60*i))
        # This is not axial. Let's use standard axial neighbors.
        # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    coords += [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    # Ring 2 (need 4)
    # Let's add (2,0), (2,-1), (1,-2), (-1,2) ? 
    # Let's try to keep it compact.
    # Add (2,0), (-2,0), (2,-2), (-2,2)? 
    # Let's just add 4 random valid spots from Ring 2 and let optimizer fix it.
    # Or better, a symmetric set: (2,0), (-2,0), (0,2), (0,-2)? No, (0,2) is not a neighbor of Ring 1 in a way that keeps compactness?
    # Axial coords for Ring 2:
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,1), (0,1)... wait.
    # Let's just pick 4 from the 12 available in Ring 2.
    # To be symmetric, pick opposite pairs.
    # (2,0) and (-2,0)
    # (2,-2) and (-2,2)
    coords += [(2,0), (-2,0), (2,-2), (-2,2)]
    
    # Convert to Cartesian
    positions = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        positions.append([x, y])
    positions = np.array(positions)
    
    # Calculate initial S
    center = np.mean(positions, axis=0)
    # Get all vertices
    verts = []
    for p in positions:
        v = get_hex_vertices(p, 0.0, 1.0)
        verts.append(v)
    all_verts = np.vstack(verts)
    
    current_S = min_bounding_hex_side(all_verts, center)
    best_S = current_S
    best_positions = positions.copy()
    best_center = center
    
    # Local Search
    # We will jitter positions and update center to centroid.
    step_size = 0.1
    no_improve = 0
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline: break
            # Jitter
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            new_pos = positions[i].copy()
            new_pos[0] += dx
            new_pos[1] += dy
            
            # Check collision
            valid = True
            for j in range(N):
                if i == j: continue
                dist = np.linalg.norm(new_pos - positions[j])
                if dist < HEX_DIST - 1e-6:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Try the move
            old_pos = positions[i].copy()
            positions[i] = new_pos
            
            # Update center to centroid
            new_center = np.mean(positions, axis=0)
            
            # Recalculate S
            verts = []
            for p in positions:
                v = get_hex_vertices(p, 0.0, 1.0)
                verts.append(v)
            all_verts = np.vstack(verts)
            new_S = min_bounding_hex_side(all_verts, new_center)
            
            if new_S < best_S - 1e-6:
                best_S = new_S
                best_positions = positions.copy()
                best_center = new_center
                improved = True
                no_improve = 0
            else:
                # Revert
                positions[i] = old_pos
                no_improve += 1
        
        if not improved:
            if no_improve > 100:
                step_size *= 0.9
                if step_size < 0.001:
                    break
            no_improve = 0

    # Construct return values
    inner = []
    for p in best_positions:
        inner.append([p[0], p[1], 0.0])
    
    return inner, (best_center[0], best_center[1]), best_S, 0.0

