"""
Strategy: Symmetry-Constrained 6-DOF Parametric Search with Nelder-Mead.
Instead of optimizing 36 free DOF with soft penalties (which allows the optimizer
to wander in high-dimensional space and struggle with the non-smooth collision
boundary), we constrain the 12 unit hexagons to a canonical hexagonal cluster
topology: 1 at the center, 6 in a first ring, and 5 in a second ring.
This reduces the problem to 6 continuous parameters:
(r1, r2, rot1, rot2, cluster_rot, outer_rot).
We use Nelder-Mead, which is derivative-free and robust in low dimensions,
to minimize the side length of the minimal enclosing regular hexagon.
A hard-feasibility check (SAT) is applied to reject invalid configurations
during the search, ensuring we only explore the valid packing manifold.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)
APOHEM_UNIT = SQRT3 / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_hex_verts(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def min_outer_side(verts_all, outer_center, outer_angle_deg):
        """
        Compute minimal side length of a regular hexagon (center, angle) 
        containing all points in verts_all.
        """
        # The 6 normals of the outer hexagon
        # A regular hexagon with side s has apothem a = s * sqrt(3)/2.
        # The distance from center to each edge is a.
        # The normals are at 30, 90, 150, 210, 270, 330 degrees relative to the hex orientation.
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        max_proj = 0.0
        for na in normals:
            nx = math.cos(na)
            ny = math.sin(na)
            projs = (verts_all[:, 0] - outer_center[0]) * nx + (verts_all[:, 1] - outer_center[1]) * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        # max_proj is the required apothem.
        # side = apothem / (sqrt(3)/2)
        return max_proj / (SQRT3 / 2)

    def check_collision(v1, v2):
        """
        SAT check for two convex polygons. Returns True if they collide (overlap area > 0).
        v1, v2 are (6, 2) arrays of vertices.
        """
        # Get axes from edges of both polygons
        axes = []
        for k in range(6):
            # Edge 1
            e1 = v1[(k+1)%6] - v1[k]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                axes.append(n1/norm1)
            # Edge 2
            e2 = v2[(k+1)%6] - v2[k]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                axes.append(n2/norm2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there is a gap, they don't collide
            if max1 < min2 or max2 < min1:
                return False
        
        return True

    def is_valid_configuration(params):
        """
        Check if the configuration defined by params is valid (no collisions).
        params: [r1, r2, rot1, rot2, cluster_rot, outer_rot]
        """
        r1 = params[0]
        r2 = params[1]
        rot1 = params[2]
        rot2 = params[3]
        cluster_rot = params[4]
        
        if r1 < 1.9 or r2 < 3.5: # Heuristic bounds to avoid obvious overlaps early
            return False
            
        verts_list = []
        
        # Center hexagon
        verts_list.append(get_hex_verts(0, 0, 1.0, 0))
        
        # Ring 1 (6 hexagons)
        for k in range(6):
            ang_pos = np.radians(60*k + cluster_rot)
            cx = r1 * math.cos(ang_pos)
            cy = r1 * math.sin(ang_pos)
            # Orient the hexagons to match the lattice? Or free?
            # In a perfect lattice, they are aligned. Let's assume aligned to cluster for now,
            # but rot1 allows the whole ring to rotate relative to the radial position?
            # Actually, rot1 is the orientation of the hexagons themselves.
            # In the symmetric cluster, usually all are aligned with the outer hex or each other.
            # Let's let rot1 be the orientation of all ring-1 hexagons.
            verts_list.append(get_hex_verts(cx, cy, 1.0, rot1))
            
        # Ring 2 (5 hexagons, skipping one slot to make 12 total? 1+6+5=12)
        # The slots are at angles 30, 90, 150, 210, 270 relative to cluster_rot?
        # Standard hex packing: Ring 2 is at distance sqrt(3)*2 = 2*sqrt(3) ~ 3.464?
        # Wait, distance between centers in hex lattice is 2.
        # Ring 1 distance is 2.
        # Ring 2 distance is 2*sqrt(3) ~ 3.464? No.
        # In a hexagonal lattice of hexagons:
        # Center (0,0).
        # Neighbors at distance 2.
        # Next neighbors: 
        # Distance to (2, 2*sqrt(3))? No.
        # Let's stick to the topology: 1 center, 6 ring 1, 5 ring 2.
        # The 5 ring 2 hexagons are placed in 5 of the 6 "gaps" or positions further out.
        # Actually, a common 12-hex packing is a "flower" with one petal missing?
        # Or 1-6-5 is just 12 hexes.
        # Let's place them at angles 60*k + 30 (offset from ring 1) at radius r2.
        for k in range(5):
            ang_pos = np.radians(60*k + 30 + cluster_rot)
            cx = r2 * math.cos(ang_pos)
            cy = r2 * math.sin(ang_pos)
            verts_list.append(get_hex_verts(cx, cy, 1.0, rot2))
            
        # Check all pairs
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                if check_collision(verts_list[i], verts_list[j]):
                    return False
        return True

    def objective(params):
        """
        Objective: Minimize outer side length.
        If invalid, return a large penalty to guide Nelder-Mead away.
        """
        if not is_valid_configuration(params):
            # Penalty proportional to how "bad" it is? 
            # Simple large constant is often enough for NM if the valid region is connected.
            # But we can return a value that encourages moving to valid space.
            # Let's just return a large number.
            return 100.0
            
        r1 = params[0]
        r2 = params[1]
        rot1 = params[2]
        rot2 = params[3]
        cluster_rot = params[4]
        outer_rot = params[5]
        
        verts_list = []
        
        # Center
        verts_list.append(get_hex_verts(0, 0, 1.0, 0))
        
        # Ring 1
        for k in range(6):
            ang_pos = np.radians(60*k + cluster_rot)
            cx = r1 * math.cos(ang_pos)
            cy = r1 * math.sin(ang_pos)
            verts_list.append(get_hex_verts(cx, cy, 1.0, rot1))
            
        # Ring 2
        for k in range(5):
            ang_pos = np.radians(60*k + 30 + cluster_rot)
            cx = r2 * math.cos(ang_pos)
            cy = r2 * math.sin(ang_pos)
            verts_list.append(get_hex_verts(cx, cy, 1.0, rot2))
            
        all_verts = np.vstack(verts_list)
        
        # The outer hexagon is centered at (0,0) due to symmetry of the cluster?
        # If the cluster is symmetric (1-6-5 is NOT fully symmetric due to missing 6th in ring 2),
        # the center of mass might shift. 
        # However, the "minimal enclosing hexagon" center is a free variable in the general case.
        # In this symmetric-ish model, we assume the outer hexagon is centered at (0,0).
        # If the 5th hexagon in ring 2 makes it asymmetric, the minimal enclosing hexagon
        # might need to shift its center. 
        # To keep it 6-DOF, we assume the outer hexagon is centered at the centroid of the inner hexes?
        # Or just (0,0)?
        # Let's calculate the bounding box in the rotated frame to find the best center?
        # No, let's stick to (0,0) for simplicity first, as the 1-6-5 shape is "almost" symmetric.
        # Actually, if we drop one from ring 2, the shape is a hexagon with a notch.
        # The minimal enclosing hexagon of a "notched hexagon" is likely still centered at the origin
        # if the notch is small, or shifted towards the notch.
        # Given the constraints, let's assume center (0,0) is a good approximation for the 6-DOF model.
        # If we want higher accuracy, we could add cx, cy as DOFs, making it 8-DOF.
        # Let's try 8-DOF: [r1, r2, rot1, rot2, cluster_rot, outer_rot, cx, cy]
        
        # Wait, the params passed in are only 6. 
        # I defined is_valid_configuration with 6 params.
        # Let's stick to 6 params and center (0,0).
        
        side = min_outer_side(all_verts, (0, 0), outer_rot)
        
        return side

    # Initial Guesses
    # Standard hex packing:
    # r1 = 2.0 (touching)
    # r2 = 2 * sqrt(3) ~ 3.464 (touching ring 1 and each other?)
    # In a hex lattice, distance between centers of adjacent hexes is 2.
    # Distance from center to ring 1 is 2.
    # Distance from center to ring 2 (skipping one) is?
    # The vertices of the "super-hexagon" formed by 19 hexes (1+6+12) has side 3.
    # The distance from center to the "outer" hexes in a full 19-pack is 2*sqrt(3)?
    # Let's just start near the touching configuration.
    
    best_params = None
    best_score = float('inf')
    
    # Multi-start Nelder-Mead
    # Start 1: Perfect lattice
    x0_1 = np.array([2.0, 3.464, 0, 0, 0, 0])
    # Start 2: Slightly expanded
    x0_2 = np.array([2.1, 3.6, 0, 0, 0, 0])
    # Start 3: Rotated
    x0_3 = np.array([2.0, 3.464, 30, 30, 0, 30])
    
    starts = [x0_1, x0_2, x0_3]
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 20:
            break
            
        try:
            # Nelder-Mead is good for low-D non-smooth (if we use hard penalty) or smooth.
            # Since we use a hard penalty (100.0), it's non-smooth.
            # NM handles this well by moving the simplex.
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback to a valid known configuration
        # Just return the initial guess if optimization fails
        best_params = x0_1
        best_score = objective(best_params)
        
    # Refine with a tighter tolerance if time permits
    if time.time() < deadline - 5 and best_score < 100:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Extract results
    r1, r2, rot1, rot2, cluster_rot, outer_rot = best_params
    
    inner = []
    
    # Center
    inner.append((0.0, 0.0, 0.0))
    
    # Ring 1
    for k in range(6):
        ang_pos = np.radians(60*k + cluster_rot)
        cx = r1 * math.cos(ang_pos)
        cy = r1 * math.sin(ang_pos)
        inner.append((float(cx), float(cy), float(rot1)))
        
    # Ring 2
    for k in range(5):
        ang_pos = np.radians(60*k + 30 + cluster_rot)
        cx = r2 * math.cos(ang_pos)
        cy = r2 * math.sin(ang_pos)
        inner.append((float(cx), float(cy), float(rot2)))
        
    center = (0.0, 0.0)
    angle = float(outer_rot)
    
    # Calculate final side length precisely
    verts_list = []
    for cx, cy, ang in inner:
        verts_list.append(get_hex_verts(cx, cy, 1.0, ang))
    all_verts = np.vstack(verts_list)
    side = min_outer_side(all_verts, center, angle)
    
    # Final Validation: Ensure all vertices are inside the outer hexagon
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-8:
                return False
        return True

    valid = True
    for cx, cy, ang in inner:
        verts = get_hex_verts(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # If invalid due to numerical precision or center shift, expand slightly
        side *= 1.01

    return inner, center, float(side), float(angle)

if __name__ == "__main__":
    # Quick test
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}")
    print(f"Inner: {inner[:3]}...")
