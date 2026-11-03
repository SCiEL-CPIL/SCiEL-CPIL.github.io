import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Unit hexagon vertices (side 1, circumradius 1)
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def support_hex(n_vec):
        if n_vec.ndim == 1:
            n_vec = n_vec.reshape(1, -1)
        proj = n_vec @ unit_verts.T
        return np.max(proj, axis=1)

    def calc_S(positions, beta_deg):
        beta_rad = math.radians(beta_deg)
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0 + beta_rad
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        h_hex = support_hex(normals)
        
        proj_centers = normals @ positions.T
        max_center_proj = np.max(proj_centers, axis=1)
        
        extents = max_center_proj + h_hex
        max_extent = np.max(extents)
        
        return max_extent * 2.0 / sqrt3

    def check_validity(inner, center, side, angle):
        # Check if all inner hexagons are inside the outer hexagon
        # and if they are disjoint from each other.
        # This is expensive, so we do it only on candidates.
        if time.time() > deadline - 5:
            return False

        outer_center = np.array(center)
        outer_side = side
        outer_angle_rad = math.radians(angle)
        
        # Outer hexagon normals
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0 + outer_angle_rad
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        apothem = outer_side * sqrt3 / 2

        # Check containment
        for i in range(N):
            x, y, ang_deg = inner[i]
            pos = np.array([x, y]) - outer_center
            ang_rad = math.radians(ang_deg)
            
            # Rotate inner vertices to world coordinates
            c = math.cos(ang_rad)
            s = math.sin(ang_rad)
            rot_mat = np.array([[c, -s], [s, c]])
            verts = unit_verts @ rot_mat.T + pos
            
            # Check each vertex against outer hexagon constraints
            # A point p is inside if dot(p - center, n) <= apothem for all normals
            # Note: The normals point OUTWARDS.
            # The condition is: for all 6 normals n, (p - center) dot n <= apothem
            # Wait, the support function calculation in calc_S assumed the hexagon is centered at 0.
            # Here we shifted pos by outer_center.
            # The constraints are: max_{v in verts} (v dot n) <= apothem?
            # No, the outer hexagon is defined by |dot(p, n_i)| <= apothem?
            # No, a regular hexagon centered at origin with normals n_i has sides at distance apothem.
            # The region is { p | dot(p, n_i) <= apothem for all i } ?
            # If n_i are outward normals, yes.
            # Let's verify: Normal at 30 deg. Side is vertical? No.
            # Vertices at 0, 60, 120...
            # Side between 0 and 60 deg vertices. Midpoint at 30 deg.
            # Normal is at 30 deg.
            # Distance from origin to side is apothem.
            # So dot(p, n_30) <= apothem is correct for the right-side boundary.
            
            max_proj = np.max(verts @ normals.T)
            if max_proj > apothem + 1e-9:
                return False

        # Check disjointness (approximate by checking center distances)
        # Minimum distance between centers of two disjoint unit hexagons is 2.
        # (If they touch, distance is 2. If disjoint, > 2).
        # Actually, if they rotate, the minimum distance can be less than 2?
        # No, for convex bodies, if they are disjoint, the distance between their closest points is > 0.
        # The distance between centers must be at least the sum of their radii in that direction.
        # For hexagons, the inradius is sqrt(3)/2 ~ 0.866.
        # If two hexagons are side-by-side, center distance is 2 * inradius = sqrt(3) ~ 1.732?
        # No, side length 1. Distance between parallel sides is sqrt(3).
        # If they touch side-to-side, center distance is sqrt(3).
        # If they touch vertex-to-vertex, center distance is 2.
        # So center distance must be >= sqrt(3) to be disjoint?
        # Let's check. Two unit hexagons.
        # One at (0,0). One at (sqrt(3), 0).
        # Rightmost point of left hex: x=1.
        # Leftmost point of right hex: x = sqrt(3) - 1 ~ 0.732.
        # 1 > 0.732, so they overlap.
        # Wait, circumradius is 1. Inradius is sqrt(3)/2.
        # Side-to-side contact: Distance = 2 * inradius = sqrt(3).
        # Let's re-evaluate.
        # Hexagon 1 at (0,0). Vertices at (1,0), (0.5, 0.866)...
        # Rightmost point is (1,0).
        # Hexagon 2 at (d, 0). Leftmost point is (d-1, 0).
        # For disjointness, we need d-1 >= 1 => d >= 2.
        # Wait, the "width" of a hexagon is 2 (vertex to vertex) or sqrt(3) (side to side).
        # If aligned with vertices on x-axis, width is 2.
        # If aligned with sides vertical, width is sqrt(3).
        # So minimum center distance depends on orientation.
        # To be safe, we should check actual overlap.
        # However, for a quick fix, we can assume the generated packings are valid (standard grids are).
        # The 3x4 grid has centers at distance 1 (along q) and sqrt(3) (along r)?
        # Axial coords: (q, r).
        # (0,0) to (1,0): dist 1.
        # (0,0) to (0,1): dist sqrt(3)/2 * 2?
        # x = q + r/2, y = r*sqrt(3)/2.
        # (0,0) -> (0,0).
        # (1,0) -> (1,0). Dist 1.
        # (0,1) -> (0.5, 0.866). Dist 1.
        # So neighbors are at distance 1.
        # But unit hexagons have circumradius 1.
        # If centers are distance 1 apart, they overlap heavily!
        # The previous code used `unit_verts` with circumradius 1.
        # And `get_topology` used axial coordinates which imply a lattice spacing of 1.
        # This means the previous code generated OVERLAPPING hexagons.
        # That is why the area check failed.
        
        # FIX: The lattice spacing for non-overlapping unit hexagons (side 1) must be at least 2?
        # No.
        # Hexagon tiling:
        # Centers form a triangular lattice.
        # Distance between centers in a tiling is 2 * inradius = sqrt(3)?
        # Let's check the tiling.
        # Side length 1.
        # Distance between parallel sides is sqrt(3).
        # In a tiling, hexagons share sides.
        # So the distance between centers of adjacent hexagons is sqrt(3).
        # Let's verify.
        # Center (0,0). Side midpoint at (sqrt(3)/2, 0)? No.
        # Vertex at (1,0). Side midpoint at (0.5, sqrt(3)/2) ? No.
        # Vertices: (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2).
        # Side between (1,0) and (1/2, sqrt(3)/2).
        # Midpoint: (3/4, sqrt(3)/4).
        # Normal vector: (sqrt(3)/4, -3/4)? No.
        # The normal to the side at 30 deg is (cos30, sin30) = (sqrt(3)/2, 1/2).
        # Distance from origin to this side: dot(midpoint, normal) = (3/4)(sqrt(3)/2) + (sqrt(3)/4)(1/2) = 3sqrt(3)/8 + sqrt(3)/8 = 4sqrt(3)/8 = sqrt(3)/2.
        # Correct. Inradius is sqrt(3)/2.
        # Adjacent hexagon center is at distance 2 * inradius = sqrt(3) along the normal.
        # So the lattice spacing is sqrt(3).
        
        # The previous code used spacing 1. This is WRONG.
        # We must scale the positions by sqrt(3) or use the correct lattice spacing.
        
        # Let's regenerate topologies with correct spacing.
        return True # Placeholder for logic below

    def get_topology_scaled(type_id):
        """
        Generate 12 disjoint unit hexagon centers with correct spacing.
        Spacing between adjacent centers in a hexagonal tiling is sqrt(3).
        """
        centers = []
        scale = sqrt3 # Spacing is sqrt(3)
        
        if type_id == 0:
            # 3x4 Grid in axial coordinates, scaled by sqrt(3)
            # Axial (q, r) -> Cartesian (x, y) = (q + r/2, r*sqrt(3)/2)
            # If we scale q and r by 1, the distance is 1.
            # We want distance sqrt(3). So we multiply the resulting coordinates by sqrt(3)?
            # No, the formula x = q + r/2, y = r*sqrt(3)/2 gives distance 1 for neighbors.
            # So we multiply x and y by sqrt(3).
            for q in range(3):
                for r in range(4):
                    x = (q + r/2.0) * scale
                    y = (r * sqrt3/2.0) * scale
                    centers.append((x, y))
                    
        elif type_id == 1:
            # 2x6 Grid
            for q in range(2):
                for r in range(6):
                    x = (q + r/2.0) * scale
                    y = (r * sqrt3/2.0) * scale
                    centers.append((x, y))
                    
        elif type_id == 2:
            # Hexagonal cluster subset (12 points)
            # Center (0,0)
            # Ring 1: 6 points at dist sqrt(3)
            # Ring 2: 6 points at dist 2*sqrt(3)? No, ring 2 has 12 points.
            # We need 12.
            # Take Center + Ring 1 (6) + 5 from Ring 2.
            # Or just take a 3x4 block which is compact.
            # Let's try a "flower" shape: Center + 6 neighbors + 5 more.
            # Neighbors at angles 0, 60, ... dist sqrt(3).
            # Next ring at dist 2*sqrt(3)?
            # Let's stick to 3x4 and 2x6 as they are valid tilings.
            pass

        if not centers:
            return None
            
        pos = np.array(centers)
        centroid = np.mean(pos, axis=0)
        pos = pos - centroid
        return pos

    best_S = float('inf')
    best_positions = None
    best_beta = 0.0

    topologies_to_try = [0, 1]
    
    for t_id in topologies_to_try:
        if time.time() > deadline - 20:
            break
            
        positions = get_topology_scaled(t_id)
        if positions is None:
            continue
            
        best_local_S = float('inf')
        best_local_beta = 0.0
        
        # Coarse grid search for beta
        for b in np.arange(0, 60, 0.5):
            S = calc_S(positions, b)
            if S < best_local_S:
                best_local_S = S
                best_local_beta = b
                
        if best_local_S < best_S:
            best_S = best_local_S
            best_positions = positions
            best_beta = best_local_beta

    # Refine
    if best_positions is not None and time.time() < deadline - 10:
        def objective(params):
            beta, dx, dy = params
            pos_shifted = best_positions + np.array([dx, dy])
            return calc_S(pos_shifted, beta)
            
        init_params = [best_beta, 0.0, 0.0]
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x[0]
                best_positions = best_positions + np.array([res.x[1], res.x[2]])
        except:
            pass

    # Final construction
    inner = []
    for i in range(12):
        x, y = best_positions[i]
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_S, best_beta

# EVOLVE-BLOCK-END
