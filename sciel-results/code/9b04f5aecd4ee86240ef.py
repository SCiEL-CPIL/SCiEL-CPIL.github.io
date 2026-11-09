"""
Strategy: Discrete Symmetry-Aware Search + Local Refinement
Instead of blindly optimizing 36 continuous variables, we recognize that 
packing regular hexagons into a regular hexagon strongly favors lattice-aligned 
configurations due to the 60-degree rotational symmetry and the fact that 
hexagons tile the plane perfectly.

We generate two specific candidate topologies:
1. Honeycomb Ring: 12 hexagons arranged in a ring around the center (radius 2 in hex grid coords).
2. Honeycomb Core: 12 hexagons arranged in a compact cluster (radius 2 in hex grid coords, 
   which actually contains 19 hexagons, so we take the 12 outermost or a specific subset? 
   Actually, a hexagon of side 2 has 19 cells. A hexagon of side 1 has 1 cell. 
   The "Ring" of 12 is the set of hexes at distance 2 from center in a hex grid? 
   No, distance 1 has 6 hexes. Distance 2 has 12 hexes. 
   So "Ring at distance 2" is a valid 12-hex configuration.
   "Core" would be the 7 hexes at dist 0,1 plus 5 more? Or just the 12 at dist 1 and 2?
   
   Let's analyze the "Ring at distance 2" (12 hexes).
   In a hex grid with axial coordinates (q, r):
   Distance from (0,0) is (|q| + |r| + |q+r|) / 2.
   Hexes at distance 2:
   (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,-2), (0,-2) -- wait, let's list them properly.
   
   Axial coords (q, r) for distance d=2:
   (2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1), (2, -1), (1, -2), (0, -2) is wrong.
   
   Let's use cube coords (x, y, z) with x+y+z=0.
   Distance d = (|x| + |y| + |z|) / 2.
   For d=2, we need |x|+|y|+|z| = 4.
   Possible sets (permutations of (2, 2, -4) is not valid since sum must be 0. (2, -2, 0) -> sum 0. |2|+|-2|+|0|=4. Yes.)
   So coordinates are permutations of (2, -2, 0) and (2, 0, -2) etc?
   Actually, the 12 hexes at distance 2 are:
   (2, -2, 0), (2, 0, -2), (0, 2, -2), (-2, 2, 0), (-2, 0, 2), (0, -2, 2) -- these are 6.
   And (2, -1, -1), (1, 2, -1), (-1, 2, -1), (-2, 1, 1), (-1, -2, 1), (1, -1, 2) -- these are another 6.
   Total 12.
   
   We convert these axial/cube coords to Cartesian.
   Then we check if they fit in an outer hexagon.
   Due to symmetry, the optimal outer hexagon will likely be aligned with the inner ones (angle 0) or rotated by 30 degrees.
   
   We will generate these 12 positions, convert to Cartesian, and then run a quick local optimization 
   to "tighten" the packing if possible (though for hexagons, the lattice packing is usually optimal or near-optimal 
   because you can't squeeze them closer without overlap due to the hexagonal symmetry).
   
   Actually, for unit hexagons, the minimal enclosing hexagon side length L can be calculated exactly 
   if the inner hexagons are on the lattice.
   
   Let's calculate L for the "Ring at distance 2".
   The vertices of the outermost inner hexagons will determine L.
   The furthest point from the center in the ring configuration will be at a specific distance.
   
   We will implement the generation of this configuration and return it. 
   We will also try a second configuration: "Two rings" or "Compact cluster".
   But 12 hexes at distance 2 is a very standard packing.
   
   To be safe, we will also run a short Nelder-Mead on the Cartesian coordinates of this ring 
   to ensure it's as tight as possible, allowing for slight rotations of individual hexes if beneficial 
   (though symmetry suggests they should all be aligned).
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon as a numpy array."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
        """Check if a point is inside a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        x_rot = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
        y_rot = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
        
        apothem = side * sqrt3 / 2.0
        
        for i in range(6):
            theta = math.radians(30.0 + i * 60.0)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = x_rot * nx + y_rot * ny
            if proj > apothem + 1e-9:
                return False
        return True

    def check_overlap(centers, angles, eps=1e-6):
        """Check if any two hexagons overlap."""
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Quick check: if distance > 2.0 (diameter), no overlap
                # Actually, for unit hexagons, if centers are distance 2 apart, they touch.
                # If dist > 2.0 + epsilon, no overlap.
                if dist > 2.0 + 1e-5:
                    continue
                
                verts_i = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
                verts_j = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                
                for vx, vy in verts_i:
                    if point_in_hex(vx, vy, centers[j][0], centers[j][1], angles[j]):
                        return True
                for vx, vy in verts_j:
                    if point_in_hex(vx, vy, centers[i][0], centers[i][1], angles[i]):
                        return True
        return False

    def required_L(centers, angles, outer_angle=0.0):
        """Calculate minimum side length of outer hexagon to enclose all inner hexagons."""
        outer_angle_rad = math.radians(outer_angle)
        max_proj = 0.0
        
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for vx, vy in verts:
                x_rot = vx * math.cos(-outer_angle_rad) - vy * math.sin(-outer_angle_rad)
                y_rot = vx * math.sin(-outer_angle_rad) + vy * math.cos(-outer_angle_rad)
                
                for k in range(6):
                    theta = math.radians(30.0 + k * 60.0)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = x_rot * nx + y_rot * ny
                    if proj > max_proj:
                        max_proj = proj
        
        return 2.0 * max_proj / sqrt3

    def objective(params):
        """
        Objective function for local refinement.
        params: 2*N values (x, y for each hex). Angles fixed at 0 for symmetry.
        """
        centers = [(params[2*i], params[2*i+1]) for i in range(N)]
        angles = [0.0] * N
        
        L = required_L(centers, angles, 0.0)
        
        if check_overlap(centers, angles):
            return L + 1000.0
        
        return L

    # --- Generate Candidate 1: Honeycomb Ring (12 hexes at distance 2) ---
    # Cube coordinates for distance 2
    # Permutations of (2, -2, 0) and (2, -1, -1)
    coords_ring = [
        (2, -2, 0), (2, 0, -2), (0, 2, -2), (-2, 2, 0), (-2, 0, 2), (0, -2, 2),
        (2, -1, -1), (1, 2, -1), (-1, 2, -1), (-2, 1, 1), (-1, -2, 1), (1, -1, 2)
    ]
    
    # Convert cube to axial (q, r) where q=x, r=z
    # Cartesian conversion for hex grid (pointy top or flat top?)
    # Standard hex grid with flat top (vertices at 0, 60, ...):
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Wait, let's verify.
    # If q=1, r=0: x=1.5, y=sqrt(3)/2. Distance = sqrt(2.25 + 0.75) = sqrt(3) ~ 1.732.
    # Distance between centers in hex grid is sqrt(3) for adjacent?
    # No, adjacent hexes in a tiling have center distance sqrt(3) if side is 1?
    # Let's check: Two unit hexes sharing an edge.
    # Apothem = sqrt(3)/2. Distance between centers = 2 * apothem = sqrt(3).
    # So yes, center distance is sqrt(3).
    
    centers_ring = []
    for (x, y, z) in coords_ring:
        q = x
        r = z
        cx = 1.5 * q
        cy = sqrt3 * (r + q/2.0)
        centers_ring.append((cx, cy))
    
    angles_ring = [0.0] * N
    
    # Refine Ring
    best_L = required_L(centers_ring, angles_ring, 0.0)
    best_params = np.array([coord for c in centers_ring for coord in c])
    
    if time.time() < deadline - 20.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_L:
                best_L = res.fun
                best_params = res.x
        except Exception:
            pass

    # --- Generate Candidate 2: Compact Cluster ---
    # Take the 12 hexes closest to the center from the distance 0, 1, 2 set.
    # Dist 0: 1 hex (0,0,0)
    # Dist 1: 6 hexes
    # Dist 2: 12 hexes
    # We need 12. So take Dist 0 (1) + Dist 1 (6) + 5 from Dist 2?
    # This breaks symmetry. 
    # Alternative: Take 6 from Dist 1 and 6 from Dist 2?
    # Or just use the "Ring" which is symmetric and likely optimal for this count.
    # Let's try a "Double Ring" subset? No, 12 is exactly the size of the second ring.
    
    # Let's try a configuration where we pack them in a 3x4 grid? 
    # No, hex grid is better.
    
    # Let's verify the Ring configuration's validity.
    final_centers = [(best_params[2*i], best_params[2*i+1]) for i in range(N)]
    final_angles = [0.0] * N
    
    # Check if the refined solution is valid
    if check_overlap(final_centers, final_angles):
        # Fallback to exact ring if refinement caused overlap
        final_centers = centers_ring
        final_angles = angles_ring
        best_L = required_L(final_centers, final_angles, 0.0)
        
    # Calculate final L
    final_L = required_L(final_centers, final_angles, 0.0)
    
    # Construct return value
    inner = []
    for i in range(N):
        inner.append([final_centers[i][0], final_centers[i][1], final_angles[i]])
        
    return inner, (0.0, 0.0), final_L, 0.0

# EVOLVE-BLOCK-END
