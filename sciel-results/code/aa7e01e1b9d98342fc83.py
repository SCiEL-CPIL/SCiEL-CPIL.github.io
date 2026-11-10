import numpy as np
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # 1. Define the 7 central hexagons (1 center + 6 neighbors)
    # Lattice basis
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    centers = []
    # Center
    centers.append(np.array([0, 0]))
    # 6 Neighbors
    neighbors = [
        u, v, v-u, -u, -v, u-v
    ]
    for n in neighbors:
        centers.append(n)
        
    # 2. Define 5 additional hexagons in the "gaps" of the next ring
    # The "side" positions of Ring 2 are at u+v, v-u, -u-v, -v+u, u-v, -u+v?
    # Let's identify the 6 side positions of Ring 2.
    # They are the sums of adjacent Ring 1 vectors.
    # u+v, v+(v-u)=2v-u, (v-u)+(-u)=v-2u, -u+(-v)=-u-v, -v+(u-v)=u-2v, (u-v)+u=2u-v
    # Wait, the "side" positions are the ones that are NOT the "corners" (2u, 2v, etc).
    # The 12 positions of Ring 2 are:
    # Corners: 2u, 2v, 2(v-u), -2u, -2v, 2(u-v)
    # Sides: u+v, v+(v-u), (v-u)+(-u), -u+(-v), -v+(u-v), (u-v)+u
    
    side_positions = [
        u + v,
        2*v - u,
        v - 2*u,
        -u - v,
        u - 2*v,
        2*u - v
    ]
    
    # We pick 5 of these 6. We leave the one at angle closest to 0 (or any) empty to minimize max radius?
    # Actually, due to symmetry, leaving any one empty results in the same max radius.
    # We leave the first one empty.
    for i in range(1, 6):
        centers.append(side_positions[i])
        
    centers = np.array(centers)
    
    # 3. Find the optimal rotation and outer hexagon size
    # The outer hexagon is regular, centered at (0,0) (we can shift later).
    # We want to minimize S such that all inner hexagons are inside.
    # An inner hexagon at c with angle 0 is inside the outer hexagon (center 0, side S, angle 0)
    # if for all 6 edges of the outer hexagon, the inner hexagon is on the inside.
    # Equivalent to: The maximum distance from the origin to any point of the inner hexagons,
    # in the direction normal to the outer hexagon edges, must be <= S * sqrt(3)/2.
    
    # The normal directions of the outer hexagon (angle 0) are at angles 30, 90, 150, 210, 270, 330.
    # Let's define a function to check if a configuration with rotation theta is valid for side S.
    # Actually, we can compute the required S for a given theta.
    # For a given theta, the outer hexagon edges are at angles theta + 30 + 60k.
    # The constraint is: For each edge normal n, max(dot(p, n)) over all p in inner hexagons <= S * sqrt(3)/2.
    
    def calc_required_s(theta):
        # Rotate centers
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ R.T
        
        # The inner hexagons are still axis-aligned relative to the global frame?
        # No, the problem says "inner a (12, 3) array of rows (x, y, angle_deg)".
        # We can set the angle of all inner hexagons to 0 in the local frame,
        # but if we rotate the whole assembly, we should rotate the hexagons too?
        # The task allows any angle for each inner hexagon.
        # To maximize packing density, we usually align them with the lattice.
        # If we rotate the assembly, we rotate the centers AND the hexagon orientations.
        # So the "shape" of the assembly is rotated.
        
        # The outer hexagon is fixed at angle 0 (we can set outer angle to 0).
        # So we check the rotated assembly against a fixed outer hexagon.
        
        # For each inner hexagon, we need to check if it fits.
        # Since all inner hexagons are identical and rotated by theta,
        # their support function in direction n is h(n) = dot(center, n) + h_hex(n - theta).
        # h_hex(alpha) for a unit hexagon is:
        # max(dot(v, alpha)) for vertices v.
        # For a unit hexagon (side 1), the support function is:
        # h(alpha) = max(cos(alpha), cos(alpha - 60), ...)? No.
        # The vertices are at angles 0, 60, ...
        # h(alpha) = max_{k} cos(alpha - k*60).
        
        max_proj = 0
        # Normals of outer hexagon (angle 0) are at 30, 90, ...
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        
        for i in range(12):
            c_i = rotated_centers[i]
            # The inner hexagon i is rotated by theta.
            # Its vertices are at angles theta + 60k.
            # Support in direction n: dot(c_i, n) + max_k dot(u_k, n) where u_k are unit vectors at theta+60k.
            # max_k dot(u_k, n) = max_k cos(n_angle - (theta + 60k)).
            
            for n_angle in normal_angles:
                n = np.array([np.cos(n_angle), np.sin(n_angle)])
                proj_center = np.dot(c_i, n)
                
                # Support of the hexagon shape
                # The hexagon has vertices at angles theta, theta+60, ...
                # The support in direction n_angle is max_{k=0..5} cos(n_angle - theta - k*60)
                # Since cos is periodic and symmetric, this is just the max of cos over 6 offsets.
                # This value is independent of theta? No, it depends on the relative angle.
                # Actually, for a regular polygon, the support function is periodic.
                # h(n) = max_{k} cos(n_angle - theta - k*pi/3).
                
                # We can compute this quickly.
                angles = n_angle - theta - np.arange(6) * np.pi/3
                h_shape = np.max(np.cos(angles))
                
                total_proj = proj_center + h_shape
                if total_proj > max_proj:
                    max_proj = total_proj
                    
        # The apothem of the outer hexagon is S * sqrt(3)/2.
        # So max_proj <= S * sqrt(3)/2
        # S >= max_proj / (sqrt(3)/2)
        return max_proj / (np.sqrt(3)/2)

    # Optimize theta
    best_s = float('inf')
    best_theta = 0
    # Sample theta from 0 to 60 degrees
    for theta in np.linspace(0, np.pi/3, 180):
        s = calc_required_s(theta)
        if s < best_s:
            best_s = s
            best_theta = theta
            
    # Final result
    c, s = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    # The inner hexagons are rotated by best_theta
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = np.rad2deg(best_theta)
        
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

if __name__ == "__main__":
    # Quick test
    i, o, s = hexagon_packing_12()
    print(f"Side length: {s}")
