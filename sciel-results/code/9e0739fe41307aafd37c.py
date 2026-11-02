import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions, outer_angle=0.0):
        """
        Compute the exact side length of the smallest regular hexagon (with given angle)
        centered at the origin that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        # Rotate vertices by -outer_angle to align with outer hexagon axes
        theta = -np.radians(outer_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        verts_rot = verts_flat @ R.T
        
        x = verts_rot[:, 0]
        y = verts_rot[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # Support functions for regular hexagon of side s centered at origin
        # The hexagon is defined by |u| <= s * sqrt(3)/2 in 3 directions
        # Let's use the 6 face normals
        # Face 1: (1, 0) -> x <= s * sqrt(3)/2
        # Face 2: (1/2, sqrt(3)/2) -> x/2 + y*sqrt(3)/2 <= s * sqrt(3)/2
        # Face 3: (-1/2, sqrt(3)/2) -> -x/2 + y*sqrt(3)/2 <= s * sqrt(3)/2
        # Face 4: (-1, 0) -> -x <= s * sqrt(3)/2
        # Face 5: (-1/2, -sqrt(3)/2) -> -x/2 - y*sqrt(3)/2 <= s * sqrt(3)/2
        # Face 6: (1/2, -sqrt(3)/2) -> x/2 - y*sqrt(3)/2 <= s * sqrt(3)/2
        
        # Let h = s * sqrt(3)/2 (apothem)
        # We need max of projections on these 6 normals <= h
        # So s >= 2/sqrt(3) * max(proj)
        
        n1 = np.array([1, 0])
        n2 = np.array([0.5, np.sqrt(3)/2])
        n3 = np.array([-0.5, np.sqrt(3)/2])
        n4 = np.array([-1, 0])
        n5 = np.array([-0.5, -np.sqrt(3)/2])
        n6 = np.array([0.5, -np.sqrt(3)/2])
        
        normals = np.array([n1, n2, n3, n4, n5, n6])
        
        projs = verts_rot @ normals.T
        max_proj = np.max(np.abs(projs)) # Since normals come in pairs, max abs is enough
        # Actually, the constraints are:
        # x <= h, -x <= h => |x| <= h
        # 0.5x + sqrt(3)/2 y <= h, -0.5x - sqrt(3)/2 y <= h => |0.5x + sqrt(3)/2 y| <= h
        # etc.
        
        # So h_required = max(|proj|) for the 3 unique directions
        # s_required = 2 * h_required / sqrt(3)
        
        # Let's just compute the max over all 6 to be safe (it's symmetric)
        h_req = np.max(np.abs(projs))
        s_req = 2.0 * h_req / np.sqrt(3.0)
        
        return s_req

    def check_validity(positions, tol=1e-5):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                # SAT check
                disjoint = False
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                
                if not disjoint:
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    def generate_positions(params):
        """
        Generate positions for 1 center + 6 ring1 + 5 ring2 hexagons.
        params: [r1, r2, offset2, scale]
        r1: radius of ring 1 (distance from center to ring1 centers)
        r2: radius of ring 2 (distance from center to ring2 centers)
        offset2: angular offset of ring 2 (degrees)
        scale: global scale factor applied to all radii
        """
        r1 = params[0] * params[3]
        r2 = params[1] * params[3]
        offset2 = params[2]
        
        positions = np.zeros((N, 2))
        # Center
        positions[0] = [0.0, 0.0]
        
        # Ring 1: 6 hexagons at 60 deg intervals
        for i in range(6):
            angle = np.radians(60 * i)
            positions[i+1, 0] = r1 * np.cos(angle)
            positions[i+1, 1] = r1 * np.sin(angle)
            
        # Ring 2: 5 hexagons. 
        # To maintain symmetry and pack well, we place them at 72 deg intervals.
        # The offset allows us to rotate this ring relative to ring 1.
        # A good initial guess is to align one ring-2 hexagon with a ring-1 hexagon direction or between them.
        for i in range(5):
            angle = np.radians(72 * i + offset2)
            positions[i+7, 0] = r2 * np.cos(angle)
            positions[i+7, 1] = r2 * np.sin(angle)
            
        return positions

    def objective(params):
        """
        Objective: Outer side length + Penalty for overlaps.
        """
        positions = generate_positions(params)
        
        # Check validity
        if not check_validity(positions):
            # Calculate overlap penalty
            # Simple penalty: sum of squared negative distances
            # We'll just return a large number if invalid
            # But to help optimization, let's add a gradient-like penalty
            # For now, just return a large value
            # A better penalty: sum of (min_dist)^2 where min_dist < 0
            # Since check_validity is boolean, we can't get the distance easily without re-implementing.
            # Let's use a heuristic penalty based on center distances.
            # If center dist < 2*apothem = sqrt(3) ~ 1.732, they overlap.
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < np.sqrt(3.0):
                        penalty += (np.sqrt(3.0) - d)**2
            return 100.0 + penalty
            
        # Calculate outer side length
        # Try a few outer angles to find the minimal one
        # Due to symmetry, 0 to 30 degrees is enough
        min_side = float('inf')
        for ang in [0, 15, 30]:
            s = required_outer_side(positions, ang)
            if s < min_side:
                min_side = s
                
        return min_side

    # Initial guesses
    # Honeycomb lattice:
    # Center at (0,0)
    # Ring 1 at distance sqrt(3) (touching flat-to-flat)
    # Ring 2 at distance 2*sqrt(3) (touching ring 1 and each other in a larger ring)
    # But we only have 5 in ring 2.
    
    inits = [
        [np.sqrt(3), 2*np.sqrt(3), 0, 1.0],
        [np.sqrt(3), 2*np.sqrt(3), 36, 1.0], # Offset by 36 deg (half of 72)
        [np.sqrt(3), 2.5*np.sqrt(3), 18, 1.0],
        [1.5*np.sqrt(3), 3.0*np.sqrt(3), 0, 1.0],
        [1.5*np.sqrt(3), 3.0*np.sqrt(3), 36, 1.0],
    ]
    
    best_positions = None
    best_side = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Ensure initial config is valid
        pos = generate_positions(init)
        if not check_validity(pos):
            # Scale up until valid
            for scale in [1.1, 1.2, 1.5, 2.0]:
                init_scaled = init.copy()
                init_scaled[3] *= scale
                pos = generate_positions(init_scaled)
                if check_validity(pos):
                    init = init_scaled
                    break
            else:
                continue
                
        try:
            res = minimize(
                objective,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < best_side:
                # Verify the result
                pos = generate_positions(res.x)
                if check_validity(pos):
                    # Find best outer angle
                    min_s = float('inf')
                    best_ang = 0.0
                    for ang in np.arange(0, 30, 2.5):
                        s = required_outer_side(pos, ang)
                        if s < min_s:
                            min_s = s
                            best_ang = ang
                            
                    if min_s < best_side:
                        best_side = min_s
                        best_positions = pos
                        best_params = res.x
                        best_outer_angle = best_ang
        except Exception as e:
            pass

    # Fallback
    if best_positions is None:
        # Use the first valid init
        for init in inits:
            pos = generate_positions(init)
            if check_validity(pos):
                best_positions = pos
                best_side = required_outer_side(pos, 0.0)
                best_outer_angle = 0.0
                break
        if best_positions is None:
            # Last resort: linear array
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions, 0.0)
            best_outer_angle = 0.0

    # Final validation
    if not check_validity(best_positions):
        # This shouldn't happen if the loop above worked, but just in case
        best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
        best_side = required_outer_side(best_positions, 0.0)
        best_outer_angle = 0.0

    # Calculate final exact side
    final_side = required_outer_side(best_positions, best_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), float(best_outer_angle)

# EVOLVE-BLOCK-END
