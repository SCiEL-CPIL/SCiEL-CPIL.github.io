import numpy as np
from scipy.optimize import minimize
import time

def construct():
    N = 12
    start_time = time.time()
    
    # Unit hexagon vertices (pointy-top, side length 1)
    # Angles: 90, 150, 210, 270, 330, 30 degrees
    base_angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    hex_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def get_all_vertices(params):
        """Get all 72 vertices from 12 hexagons."""
        all_verts = []
        for i in range(N):
            cx, cy, theta = params[3*i], params[3*i+1], params[3*i+2]
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            rv = R @ hex_verts.T + np.array([cx, cy])
            all_verts.append(rv.T)
        return np.vstack(all_verts)
    
    def min_enclosing_hexagon_L(all_verts):
        """Find minimum L and optimal rotation for enclosing hexagon."""
        # The outer hexagon has 6 sides. For a pointy-top hexagon with side L,
        # the constraint for a point (x,y) is:
        # |y| <= L, |x| <= L*sqrt(3)/2, |y + x/sqrt(3)| <= L, |y - x/sqrt(3)| <= L
        # Actually for a regular hexagon with "radius" (center to vertex) = L:
        # The 6 directions are at angles 0, 60, 120, 180, 240, 300
        # A point is inside if max over 6 directions of |dot(p, n_k)| <= L
        # where n_k are unit vectors at those angles
        # But wait, for a regular hexagon, the support function gives the max projection
        # onto the 6 normal directions. For a hexagon with side length s, the apothem is s*sqrt(3)/2
        # and the circumradius is s.
        # The normals to the sides are at angles 30, 90, 150, 210, 270, 330 for pointy-top
        # Actually, let me reconsider. A regular hexagon with vertices at angles 90,150,210,270,330,30
        # has sides perpendicular to directions 30, 90, 150, 210, 270, 330.
        # The distance from center to each side (apothem) = s * sqrt(3)/2 where s is side length.
        # So for containment: for each of 6 normal directions n_k, max |dot(p - c, n_k)| <= s*sqrt(3)/2
        # This means L = s*sqrt(3)/2 is the apothem.
        # But the task says "side length" of outer hexagon. So we want to minimize s.
        # s = 2*L/sqrt(3) where L is the apothem.
        
        # Let's parametrize by rotation phi of the outer hexagon.
        # Normal directions: phi + 30, phi + 90, phi + 150, phi + 210, phi + 270, phi + 330 (degrees)
        # For a given phi, the required apothem is:
        # L_apothem = max over all points p of max over 6 directions of |dot(p, n_k)|
        # (assuming center at origin; we'll optimize center separately)
        
        # For efficiency, let's just compute for a range of phi and find minimum.
        best_L = float('inf')
        best_phi = 0
        
        # Coarse search over phi
        for phi_deg in range(0, 60, 2):
            phi = np.deg2rad(phi_deg)
            normals = np.column_stack([np.cos(phi + np.deg2rad([30, 90, 150, 210, 270, 330])),
                                       np.sin(phi + np.deg2rad([30, 90, 150, 210, 270, 330]))])
            # Project all vertices onto each normal
            projs = all_verts @ normals.T  # shape (72, 6)
            # For each normal direction, we need |proj| <= L_apothem
            # But this assumes center at origin. The optimal center for fixed phi is the midpoint
            # of the range of projections in each direction.
            proj_max = projs.max(axis=0)
            proj_min = projs.min(axis=0)
            # The required half-width in each direction is (proj_max - proj_min)/2
            # But we need the same L for all 6 directions, so L_apothem = max of these half-widths
            half_widths = (proj_max - proj_min) / 2
            L_apothem = np.max(half_widths)
            
            if L_apothem < best_L:
                best_L = L_apothem
                best_phi = phi
        
        # Fine search
        for phi_deg in np.linspace(best_phi - np.deg2rad(2), best_phi + np.deg2rad(2), 20):
            phi = np.deg2rad(phi_deg)
            normals = np.column_stack([np.cos(phi + np.deg2rad([30, 90, 150, 210, 270, 330])),
                                       np.sin(phi + np.deg2rad([30, 90, 150, 210, 270, 330]))])
            projs = all_verts @ normals.T
            proj_max = projs.max(axis=0)
            proj_min = projs.min(axis=0)
            half_widths = (proj_max - proj_min) / 2
            L_apothem = np.max(half_widths)
            
            if L_apothem < best_L:
                best_L = L_apothem
                best_phi = phi
        
        return best_L, best_phi
    
    def pairwise_min_distance(params):
        """Compute minimum distance between any two hexagon edges.
        For simplicity, use center-to-center distance minus 2*apothem as a proxy,
        but this is only valid for same orientation. For different orientations,
        we need to check edge-edge distances.
        
        A simpler but less accurate approach: check all vertex-edge distances.
        For each pair of hexagons, check if any vertex of one is inside the other,
        and compute min distance between edges.
        
        For speed, let's use a conservative lower bound: 
        min over pairs of (center_dist - 2*circumradius)
        where circumradius = 1 (side length 1 hexagon has circumradius 1).
        This gives a lower bound on the actual minimum edge distance.
        If this is positive, hexagons don't overlap. If negative, they might overlap.
        
        Actually, for regular hexagons with side 1, the apothem is sqrt(3)/2 ≈ 0.866.
        Two hexagons with same orientation don't overlap if center_dist >= 2*apothem = sqrt(3).
        With different orientations, the condition is more complex.
        
        For the penalty, let's compute the actual min distance between edges for each pair.
        """
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                cx1, cy1, th1 = params[3*i], params[3*i+1], params[3*i+2]
                cx2, cy2, th2 = params[3*j], params[3*j+1], params[3*j+2]
                
                # Get vertices of both hexagons
                c1, s1 = np.cos(th1), np.sin(th1)
                R1 = np.array([[c1, -s1], [s1, c1]])
                verts1 = R1 @ hex_verts.T + np.array([cx1, cy1])
                
                c2, s2 = np.cos(th2), np.sin(th2)
                R2 = np.array([[c2, -s2], [s2, c2]])
                verts2 = R2 @ hex_verts.T + np.array([cx2, cy2])
                
                # Check all vertex-edge pairs
                # For each vertex of hex1, find min distance to edges of hex2
                # and vice versa
                for v in verts1:
                    for k in range(6):
                        v1 = verts2[k]
                        v2 = verts2[(k+1)%6]
                        # Distance from point v to line segment v1-v2
                        d = point_segment_distance(v, v1, v2)
                        if d < min_dist:
                            min_dist = d
                for v in verts2:
                    for k in range(6):
                        v1 = verts1[k]
                        v2 = verts1[(k+1)%6]
                        d = point_segment_distance(v, v1, v2)
                        if d < min_dist:
                            min_dist = d
        
        return min_dist
    
    def point_segment_distance(p, a, b):
        """Distance from point p to line segment ab."""
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
    
    def objective(params):
        """Objective: minimize outer hexagon side length + penalty for overlap."""
        if time.time() - start_time > 110:
            return 1e10
        
        # Compute enclosing hexagon
        all_verts = get_all_vertices(params)
        L_apothem, phi = min_enclosing_hexagon_L(all_verts)
        side_length = 2 * L_apothem / np.sqrt(3)
        
        # Compute overlap penalty
        min_dist = pairwise_min_distance(params)
        if min_dist < 0:
            penalty = 100 * min_dist**2
        else:
            penalty = 0
        
        return side_length + penalty
    
    # Initial guess: hexagonal tiling
    centers = []
    for i in range(-1, 2):
        for j in range(-1, 3):
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    init = np.zeros(3*N)
    for i in range(N):
        init[3*i] = centers[i, 0]
        init[3*i+1] = centers[i, 1]
        init[3*i+2] = 0
    
    # Optimize
    result = minimize(objective, init, method='L-BFGS-B', 
                      options={'maxiter': 100, 'maxfun': 5000, 'maxls': 30})
    
    if time.time() - start_time > 115:
        best_params = init
    else:
        best_params = result.x
    
    # Compute final L and phi
    all_verts = get_all_vertices(best_params)
    final_L_apothem, final_phi = min_enclosing_hexagon_L(all_verts)
    final_side = 2 * final_L_apothem / np.sqrt(3)
    
    # Compute optimal center for the outer hexagon
    normals = np.column_stack([np.cos(final_phi + np.deg2rad([30, 90, 150, 210, 270, 330])),
                               np.sin(final_phi + np.deg2rad([30, 90, 150, 210, 270, 330]))])
    projs = all_verts @ normals.T
    proj_max = projs.max(axis=0)
    proj_min = projs.min(axis=0)
    midpoints = (proj_max + proj_min) / 2
    c_outer = np.linalg.pinv(normals.T) @ midpoints
    
    # Construct output
    inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        theta = best_params[3*i+2]
        inner.append([cx, cy, np.rad2deg(theta)])
    
    return inner, (c_outer[0], c_outer[1]), final_side, np.rad2deg(final_phi)
