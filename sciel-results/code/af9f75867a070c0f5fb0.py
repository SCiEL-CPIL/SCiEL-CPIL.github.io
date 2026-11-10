# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from itertools import combinations
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    time_limit = 630.0
    n = 11

    # Precompute geometry
    inner_angles_rad = np.deg2rad([0, 60, 120, 180, 240, 300])
    inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles_rad])
    
    def get_hex_verts(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + s*np.cos(angles), cy + s*np.sin(angles)])
    
    def check_separation(verts1, verts2):
        """
        Check if two convex polygons are separated (non-overlapping).
        Returns True if separated (no intersection, touching allowed with tolerance).
        Uses the Separating Axis Theorem.
        """
        # Get edge normals for both polygons
        def get_normals(verts):
            normals = []
            for i in range(len(verts)):
                p1 = verts[i]
                p2 = verts[(i+1)%len(verts)]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normals.append(normal / norm)
            return normals
        
        normals1 = get_normals(verts1)
        normals2 = get_normals(verts2)
        all_normals = normals1 + normals2
        
        for n_vec in all_normals:
            # Project both polygons onto the normal
            proj1 = verts1 @ n_vec
            proj2 = verts2 @ n_vec
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
        
        return False

    def is_valid_packing(centers, angles_deg, outer_center, outer_angle_deg, side):
        # Check all pairs of inner hexagons
        for i in range(n):
            for j in range(i+1, n):
                v1 = get_hex_verts(centers[i,0], centers[i,1], 1.0, angles_deg[i])
                v2 = get_hex_verts(centers[j,0], centers[j,1], 1.0, angles_deg[j])
                if not check_separation(v1, v2):
                    return False
        
        # Check containment
        outer_verts = get_hex_verts(outer_center[0], outer_center[1], side, outer_angle_deg)
        
        # For each inner hexagon vertex, check if it's inside the outer hexagon
        # Using the cross product method for convex polygon containment
        def point_in_convex_polygon(p, poly):
            n_poly = len(poly)
            for i in range(n_poly):
                p1 = poly[i]
                p2 = poly[(i+1)%n_poly]
                cross = np.cross(p2 - p1, p - p1)
                if cross < -1e-6:
                    return False
            return True
        
        for i in range(n):
            v1 = get_hex_verts(centers[i,0], centers[i,1], 1.0, angles_deg[i])
            for p in v1:
                if not point_in_convex_polygon(p, outer_verts):
                    return False
        
        return True

    def compute_min_side(centers, angles_deg, outer_center, outer_angle_deg):
        """
        Compute the minimum side length of the outer hexagon (fixed center and angle)
        such that all inner hexagons fit inside.
        """
        max_proj = 0.0
        # Normals of the outer hexagon
        outer_normals_angles = np.deg2rad([30, 90, 150, 210, 270, 330]) + np.deg2rad(outer_angle_deg)
        outer_normals = np.array([[np.cos(a), np.sin(a)] for a in outer_normals_angles])
        
        for i in range(n):
            verts = get_hex_verts(centers[i,0], centers[i,1], 1.0, angles_deg[i])
            verts_shifted = verts - outer_center
            projs = verts_shifted @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
            
        # Side length S relates to apothem A by A = S * sqrt(3) / 2
        # So S = 2 * A / sqrt(3)
        # The max projection is the apothem required.
        side = max_proj * 2 / np.sqrt(3)
        return side

    def objective(params):
        """
        params: [cx0, cy0, a0, cx1, cy1, a1, ..., cx10, cy10, a10, ox, oy, oa]
        """
        centers = params[:2*n].reshape(n, 2)
        angles_deg = params[2*n:3*n]
        outer_center = params[3*n:3*n+2]
        outer_angle_deg = params[3*n+2]
        
        # First, compute the side length required for the current configuration
        side = compute_min_side(centers, angles_deg, outer_center, outer_angle_deg)
        
        # Check validity with a slightly larger side to allow for numerical errors
        if not is_valid_packing(centers, angles_deg, outer_center, outer_angle_deg, side * 1.01):
            # Penalty for invalid packing
            # We need to estimate how much invalidity there is.
            # For now, just return a large penalty.
            return 100.0 + side
            
        return side

    # Initial configuration: Hexagonal cluster
    # Centers in a hex grid
    # Hex grid basis:
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 1.5)
    # Distance between centers is sqrt(3)
    
    def build_initial_config():
        # 11 hexagons
        # Center at (0,0)
        # Ring 1: 6 hexagons
        # Ring 2: 5 hexagons (to make 11)
        
        centers = np.zeros((n, 2))
        angles_deg = np.zeros(n)
        
        # Center
        centers[0] = [0, 0]
        
        # Ring 1
        r = np.sqrt(3)
        for i in range(6):
            angle = np.deg2rad(60 * i)
            centers[i+1] = [r * np.cos(angle), r * np.sin(angle)]
            
        # Ring 2 (partial)
        # We need 4 more.
        # Positions in Ring 2 are at distance 2*sqrt(3) or sqrt(3)*sqrt(3+1) ?
        # In a hex grid, the next ring is at distance 2*sqrt(3) in the 6 main directions,
        # and at distance sqrt(3)*sqrt(7) in the 6 intermediate directions?
        # Let's just use a simple grid.
        # (0,0), (1,0), (0,1), (1,1), (2,0), (2,1), (3,0), (0,2), (1,2), (2,2), (3,1)
        # This is a 4x3 grid minus one corner.
        # Let's use a more compact shape.
        # A "plus" shape or similar.
        
        # Let's restart with a known good compact shape.
        # 1 in center, 6 around, 4 on top of the ring.
        
        # Coordinates for 11 hexagons in a compact cluster
        # Using axial coordinates (q, r)
        # (0,0)
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # (2,0), (1,1), (0,2), (-1,2) -> 1+6+4 = 11
        
        axial_coords = [
            (0,0),
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1), (0,2), (-1,2)
        ]
        
        for i, (q, r) in enumerate(axial_coords):
            x = r * np.sqrt(3) / 2 + q * np.sqrt(3)
            y = r * 1.5
            centers[i] = [x, y]
            
        return centers, angles_deg

    centers0, angles_deg0 = build_initial_config()
    outer_center0 = np.array([0.0, 0.0])
    outer_angle0 = 0.0
    
    params0 = np.concatenate([centers0.flatten(), angles_deg0, outer_center0, [outer_angle0]])
    
    best_params = params0
    best_score = objective(params0)
    
    # Optimization Loop
    # Use Nelder-Mead first, then L-BFGS-B
    
    time_left = time_limit - (time.time() - start_time)
    
    # Stage 1: Nelder-Mead
    try:
        res = minimize(objective, params0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception as e:
        pass

    if time.time() - start_time > time_limit - 120:
        return _format_result(best_params)
        
    # Stage 2: L-BFGS-B (might fail due to non-differentiability, but worth a try)
    try:
        res2 = minimize(objective, best_params, method='L-BFGS-B', 
                        options={'maxiter': 200})
        if res2.fun < best_score:
            best_params = res2.x
            best_score = res2.fun
    except Exception as e:
        pass

    # Stage 3: Local refinement
    # Try to refine each parameter individually
    for _ in range(3): # 3 iterations of local refinement
        if time.time() - start_time > time_limit - 60:
            break
            
        current_params = best_params.copy()
        for i in range(len(current_params)):
            # Try small perturbations
            for delta in [1e-4, -1e-4, 1e-3, -1e-3]:
                new_params = current_params.copy()
                new_params[i] += delta
                score = objective(new_params)
                if score < best_score:
                    best_score = score
                    best_params = new_params
                    current_params = new_params.copy()

    return _format_result(best_params)

def _format_result(params):
    n = 11
    centers = params[:2*n].reshape(n, 2)
    angles_deg = params[2*n:3*n]
    outer_center = params[3*n:3*n+2]
    outer_angle_deg = params[3*n+2]
    
    inner_angles_rad = np.deg2rad([0, 60, 120, 180, 240, 300])
    inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles_rad])
    
    def get_hex_verts(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + s*np.cos(angles), cy + s*np.sin(angles)])
    
    # Compute side
    max_proj = 0.0
    outer_normals_angles = np.deg2rad([30, 90, 150, 210, 270, 330]) + np.deg2rad(outer_angle_deg)
    outer_normals = np.array([[np.cos(a), np.sin(a)] for a in outer_normals_angles])
    
    for i in range(n):
        verts = get_hex_verts(centers[i,0], centers[i,1], 1.0, angles_deg[i])
        verts_shifted = verts - outer_center
        projs = verts_shifted @ outer_normals.T
        max_proj = max(max_proj, np.max(projs))
        
    side = max_proj * 2 / np.sqrt(3)
    side += 1e-5 # Safety margin
    
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = angles_deg
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle_deg])
    
    return inner_hex_data, outer_hex_data, float(side)

# EVOLVE-BLOCK-END
