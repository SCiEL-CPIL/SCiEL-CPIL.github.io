import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Hexagon vertices for unit side length
    def hex_vertices(theta):
        """Return 6 vertices of a unit hexagon centered at origin, rotated by theta."""
        angles = theta + np.arange(6) * np.pi / 3.0
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def get_hex_verts(cx, cy, theta):
        """Get hexagon vertices centered at (cx, cy) with rotation theta."""
        verts = hex_vertices(theta)
        return verts + np.array([cx, cy])
    
    # Non-overlap constraint between two hexagons
    # Using Minkowski sum approach: two convex polygons P and Q don't overlap
    # iff there exists a separating line. We check all 6 directions of the first
    # hexagon's edges (and by symmetry, we need to check if P - Q has a separating axis)
    
    def separation_distance(v1, v2):
        """
        v1, v2: (6,2) arrays of vertices of two hexagons
        Returns the minimum separation distance (positive if non-overlapping, 0 if touching, negative if overlapping)
        """
        # For two convex polygons, we can check all edge normals of both polygons
        # Here we check the 6 normals from each polygon (12 total, but 6 are redundant due to symmetry)
        
        # Get edge normals for polygon 1
        edges1 = np.roll(v1, -1, axis=0) - v1  # 6 edge vectors
        # Outward normals (rotate edge by 90 degrees)
        normals1 = np.column_stack([edges1[:, 1], -edges1[:, 0]])
        normals1 = normals1 / (np.linalg.norm(normals1, axis=1, keepdims=True) + 1e-15)
        
        # For each normal, compute the projection range of both polygons
        max_sep = -np.inf
        
        for n in normals1:
            # Project vertices of polygon 1
            proj1 = v1 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            
            # Project vertices of polygon 2
            proj2 = v2 @ n
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Separation: if max1 < min2 or max2 < min1
            # Distance = min(min2 - max1, min1 - max2) if positive
            sep = min(min2 - max1, min1 - max2)
            max_sep = max(max_sep, sep)
        
        # Also check normals from polygon 2
        edges2 = np.roll(v2, -1, axis=0) - v2
        normals2 = np.column_stack([edges2[:, 1], -edges2[:, 0]])
        normals2 = normals2 / (np.linalg.norm(normals2, axis=1, keepdims=True) + 1e-15)
        
        for n in normals2:
            proj1 = v1 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            
            proj2 = v2 @ n
            min2, max2 = np.min(proj2), np.max(proj2)
            
            sep = min(min2 - max1, min1 - max2)
            max_sep = max(max_sep, sep)
        
        return max_sep
    
    def compute_all_separations(params):
        """
        params: array of shape (2*N + N,) = [x1,y1,theta1, x2,y2,theta2, ..., xN,yN,thetaN]
        Returns list of separation distances for all pairs
        """
        N_hex = len(params) // 3
        seps = []
        for i in range(N_hex):
            xi = params[3*i]
            yi = params[3*i+1]
            ti = params[3*i+2]
            for j in range(i+1, N_hex):
                xj = params[3*j]
                yj = params[3*j+1]
                tj = params[3*j+2]
                v1 = get_hex_verts(xi, yi, ti)
                v2 = get_hex_verts(xj, yj, tj)
                seps.append(separation_distance(v1, v2))
        return seps
    
    def outer_hex_containment_penalty(params, outer_center, outer_side, outer_theta):
        """
        Penalty for hexagons not being contained in outer hexagon.
        outer_center: (2,) array
        outer_side: scalar
        outer_theta: scalar
        """
        N_hex = len(params) // 3
        penalty = 0.0
        
        # Outer hexagon vertices
        outer_verts = np.array([])
        if outer_side > 0:
            outer_scale = outer_side
            angles = outer_theta + np.arange(6) * np.pi / 3.0
            outer_verts = outer_scale * np.column_stack([np.cos(angles), np.sin(angles)])
            outer_verts = outer_verts + outer_center
        
        # For containment, each vertex of each inner hex must be inside the outer hex
        # We use the edge normals of the outer hex to check
        if len(outer_verts) == 0:
            return 1e10
        
        edges_outer = np.roll(outer_verts, -1, axis=0) - outer_verts
        normals_outer = np.column_stack([edges_outer[:, 1], -edges_outer[:, 0]])
        normals_outer = normals_outer / (np.linalg.norm(normals_outer, axis=1, keepdims=True) + 1e-15)
        
        for k in range(6):
            n = normals_outer[k]
            # Project outer hex vertices
            proj_outer = outer_verts @ n
            min_outer = np.min(proj_outer)
            max_outer = np.max(proj_outer)
            
            for i in range(N_hex):
                xi = params[3*i]
                yi = params[3*i+1]
                ti = params[3*i+2]
                v_inner = get_hex_verts(xi, yi, ti)
                proj_inner = v_inner @ n
                min_inner = np.min(proj_inner)
                max_inner = np.max(proj_inner)
                
                # If inner extends beyond outer
                over1 = min_outer - min_inner  # should be <= 0
                over2 = max_inner - max_outer  # should be <= 0
                if over1 > 0:
                    penalty += over1 ** 2
                if over2 > 0:
                    penalty += over2 ** 2
        
        return penalty
    
    def packing_objective(params):
        """
        Objective: minimize outer side length + penalties for non-overlap and non-containment
        params: [x1,y1,theta1, ..., xN,yN,thetaN, outer_cx, outer_cy, outer_theta, outer_side]
        """
        N_hex = len(params) // 3
        if N_hex != N:
            return 1e10
        
        outer_cx = params[3*N]
        outer_cy = params[3*N+1]
        outer_theta = params[3*N+2]
        outer_side = params[3*N+3]
        
        if outer_side <= 0:
            return 1e10
        
        # Non-overlap penalty
        seps = compute_all_separations(params[:3*N])
        overlap_penalty = 0.0
        for s in seps:
            if s < 0:
                overlap_penalty += (-s) ** 2
        
        # Containment penalty
        contain_penalty = outer_hex_containment_penalty(params[:3*N], np.array([outer_cx, outer_cy]), outer_side, outer_theta)
        
        # Objective: side length + weighted penalties
        weight = 10.0
        return outer_side + weight * overlap_penalty + weight * contain_penalty
    
    # Initial guess: place hexagons in a lattice pattern
    def initial_guess():
        """Create initial parameters based on lattice arrangement."""
        # Use axial coordinates for initial placement
        axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        
        # Convert to Cartesian with scale factor for separation
        scale = 1.0
        params = np.zeros(3*N + 4)
        
        for i, (q, r) in enumerate(axial):
            x = math.sqrt(3) * (q + r / 2.0) * scale
            y = 1.5 * r * scale
            params[3*i] = x
            params[3*i+1] = y
            params[3*i+2] = 0.0  # no initial rotation
        
        # Estimate outer hexagon
        pts = np.column_stack([params[0:3*N:3], params[1:3*N:3]])
        if len(pts) > 0:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
        else:
            cx, cy = 0, 0
        
        # Estimate side length from max projection
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        
        side_est = (max_proj + APOTHEM) / APOTHEM
        
        params[3*N] = cx
        params[3*N+1] = cy
        params[3*N+2] = 0.0
        params[3*N+3] = side_est
        
        return params
    
    x0 = initial_guess()
    
    # Try multiple random restarts
    best_result = None
    best_S = float('inf')
    
    n_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        if restart == 0:
            x0_curr = x0.copy()
        else:
            x0_curr = x0.copy()
            # Add small perturbations
            perturb = rng.normal(0, 0.1, size=x0_curr.shape)
            x0_curr[:3*N] += perturb[:3*N]
            x0_curr[3*N:3*N+2] += perturb[3*N:3*N+2] * 0.1
            x0_curr[3*N+2] = rng.uniform(0, math.pi/6)
            x0_curr[3*N+3] *= (1 + rng.normal(0, 0.05))
        
        try:
            # Use SLSQP for constrained optimization
            result = minimize(packing_objective, x0_curr, method='SLSQP',
                             options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.fun < 1e9:
                # Extract side length from result
                S_est = result.x[3*N+3]
                
                # Verify non-overlap and containment
                params_verified = result.x[:3*N]
                seps = compute_all_separations(params_verified)
                min_sep = min(seps) if seps else 1e10
                
                outer_cx = result.x[3*N]
                outer_cy = result.x[3*N+1]
                outer_theta = result.x[3*N+2]
                outer_side = result.x[3*N+3]
                
                contain_pen = outer_hex_containment_penalty(params_verified, np.array([outer_cx, outer_cy]), outer_side, outer_theta)
                
                if min_sep >= -1e-6 and contain_pen < 1e-6:
                    if S_est < best_S:
                        best_S = S_est
                        best_result = result.x.copy()
        except Exception as e:
            pass
    
    # If optimization failed, fall back to lattice solution
    if best_result is None:
        best_result = x0.copy()
        # Compute actual side length
        params_verified = best_result[:3*N]
        outer_cx = best_result[3*N]
        outer_cy = best_result[3*N+1]
        outer_theta = best_result[3*N+2]
        
        # Recompute proper side length
        pts = np.column_stack([params_verified[0::3], params_verified[1::3]])
        max_proj = 0
        for k in range(6):
            ang = outer_theta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - outer_cx) * nx + (pts[:, 1] - outer_cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM
        best_result[3*N+3] = best_S
    
    # Extract final parameters
    params_final = best_result
    inner = []
    for i in range(N):
        x = float(params_final[3*i])
        y = float(params_final[3*i+1])
        theta = float(params_final[3*i+2])
        inner.append([x, y, math.degrees(theta)])
    
    center = [float(params_final[3*N]), float(params_final[3*N+1])]
    side = float(params_final[3*N+3])
    rotation = float(math.degrees(params_final[3*N+2]))
    
    # Final verification and adjustment
    # Ensure hexagons are properly contained by adjusting side length if needed
    pts = np.column_stack([inner[i][0] for i in range(N)], 
                          [inner[i][1] for i in range(N)])
    angles = [inner[i][2] * math.pi / 180.0 for i in range(N)]
    
    max_proj = 0
    for k in range(6):
        ang = rotation * math.pi / 180.0 + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        for i in range(N):
            cx_i, cy_i, ti = inner[i][0], inner[i][1], angles[i]
            verts = get_hex_verts(cx_i, cy_i, ti)
            proj = verts @ np.array([nx, ny])
            proj_centered = proj - (cx_i * nx + cy_i * ny)
            # Project relative to outer center
            rel_proj = ((verts - np.array([center[0], center[1]])) @ np.array([nx, ny]))
            max_proj = max(max_proj, np.max(rel_proj), -np.min(rel_proj))
    
    final_side = (max_proj + APOTHEM) / APOTHEM
    side = float(final_side)
    
    return inner, center, side, rotation
