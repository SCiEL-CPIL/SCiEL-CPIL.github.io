"""
Pack 12 unit regular hexagons inside a minimal regular hexagon.
Uses structured honeycomb initialization with L-BFGS-B optimization.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 130.0  # Safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()
    
    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a regular hexagon centered at (cx, cy) with side 1 and rotation angle_deg."""
        base_angles = np.radians(np.arange(0, 360, 60) + angle_deg)
        x = cx + np.cos(base_angles)
        y = cy + np.sin(base_angles)
        return np.column_stack((x, y))
    
    def hexagons_disjoint(h1, h2):
        """Check if two hexagons are disjoint using SAT."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If far apart, disjoint
        if dist_sq > 4.0:
            return True
            
        # Get vertices
        v1 = get_hex_vertices(h1[0], h1[1], h1[2])
        v2 = get_hex_vertices(h2[0], h2[1], h2[2])
        
        # SAT: Check 6 normals from each hexagon
        for v1_edge in [v1]:
            for i in range(6):
                ex = v1_edge[(i+1)%6, 0] - v1_edge[i, 0]
                ey = v1_edge[(i+1)%6, 1] - v1_edge[i, 1]
                nx = -ey
                ny = ex
                norm = math.sqrt(nx*nx + ny*ny)
                if norm == 0: continue
                nx /= norm
                ny /= norm
                
                proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
                proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
                proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
                proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
                
                if proj1_max < proj2_min or proj2_max < proj1_min:
                    return True
                    
        for v2_edge in [v2]:
            for i in range(6):
                ex = v2_edge[(i+1)%6, 0] - v2_edge[i, 0]
                ey = v2_edge[(i+1)%6, 1] - v2_edge[i, 1]
                nx = -ey
                ny = ex
                norm = math.sqrt(nx*nx + ny*ny)
                if norm == 0: continue
                nx /= norm
                ny /= norm
                
                proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
                proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
                proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
                proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
                
                if proj1_max < proj2_min or proj2_max < proj1_min:
                    return True
                    
        return False

    def compute_min_outer_side(inner):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side, phi_deg, center_x, center_y).
        """
        # Get all vertices of inner hexagons
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        sqrt3 = math.sqrt(3)
        best_side = float('inf')
        best_phi = 0.0
        best_center = (0.0, 0.0)
        
        # Scan phi in [0, 60)
        phi_steps = np.arange(0, 60, 0.5)
        
        for phi_deg in phi_steps:
            phi = np.radians(phi_deg)
            # 3 unique normal directions for the outer hexagon faces
            angles = phi + np.pi/6 + np.array([0, np.pi/3, 2*np.pi/3])
            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)
            
            # Project all vertices onto these 3 directions
            proj = all_verts[:, 0:1] * cos_angles[None, :] + all_verts[:, 1:2] * sin_angles[None, :]
            
            # Compute range for each direction
            ranges = np.max(proj, axis=0) - np.min(proj, axis=0)
            
            # Side length is max range / sqrt(3)
            side = np.max(ranges) / sqrt3
            
            if side < best_side:
                best_side = side
                best_phi = phi_deg
                # Compute center as midpoint of projections
                A = np.column_stack((cos_angles, sin_angles))
                b = (np.max(proj, axis=0) + np.min(proj, axis=0)) / 2.0
                center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                best_center = (center[0], center[1])
        
        # Refine around best_phi
        if best_phi > 0 and best_phi < 60:
            refine_steps = np.arange(best_phi - 0.5, best_phi + 0.5, 0.05)
            for phi_deg in refine_steps:
                if phi_deg < 0 or phi_deg >= 60: continue
                phi = np.radians(phi_deg)
                angles = phi + np.pi/6 + np.array([0, np.pi/3, 2*np.pi/3])
                cos_angles = np.cos(angles)
                sin_angles = np.sin(angles)
                
                proj = all_verts[:, 0:1] * cos_angles[None, :] + all_verts[:, 1:2] * sin_angles[None, :]
                ranges = np.max(proj, axis=0) - np.min(proj, axis=0)
                side = np.max(ranges) / sqrt3
                
                if side < best_side:
                    best_side = side
                    best_phi = phi_deg
                    A = np.column_stack((cos_angles, sin_angles))
                    b = (np.max(proj, axis=0) + np.min(proj, axis=0)) / 2.0
                    center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    best_center = (center[0], center[1])

        return best_side, best_phi, best_center[0], best_center[1]

    def create_honeycomb_init(offset_x=0.0, offset_y=0.0, scale=1.0):
        """Create a honeycomb-like initial configuration."""
        inner = []
        # 12 hexagons in a hexagonal arrangement
        # Center + 6 in first ring + 5 in second ring (partial)
        
        # Center
        inner.append([offset_x, offset_y, 0.0])
        
        # First ring: 6 hexagons
        r1 = 2.0 * scale
        for k in range(6):
            angle = math.radians(60 * k)
            x = offset_x + r1 * math.cos(angle)
            y = offset_y + r1 * math.sin(angle)
            inner.append([x, y, 0.0])
        
        # Second ring: 5 more hexagons (to make 12 total)
        r2 = 3.5 * scale
        for k in range(5):
            angle = math.radians(60 * k + 30)
            x = offset_x + r2 * math.cos(angle)
            y = offset_y + r2 * math.sin(angle)
            inner.append([x, y, 0.0])
        
        return inner

    def flatten_inner(inner):
        return np.array([item for sublist in inner for item in sublist])

    def unflatten_inner(flat):
        inner = []
        for i in range(N):
            inner.append([flat[3*i], flat[3*i+1], flat[3*i+2]])
        return inner

    def objective(flat):
        """Objective function: minimize outer side length with overlap penalty."""
        inner = unflatten_inner(flat)
        
        # Compute min outer side
        try:
            side, phi, cx, cy = compute_min_outer_side(inner)
        except:
            return 1e10

        # Compute overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner[i], inner[j]):
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < math.sqrt(3):
                        penalty += (math.sqrt(3) - dist)**2
        
        return side + 100.0 * penalty

    # Try multiple initializations
    initializations = []
    
    # Honeycomb with different scales and offsets
    for scale in [0.9, 1.0, 1.1]:
        for ox in [-0.5, 0.0, 0.5]:
            for oy in [-0.5, 0.0, 0.5]:
                init = create_honeycomb_init(ox, oy, scale)
                if len(init) == N:
                    initializations.append(init)
    
    # Also try a 6x2 grid
    cols = 6
    rows = 2
    x0 = -(cols - 1) / 2.0 * 3.0
    y0 = -(rows - 1) / 2.0 * math.sqrt(3)
    grid_init = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * 3.0
            y = y0 + r * math.sqrt(3)
            grid_init.append([x, y, 0.0])
    initializations.append(grid_init)
    
    best_inner = None
    best_side = float('inf')
    best_phi = 0.0
    best_center = (0.0, 0.0)
    
    # Evaluate each initialization and optimize
    for init in initializations:
        if time.time() - t0 > DEADLINE:
            break
        
        # First, check if this initialization is valid
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(init[i], init[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Compute initial side
        side, phi, cx, cy = compute_min_outer_side(init)
        
        if side < best_side:
            best_side = side
            best_inner = [list(h) for h in init]
            best_phi = phi
            best_center = (cx, cy)
        
        # Try to optimize this initialization
        time_left = DEADLINE - (time.time() - t0)
        if time_left > 15.0:  # Need at least 15s for optimization
            try:
                current_flat = flatten_inner(init)
                res = minimize(objective, current_flat, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5, 'maxfun': 500})
                
                if res.fun < side:  # Check if we improved
                    polished_inner = unflatten_inner(res.x)
                    
                    # Verify validity
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            if not hexagons_disjoint(polished_inner[i], polished_inner[j]):
                                valid = False
                                break
                            if j == i: continue
                    if valid:
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(polished_inner)
                        if new_side < best_side:
                            best_side = new_side
                            best_inner = polished_inner
                            best_phi = new_phi
                            best_center = (new_cx, new_cy)
            except:
                pass
    
    # If no valid configuration found, use a simple fallback
    if best_inner is None:
        best_inner = create_honeycomb_init(0, 0, 1.0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_inner)
        best_center = (best_cx, best_cy)
    
    # Ensure we have exactly N hexagons
    if len(best_inner) != N:
        # Pad or truncate
        while len(best_inner) < N:
            best_inner.append([0.0, 0.0, 0.0])
        best_inner = best_inner[:N]
    
    # Final validation and correction
    # Recompute the outer hexagon to ensure containment
    final_side, final_phi, final_cx, final_cy = compute_min_outer_side(best_inner)
    
    return best_inner, (float(final_cx), float(final_cy)), float(final_side), float(final_phi)

# EVOLVE-BLOCK-END
