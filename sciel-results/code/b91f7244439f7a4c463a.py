"""
Strategy: Global Topology-First Approach.
1. Define a set of topological templates (Ring, Double Ring, Lattice, etc.) that are 
   structurally guaranteed to be non-overlapping for a range of parameters.
2. Use Differential Evolution to globally optimize the continuous parameters of these 
   templates (radii, rotations, global orientation) to minimize the enclosing hexagon size.
3. Use L-BFGS-B for final local polishing.
4. This avoids the expensive SAT overlap checks during the main search, allowing for 
   much faster and more robust global exploration.
"""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Precompute edge normals for a standard hexagon (vertices at 0, 60, ...)
    # Edge i is perpendicular to radius at 30 + i*60
    edge_normals = np.array([
        [math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] 
        for i in range(6)
    ])
    
    # Precompute vertices of a unit hexagon centered at origin (angle 0)
    unit_verts = np.array([
        [math.cos(math.radians(i*60)), math.sin(math.radians(i*60))] 
        for i in range(6)
    ])

    def get_vertices(cx, cy, angle_rad, side=1.0):
        """Get vertices of a regular hexagon."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return np.array([cx, cy]) + side * (unit_verts @ R.T)

    def calculate_L(centers, angles_rad, outer_angle_rad=0.0):
        """Calculate minimum outer side length L to enclose all inner hexagons."""
        c, s = math.cos(-outer_angle_rad), math.sin(-outer_angle_rad)
        R_outer_inv = np.array([[c, -s], [s, c]])
        
        max_proj = 0.0
        for i in range(N):
            verts = get_vertices(centers[i][0], centers[i][1], angles_rad[i])
            # Rotate vertices to outer frame
            verts_rot = verts @ R_outer_inv.T
            # Project onto outer edge normals
            projs = verts_rot @ edge_normals.T
            local_max = projs.max(axis=0).max()
            if local_max > max_proj:
                max_proj = local_max
        
        return 2.0 * max_proj / sqrt3

    def check_overlap_fast(centers, angles_rad, eps=1e-7):
        """
        Check if any two hexagons overlap using Separating Axis Theorem (SAT).
        """
        n = len(centers)
        all_verts = []
        all_normals = []
        
        for i in range(n):
            cx, cy = centers[i]
            ang = angles_rad[i]
            verts = get_vertices(cx, cy, ang)
            all_verts.append(verts)
            
            c, s = math.cos(ang), math.sin(ang)
            R = np.array([[c, -s], [s, c]])
            normals = edge_normals @ R.T
            all_normals.append(normals)

        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.5: 
                    continue

                v_i = all_verts[i]
                v_j = all_verts[j]
                n_i = all_normals[i]
                n_j = all_normals[j]
                
                overlap = True
                for k in range(6):
                    axis = n_i[k]
                    proj_i = v_i @ axis
                    min_i, max_i = proj_i.min(), proj_i.max()
                    proj_j = v_j @ axis
                    min_j, max_j = proj_j.min(), proj_j.max()
                    
                    if max_i < min_j + eps or max_j < min_i + eps:
                        overlap = False
                        break
                    
                    axis = n_j[k]
                    proj_i = v_i @ axis
                    min_i, max_i = proj_i.min(), proj_i.max()
                    proj_j = v_j @ axis
                    min_j, max_j = proj_j.min(), proj_j.max()
                    
                    if max_i < min_j + eps or max_j < min_i + eps:
                        overlap = False
                        break
                
                if overlap:
                    return True
        return False

    # --- Template Definitions ---
    # Each template returns (centers, angles) given parameters.
    # Parameters are chosen to ensure non-overlap structurally where possible,
    # or the optimizer will push them into valid regions if the penalty is high.
    # However, we design them to be valid for a wide range.

    def template_ring(params):
        """
        12 hexagons in a single ring.
        params: [radius, global_angle, individual_angle_offset]
        """
        r = params[0]
        ga = params[1]
        ia = params[2]
        
        centers = []
        angles = []
        for i in range(N):
            theta = ga + 2 * math.pi * i / N
            cx = r * math.cos(theta)
            cy = r * math.sin(theta)
            # Orient hexagon to face outward or align with ring
            ang = theta + ia
            centers.append((cx, cy))
            angles.append(ang)
        return centers, angles

    def template_double_ring(params):
        """
        6 hexagons in inner ring, 6 in outer ring.
        params: [r_inner, r_outer, angle_inner, angle_outer, offset]
        """
        r_in = params[0]
        r_out = params[1]
        a_in = params[2]
        a_out = params[3]
        offset = params[4]
        
        centers = []
        angles = []
        
        # Inner 6
        for i in range(6):
            theta = a_in + 2 * math.pi * i / 6
            cx = r_in * math.cos(theta)
            cy = r_in * math.sin(theta)
            centers.append((cx, cy))
            angles.append(theta) # Face outward
            
        # Outer 6
        for i in range(6):
            theta = a_out + offset + 2 * math.pi * i / 6
            cx = r_out * math.cos(theta)
            cy = r_out * math.sin(theta)
            centers.append((cx, cy))
            angles.append(theta)
            
        return centers, angles

    def template_lattice(params):
        """
        Hexagonal lattice structure.
        We use a subset of a larger lattice.
        Coords: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7
        Add: (2,0), (1,1), (0,2), (-1,2), (-2,1) -> 5 more. Total 12.
        params: [scale, rotation]
        """
        scale = params[0]
        rot = params[1]
        
        hex_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        
        c_rot, s_rot = math.cos(rot), math.sin(rot)
        R = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
        
        centers = []
        angles = []
        for q, r in hex_coords:
            # Axial to Cartesian (flat top)
            cx = sqrt3 * (q + r/2.0) * scale
            cy = 1.5 * r * scale
            # Apply rotation
            p = np.array([cx, cy]) @ R.T
            centers.append((p[0], p[1]))
            angles.append(rot)
            
        return centers, angles

    def template_star(params):
        """
        1 in center, 5 in inner ring, 6 in outer ring? 
        Let's do: 1 center, 5 around it (pentagon-like), 6 around that.
        Actually, let's stick to hex symmetry.
        1 center, 6 inner, 5 outer? No, 12 total.
        1 center + 5 inner + 6 outer = 12.
        params: [r_in, r_out, a_in, a_out]
        """
        r_in = params[0]
        r_out = params[1]
        a_in = params[2]
        a_out = params[3]
        
        centers = [(0,0)]
        angles = [0.0]
        
        for i in range(5):
            theta = a_in + 2 * math.pi * i / 5
            cx = r_in * math.cos(theta)
            cy = r_in * math.sin(theta)
            centers.append((cx, cy))
            angles.append(theta)
            
        for i in range(6):
            theta = a_out + 2 * math.pi * i / 6
            cx = r_out * math.cos(theta)
            cy = r_out * math.sin(theta)
            centers.append((cx, cy))
            angles.append(theta)
            
        return centers, angles

    def objective_template(template_func, params):
        """
        Objective function for a specific template.
        """
        try:
            centers, angles = template_func(params)
            
            # Quick sanity check: if centers are too close, penalty
            # But we rely on the structure. Let's add a soft penalty for very close centers
            # to guide the optimizer away from invalid regions if the template isn't rigid.
            min_dist_sq = float('inf')
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    d2 = dx*dx + dy*dy
                    if d2 < min_dist_sq:
                        min_dist_sq = d2
                    
            # Distance 2.0 is vertex touch. 1.732 is side touch.
            # If min_dist_sq < (1.732)^2 ~ 3.0, they likely overlap.
            # We use a penalty to discourage this, but the main driver is L.
            penalty = 0.0
            if min_dist_sq < 2.8:
                penalty = (2.8 - min_dist_sq) * 100.0
            
            L = calculate_L(centers, angles, 0.0)
            return L + penalty
        except Exception:
            return 1000.0

    best_L = float('inf')
    best_centers = None
    best_angles = None
    
    # Define bounds for each template
    # Template 1: Ring
    bounds_ring = [(1.0, 5.0), (0.0, 2*math.pi), (0.0, math.pi/3)]
    
    # Template 2: Double Ring
    bounds_dr = [(0.5, 3.0), (2.0, 6.0), (0.0, 2*math.pi), (0.0, 2*math.pi), (0.0, math.pi/3)]
    
    # Template 3: Lattice
    bounds_lat = [(0.8, 2.0), (0.0, 2*math.pi)]
    
    # Template 4: Star
    bounds_star = [(1.5, 4.0), (3.0, 6.0), (0.0, 2*math.pi), (0.0, 2*math.pi)]

    templates = [
        ("ring", template_ring, bounds_ring),
        ("double_ring", template_double_ring, bounds_dr),
        ("lattice", template_lattice, bounds_lat),
        ("star", template_star, bounds_star)
    ]

    for name, t_func, bounds in templates:
        if time.time() > deadline:
            break
            
        # Global Optimization with Differential Evolution
        try:
            res_de = differential_evolution(
                lambda p: objective_template(t_func, p),
                bounds,
                maxiter=100,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False # We will polish manually
            )
            
            if res_de.fun < best_L:
                # Polish with L-BFGS-B
                res_lbfgs = minimize(
                    lambda p: objective_template(t_func, p),
                    res_de.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                
                if res_lbfgs.fun < best_L:
                    best_L = res_lbfgs.fun
                    best_centers, best_angles = t_func(res_lbfgs.x)
                    
        except Exception:
            continue

    # Fallback if no template worked or all failed
    if best_centers is None:
        # Use a known valid configuration: simple ring with large radius
        r = 3.0
        best_centers = []
        best_angles = []
        for i in range(N):
            theta = 2 * math.pi * i / N
            best_centers.append((r * math.cos(theta), r * math.sin(theta)))
            best_angles.append(theta)
        best_L = calculate_L(best_centers, best_angles, 0.0)

    # Final Verification
    # Even with templates, numerical errors or bad local minima in the penalty might cause overlap.
    # We MUST verify.
    if check_overlap_fast(best_centers, best_angles):
        # If invalid, try to fix by expanding the configuration slightly
        # or fall back to a very safe configuration.
        # Safe config: Grid with large spacing
        safe_centers = []
        safe_angles = []
        # 4x3 grid
        for i in range(4):
            for j in range(3):
                cx = (i - 1.5) * 2.2
                cy = (j - 1.0) * 2.2
                safe_centers.append((cx, cy))
                safe_angles.append(0.0)
        best_centers = safe_centers
        best_angles = safe_angles
        best_L = calculate_L(best_centers, best_angles, 0.0)

    # Calculate final L
    final_L = calculate_L(best_centers, best_angles, 0.0)
    
    # Construct return value
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])
        
    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
