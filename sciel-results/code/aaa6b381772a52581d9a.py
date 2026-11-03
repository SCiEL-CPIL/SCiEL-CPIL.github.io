"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        # 12 axes: 6 from v1, 6 from v2
        axes = []
        for i in range(6):
            edge1 = v1[(i+1) % 6] - v1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            
            edge2 = v2[(i+1) % 6] - v2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True

    def check_all_overlaps(positions, angles):
        """Check if any pair of hexagons overlaps."""
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return True
        return False

    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)
            
        if not all_verts:
            return 0.0
            
        all_verts = np.vstack(all_verts)
        
        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        
        # For a regular hexagon, side = 2 * max_proj / sqrt(3)
        side = 2 * max_proj / math.sqrt(3)
        return side

    def get_config_score(positions, angles, outer_angle):
        """Return side length if valid, else infinity."""
        if check_all_overlaps(positions, angles):
            return float('inf')
        return compute_min_side(positions, angles, outer_angle)

    # === NEW: Structural Honeycomb Generation ===
    
    def axial_to_cartesian(q, r):
        """Convert axial coordinates to Cartesian for a hex grid with side length 1."""
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def generate_honeycomb_positions():
        """
        Generate the 12 positions for the optimal honeycomb packing.
        """
        
        topologies = []
        
        def make_grid(rows, cols):
            coords = []
            for r in range(rows):
                for c in range(cols):
                    x = math.sqrt(3) * (c + (r % 2) * 0.5)
                    y = 1.5 * r
                    coords.append((x, y))
            return coords

        # Topology 2: 3 rows of 4
        topologies.append(make_grid(3, 4))
        
        # Topology 3: 4 rows of 3
        topologies.append(make_grid(4, 3))
        
        # Topology 4: 2 rows of 6
        topologies.append(make_grid(2, 6))
        
        # Topology 5: 6 rows of 2
        topologies.append(make_grid(6, 2))
        
        # Topology 6: "Flower" (1+6+5)
        # Center (0,0)
        # Neighbors: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Next 5: Let's pick (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) ?? 
        layout_f = [(0, 0)]
        ring1_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        layout_f.extend(ring1_dirs)
        ring2_candidates = []
        for (q, r) in ring1_dirs:
            for dq, dr in ring1_dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in layout_f and (nq, nr) not in ring2_candidates:
                    ring2_candidates.append((nq, nr))
        layout_f.extend(ring2_candidates[:5])
        
        # Convert axial to cartesian for layout_f
        coords_f = [axial_to_cartesian(q, r) for q, r in layout_f]
        topologies.append(coords_f)

        positions_list = []
        for coords in topologies:
            # Center the layout
            cx = sum(p[0] for p in coords) / len(coords)
            cy = sum(p[1] for p in coords) / len(coords)
            centered = [(x - cx, y - cy) for x, y in coords]
            positions_list.append(centered)
            
        return positions_list

    def optimize_positions(positions, angles, outer_angle):
        """
        Optimize positions and angles of inner hexagons to minimize outer side length.
        Uses differential evolution for global search, then local refinement.
        """
        # Flatten parameters: x1, y1, a1, ..., x12, y12, a12
        def objective(params):
            if time.time() > deadline:
                return 1e9
                
            pos = []
            ang = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3 + 1]
                a = params[i*3 + 2]
                pos.append((x, y))
                ang.append(a)
            
            if check_all_overlaps(pos, ang):
                return 1e9
            
            return compute_min_side(pos, ang, outer_angle)

        # Bounds: x, y in [-5, 5], angle in [0, 360]
        bounds = [(-5, 5), (-5, 5), (0, 360)] * N
        
        # Initial guess
        x0 = []
        for i in range(N):
            x0.append(positions[i][0])
            x0.append(positions[i][1])
            x0.append(angles[i])
            
        # Use differential evolution for global optimization
        # We restrict the time for this step
        start_opt = time.time()
        max_opt_time = 30.0
        
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                x0=x0,
                maxiter=100, 
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if result.fun < 1e8:
                best_params = result.x
            else:
                best_params = x0
                
        except Exception:
            best_params = x0
            
        # Local refinement using Nelder-Mead or BFGS if time permits
        if time.time() < deadline - 10:
            try:
                local_result = minimize(
                    objective, 
                    best_params, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if local_result.fun < objective(best_params):
                    best_params = local_result.x
            except Exception:
                pass

        # Extract best positions and angles
        best_pos = []
        best_ang = []
        for i in range(N):
            best_pos.append((best_params[i*3], best_params[i*3 + 1]))
            best_ang.append(best_params[i*3 + 2])
            
        # Check validity and compute score
        if check_all_overlaps(best_pos, best_ang):
            return positions, angles, float('inf')
            
        score = compute_min_side(best_pos, best_ang, outer_angle)
        return best_pos, best_ang, score

    # Main Logic
    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    # Generate candidate topologies
    candidate_positions = generate_honeycomb_positions()
    
    # We will optimize each topology with a few different outer angles
    # The outer angle affects the objective function shape, so we try a few
    # 0, 30, 15, 45 degrees
    outer_angles_to_try = [0.0, 30.0, 15.0, 45.0]
    
    for pos in candidate_positions:
        angles = [0.0] * N
        
        # Quick check: is this topology valid (no overlaps)?
        if check_all_overlaps(pos, angles):
            continue
            
        for oa in outer_angles_to_try:
            if time.time() > deadline - 20:
                break
                
            # Optimize positions for this outer angle
            opt_pos, opt_ang, score = optimize_positions(pos, angles, oa)
            
            if score < best_score:
                best_score = score
                best_positions = opt_pos
                best_angles = opt_ang
                best_outer_angle = oa

    # If no valid configuration was found, fallback
    if best_positions is None or best_score == float('inf'):
        positions = []
        for i in range(N):
            r = i // 4
            c = i % 4
            positions.append((c * 4.0, r * 4.0))
        best_positions = positions
        best_angles = [0.0] * N
        best_outer_angle = 0.0
        best_score = compute_min_side(positions, best_angles, 0.0)

    # Final cleanup
    # Ensure positions are centered at (0,0) for the return value
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0,0)] * N
        best_angles = [0] * N
        best_score = 10.0

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
