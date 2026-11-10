# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fsolve

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _solve_topology(topology_type):
    """
    Solves for positions of 11 hexagons given a specific contact topology.
    Uses fsolve to satisfy distance constraints (d=2) and symmetry.
    """
    if topology_type == "symmetric_6_4":
        # 1 center, 6 ring (dist 2 from center), 4 outer (dist 2 from ring)
        # Symmetry: 6-fold for ring, 4 outer at specific angles.
        # Variables: ring_angle_offset (0), outer_radius, outer_angle_offset
        # Actually, let's just solve for the 4 outer hexagon positions (8 vars)
        # with constraints: dist(outer_k, ring_j) = 2 for specific contacts.
        
        # Initial guess:
        # Center at (0,0)
        # Ring at radius 2, angles 30, 90, 150, 210, 270, 330
        # Outer at radius ~3.5, angles 0, 60, 120, 180 (or similar)
        
        def equations(vars):
            # vars: [x1, y1, x2, y2, x3, y3, x4, y4] for 4 outer hexagons
            # Outer hexagons are at angles 0, 60, 120, 180 relative to ring?
            # Let's assume outer hexagons are placed in the "valleys" between ring hexagons.
            # Ring hexagons are at angles 30, 90, 150, 210, 270, 330.
            # Valleys are at 0, 60, 120, 180, 240, 300.
            # We only have 4 outer hexagons. Let's place them at 0, 120, 240, 300? Or 0, 60, 120, 180?
            # To maintain some symmetry, let's try placing them at 0, 120, 240, 300 is 4-fold? No.
            # Let's try 0, 60, 120, 180. This breaks 6-fold symmetry but keeps some order.
            
            # Better: Use the fact that the outer hexagons must touch 2 ring hexagons each?
            # If an outer hexagon is in the valley between ring hexagon k and k+1, it touches both.
            # Let's assume the 4 outer hexagons are in 4 of the 6 valleys.
            # Which 4? To minimize the bounding box, we should probably leave the two opposite valleys empty?
            # Or leave adjacent ones empty?
            # Let's try leaving valleys at 240 and 300 empty, so outer hexagons are at 0, 60, 120, 180.
            
            # Ring positions:
            ring_pos = []
            for k in range(6):
                a = np.radians(30 + 60 * k)
                ring_pos.append([2 * np.cos(a), 2 * np.sin(a)])
            
            # Outer positions to solve for:
            # Valley 0 (angle 0): touches ring 0 (30 deg) and ring 5 (330 deg)
            # Valley 1 (angle 60): touches ring 1 (90 deg) and ring 0 (30 deg)
            # Valley 2 (angle 120): touches ring 2 (150 deg) and ring 1 (90 deg)
            # Valley 3 (angle 180): touches ring 3 (210 deg) and ring 2 (150 deg)
            
            # Equations:
            # dist(outer_0, ring_0) = 2
            # dist(outer_0, ring_5) = 2
            # dist(outer_1, ring_1) = 2
            # dist(outer_1, ring_0) = 2
            # dist(outer_2, ring_2) = 2
            # dist(outer_2, ring_1) = 2
            # dist(outer_3, ring_3) = 2
            # dist(outer_3, ring_2) = 2
            
            eqs = []
            # Outer 0 (valley 0)
            eqs.append(np.sum((vars[0:2] - np.array(ring_pos[0]))**2) - 4)
            eqs.append(np.sum((vars[0:2] - np.array(ring_pos[5]))**2) - 4)
            # Outer 1 (valley 60)
            eqs.append(np.sum((vars[2:4] - np.array(ring_pos[1]))**2) - 4)
            eqs.append(np.sum((vars[2:4] - np.array(ring_pos[0]))**2) - 4)
            # Outer 2 (valley 120)
            eqs.append(np.sum((vars[4:6] - np.array(ring_pos[2]))**2) - 4)
            eqs.append(np.sum((vars[4:6] - np.array(ring_pos[1]))**2) - 4)
            # Outer 3 (valley 180)
            eqs.append(np.sum((vars[6:8] - np.array(ring_pos[3]))**2) - 4)
            eqs.append(np.sum((vars[6:8] - np.array(ring_pos[2]))**2) - 4)
            
            return eqs
        
        # Initial guess for outer hexagons
        # Valley 0 is at angle 0. Distance from origin is approx 2 * cos(30) + 2 * cos(30) = 2*sqrt(3)?
        # Actually, the center of a hexagon touching two hexagons at distance 2 from origin...
        # Triangle formed by Center, Ring0, Outer0 is equilateral? No.
        # Center(0,0), Ring0(2, 30 deg). Outer0 touches Center? No, Outer0 touches Ring0 and Ring5.
        # Distance Center to Outer0?
        # Let's just guess radius 3.5 for outer hexagons.
        guess = []
        for k in range(4):
            a = np.radians(0 + 60 * k)
            guess.extend([3.5 * np.cos(a), 3.5 * np.sin(a)])
        
        try:
            sol, info, ier, msg = fsolve(equations, guess, full_output=True)
            if ier != 1:
                return None
            
            # Construct full positions
            positions = np.zeros((11, 2))
            angles = np.zeros(11)
            
            # Center
            positions[0] = [0, 0]
            
            # Ring
            for k in range(6):
                a = np.radians(30 + 60 * k)
                positions[k+1] = [2 * np.cos(a), 2 * np.sin(a)]
            
            # Outer
            for k in range(4):
                positions[7+k] = sol[2*k:2*k+2]
                
            # Check separation
            if not _check_separation_fast(positions, angles):
                return None
            
            return positions, angles
        
        except Exception:
            return None
            
    return None

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    # Phase 1: Solve specific topologies
    topologies = ["symmetric_6_4"]
    
    for topo in topologies:
        if time.time() > deadline - 120:
            break
            
        res = _solve_topology(topo)
        if res is not None:
            positions, angles = res
            side = _min_enclosing_side(positions, angles)
            
            # Refine with rotation
            def obj_rot(theta):
                # Rotate positions by theta
                c = np.cos(np.radians(theta))
                s = np.sin(np.radians(theta))
                rot_mat = np.array([[c, -s], [s, c]])
                pos_rot = positions @ rot_mat.T
                # Angles also rotate?
                # The problem says "angle_deg" is the orientation of the hexagon itself.
                # If we rotate the whole packing, the hexagon orientations should also rotate.
                angles_rot = (angles + theta) % 360
                return _min_enclosing_side(pos_rot, angles_rot)
            
            # Grid search for best rotation
            best_rot = 0
            best_rot_side = side
            for t in np.linspace(0, 60, 61):
                s_t = obj_rot(t)
                if s_t < best_rot_side:
                    best_rot_side = s_t
                    best_rot = t
            
            # Fine tune rotation
            try:
                res_rot = minimize(obj_rot, [best_rot], method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10})
                if res_rot.fun < best_rot_side:
                    best_rot = res_rot.x[0]
                    best_rot_side = res_rot.fun
            except:
                pass
            
            if best_rot_side < best_side:
                best_side = best_rot_side
                c = np.cos(np.radians(best_rot))
                s = np.sin(np.radians(best_rot))
                rot_mat = np.array([[c, -s], [s, c]])
                best_positions = positions @ rot_mat.T
                best_angles = (angles + best_rot) % 360
                
    # Fallback if solver failed
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2 * np.cos(a), 2 * np.sin(a)]
        # Place 4 outer hexagons roughly
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)
        
    # Phase 2: Local refinement of positions (asymmetric)
    if time.time() < deadline - 60:
        def objective_full(pos_flat):
            pos = pos_flat.reshape(11, 2)
            if not _check_separation_fast(pos, best_angles):
                return 1e10
            return _min_enclosing_side(pos, best_angles)
        
        try:
            res = minimize(
                objective_full,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_side and res.fun < 1e9:
                trial_pos = res.x.reshape(11, 2)
                if _check_separation_fast(trial_pos, best_angles):
                    side = _min_enclosing_side(trial_pos, best_angles)
                    if side < best_side:
                        best_side = side
                        best_positions = trial_pos
        except Exception:
            pass

    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback to a known valid configuration if separation failed
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2 * np.cos(a), 2 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001

    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
