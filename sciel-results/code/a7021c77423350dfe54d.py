"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        angles = angle_rad + math.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * math.sqrt(3) / 2
        normal_angles = math.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside_outer(inner, oc, os, oa) and pairwise_disjoint(inner)

    def compute_minimal_outer(inner_positions, outer_angle_deg):
        """Compute minimal side length of outer hexagon centered at origin to contain all inner hexagons."""
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        angle_rad = math.radians(outer_angle_deg)
        apothem_factor = math.sqrt(3) / 2
        
        normal_angles = angle_rad + math.radians(np.arange(30, 360, 60))
        
        max_apothem_needed = 0
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dists = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dist = np.max(dists)
            if max_dist > 0:
                needed_side = max_dist / apothem_factor
                max_apothem_needed = max(max_apothem_needed, needed_side)
        
        return max_apothem_needed

    def make_hex_cluster():
        """Create a dense honeycomb cluster of 12 hexagons."""
        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        
        # First ring: 6 hexagons
        for k in range(6):
            angle = math.radians(60 * k)
            x = 2.0 * math.cos(angle)
            y = 2.0 * math.sin(angle)
            positions.append((x, y, 0.0))
        
        # Second ring: pick 5 from 10 candidates to make 12 total
        sqrt3 = math.sqrt(3)
        second_ring_candidates = [
            (4.0, 0.0),
            (3.0, sqrt3),
            (1.0, 2*sqrt3),
            (-1.0, 2*sqrt3),
            (-3.0, sqrt3),
            (-4.0, 0.0),
            (-3.0, -sqrt3),
            (-1.0, -2*sqrt3),
            (1.0, -2*sqrt3),
            (3.0, -sqrt3),
        ]
        
        # For a symmetric cluster, pick 5 that maintain balance
        # Indices 0, 1, 2, 6, 7 give a nice symmetric shape
        best_combo = [0, 1, 2, 6, 7]
        
        for idx in best_combo:
            x, y = second_ring_candidates[idx]
            positions.append((x, y, 0.0))
        
        return positions

    def make_staggered_grid():
        """Create a staggered grid arrangement."""
        positions = []
        sqrt3 = math.sqrt(3)
        # 4 columns, 3 rows staggered
        for row in range(3):
            y = row * 3.0 - 3.0
            for col in range(4):
                x = col * 2.0 - 3.0
                if row % 2 == 1:
                    x += 1.0
                positions.append((x, y, 0.0))
        return positions

    def make_circular():
        """Create a circular arrangement."""
        positions = []
        positions.append((0.0, 0.0, 0.0))
        for k in range(11):
            angle = math.radians(360 * k / 11)
            r = 3.5
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append((x, y, 0.0))
        return positions

    def objective_smooth(params):
        """
        Smooth objective using support functions.
        params: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, ox,oy,os,oa]
        Returns: os + penalty terms (smooth approximations)
        """
        n_inner = N
        inner_pos = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_pos.append((x, y, a))
        
        ox = params[3*n_inner]
        oy = params[3*n_inner+1]
        os = max(params[3*n_inner+2], 0.1)
        oa = params[3*n_inner+3]
        
        outer_center = (ox, oy)
        outer_side = os
        outer_angle = oa
        
        # Compute all vertices
        all_verts = []
        for cx, cy, angle in inner_pos:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Containment penalty: for each face normal of outer hex, 
        # compute max projection of inner vertices, compare to required apothem
        angle_rad = math.radians(outer_angle)
        apothem_factor = math.sqrt(3) / 2
        normal_angles = angle_rad + math.radians(np.arange(30, 360, 60))
        
        containment_penalty = 0.0
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dists = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dist = np.max(dists)
            required_apothem = outer_side * apothem_factor
            if max_dist > required_apothem:
                excess = max_dist - required_apothem
                containment_penalty += excess * excess
        
        # Disjointness penalty: for each pair, compute minimum separation
        # using SAT - approximate with smooth minimum
        disjoint_penalty = 0.0
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                cx1, cy1, a1 = inner_pos[i]
                cx2, cy2, a2 = inner_pos[j]
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                
                # Collect axes
                axes = []
                for k in range(6):
                    edge1 = v1[(k+1)%6] - v1[k]
                    n1 = np.array([-edge1[1], edge1[0]])
                    norm1 = np.linalg.norm(n1)
                    if norm1 > 1e-9:
                        axes.append(n1 / norm1)
                    edge2 = v2[(k+1)%6] - v2[k]
                    n2 = np.array([-edge2[1], edge2[0]])
                    norm2 = np.linalg.norm(n2)
                    if norm2 > 1e-9:
                        axes.append(n2 / norm2)
                
                min_gap = float('inf')
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    gap = min(np.max(proj2) - np.min(proj1), np.max(proj1) - np.min(proj2))
                    min_gap = min(min_gap, gap)
                
                if min_gap < 0:
                    # Overlapping: penalize the negative gap
                    overlap = -min_gap
                    disjoint_penalty += overlap * overlap
        
        return outer_side + 10.0 * (containment_penalty + disjoint_penalty)

    def check_validity(inner, oc, os, oa):
        return is_valid(inner, oc, os, oa)

    # Generate initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal cluster
    hex_positions = make_hex_cluster()
    initial_configs.append(hex_positions)
    
    # Config 2: Staggered grid
    staggered = make_staggered_grid()
    initial_configs.append(staggered)
    
    # Config 3: Circular
    circular = make_circular()
    initial_configs.append(circular)
    
    # Find best initial configuration
    best_params = None
    best_score = float('inf')
    best_inner_init = None
    best_oc_init = None
    best_os_init = None
    best_oa_init = None
    
    for inner_positions in initial_configs:
        for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = compute_minimal_outer(inner_positions, oa)
            oc = (0.0, 0.0)
            
            # Ensure validity by expanding if needed
            os_try = side
            while not all_inside_outer(inner_positions, oc, os_try, oa):
                os_try += 0.05
                if os_try > 15:
                    break
            
            params = []
            for pos in inner_positions:
                params.extend([pos[0], pos[1], pos[2]])
            params.extend([oc[0], oc[1], os_try, oa])
            params = np.array(params, dtype=float)
            
            score = objective_smooth(params)
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
                best_inner_init = inner_positions
                best_oc_init = oc
                best_os_init = os_try
                best_oa_init = oa
    
    if best_params is None:
        # Fallback to hex cluster
        inner_init = make_hex_cluster()
        best_inner_init = inner_init
        best_oc_init = (0.0, 0.0)
        best_os_init = 6.0
        best_oa_init = 30.0
        params = []
        for pos in best_inner_init:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([best_oc_init[0], best_oc_init[1], best_os_init, best_oa_init])
        best_params = np.array(params, dtype=float)
        best_score = objective_smooth(best_params)
    
    # Optimization with L-BFGS-B
    current_params = best_params.copy()
    current_score = best_score
    
    # First optimization pass
    if time.time() < deadline - 20:
        try:
            res = minimize(objective_smooth, current_params, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4})
            if res.fun < current_score:
                current_score = res.fun
                current_params = res.x.copy()
        except Exception:
            pass
    
    # Second optimization pass with tighter tolerance
    if time.time() < deadline - 10 and current_score < 5.0:
        try:
            res2 = minimize(objective_smooth, current_params, method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6})
            if res2.fun < current_score:
                current_score = res2.fun
                current_params = res2.x.copy()
        except Exception:
            pass
    
    # Third optimization pass if time allows
    if time.time() < deadline - 5 and current_score < 4.5:
        try:
            res3 = minimize(objective_smooth, current_params, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res3.fun < current_score:
                current_score = res3.fun
                current_params = res3.x.copy()
        except Exception:
            pass
    
    # Parse final parameters
    final_params = current_params
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        a = final_params[3*i+2]
        inner.append((x, y, a))
    oc = (final_params[3*N], final_params[3*N+1])
    os = final_params[3*N+2]
    oa = final_params[3*N+3]
    
    # Validate and fallback if needed
    if not check_validity(inner, oc, os, oa):
        inner = best_inner_init
        oc = best_oc_init
        os = best_os_init
        oa = best_oa_init
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.05
            if os > 10:
                break
    
    return inner, oc, os, oa
# EVOLVE-BLOCK-END

