import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
        start_time = time.time()
        deadline = start_time + 590
        
        SQRT3 = np.sqrt(3.0)
        hex_angle_offsets = np.radians(60 * np.arange(6))
        
        def hex_vertices(cx, cy, s, a_rad):
            angles = a_rad + hex_angle_offsets
            return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
            
        def all_hex_vertices(positions, angles):
            n = len(positions)
            verts = np.zeros((n, 6, 2))
            for i in range(n):
                verts[i] = hex_vertices(positions[i, 0], positions[i, 1], 1.0, np.radians(angles[i]))
            return verts

        def min_separation(v1, v2):
            """
            Returns the minimum gap between two convex polygons v1 and v2.
            If they overlap, returns a negative value (penetration depth).
            If they are separated, returns the distance.
            """
            min_gap = np.inf
            
            # Check normals of v1
            for i in range(6):
                e1 = v1[(i+1) % 6] - v1[i]
                n1 = np.array([-e1[1], e1[0]])
                norm = np.linalg.norm(n1)
                if norm > 1e-15:
                    n1 = n1 / norm
                proj1 = v1 @ n1
                proj2 = v2 @ n1
                # The gap along this normal is min(proj1) - max(proj2) if v1 is 'left' of v2
                # or min(proj2) - max(proj1) if v2 is 'left' of v1
                # We want the smallest positive gap or largest negative overlap
                gap = min(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                if gap < min_gap:
                    min_gap = gap
                    # Early exit if we find a large overlap? No, we need the max overlap (most negative)
                    # Actually, for penalty, we care about the maximum penetration.
                    # If gap is negative, it's an overlap. We want the most negative gap (deepest penetration).
                    # So we should track min_gap as the minimum value found.
            
            # Check normals of v2
            for i in range(6):
                e2 = v2[(i+1) % 6] - v2[i]
                n2 = np.array([-e2[1], e2[0]])
                norm = np.linalg.norm(n2)
                if norm > 1e-15:
                    n2 = n2 / norm
                proj1 = v1 @ n2
                proj2 = v2 @ n2
                gap = min(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                if gap < min_gap:
                    min_gap = gap
            
            return min_gap

        def enclosing_hex_side(positions, angles, outer_center, outer_angle_deg):
            ocx, ocy = outer_center
            oa = np.radians(outer_angle_deg)
            
            all_verts = []
            for i in range(len(positions)):
                cx, cy = positions[i]
                a_rad = np.radians(angles[i])
                v = hex_vertices(cx, cy, 1.0, a_rad)
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            all_verts_shifted = all_verts - np.array([ocx, ocy])
            
            min_s = 0.0
            for i in range(6):
                a_i = oa + np.radians(60 * i)
                a_ip1 = oa + np.radians(60 * (i + 1))
                e_unit = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
                crosses = all_verts_shifted[:, 0] * e_unit[1] - all_verts_shifted[:, 1] * e_unit[0]
                min_cross = np.min(crosses)
                required_s = -2 * min_cross / SQRT3
                if required_s > min_s:
                    min_s = required_s
            return max(min_s, 0.0)

        def objective_symmetric(params):
            if time.time() > deadline:
                return 1e10
            if not np.all(np.isfinite(params)):
                return 1e10
            
            # Params: 5 pairs * (x, y, angle) + outer_angle
            # Center is fixed at (0,0)
            # Outer center fixed at (0,0)
            
            pairs = params[:15].reshape(5, 3)
            outer_angle = params[15]
            
            # Construct 11 hexagons
            positions = np.zeros((11, 2))
            angles = np.zeros(11)
            
            # Hex 0: Center
            positions[0] = [0.0, 0.0]
            angles[0] = 0.0
            
            idx = 1
            for i in range(5):
                x, y, a = pairs[i]
                # Hex i+1
                positions[idx] = [x, y]
                angles[idx] = a
                # Hex i+2 (opposite)
                positions[idx+1] = [-x, -y]
                angles[idx+1] = a + 180.0
                idx += 2
                
            # Compute side
            s = enclosing_hex_side(positions, angles, [0.0, 0.0], outer_angle)
            
            # Check overlaps
            penalty = 0.0
            verts = all_hex_vertices(positions, angles)
            for i in range(11):
                for j in range(i+1, 11):
                    # Quick distance check to avoid expensive SAT for far apart hexes
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < 4.0: # Threshold for potential overlap
                        gap = min_separation(verts[i], verts[j])
                        if gap < 1e-6:
                            # Penalty for overlap or touching
                            # gap is negative if overlapping
                            penalty += 100.0 * (1e-6 - gap) ** 2 + 10.0 * (1e-6 - gap)
                                
            if s < np.sqrt(11):
                s = np.sqrt(11)
            
            return s + penalty

        # Initial guess for symmetric case
        # A known valid configuration:
        # Center at (0,0)
        # 6 around it at distance 2 (tangent)
        # 4 in the "corners" of the hexagonal lattice
        # Lattice vectors: (2,0), (1, sqrt3), (-1, sqrt3), etc.
        # Let's place them:
        # 1: (2, 0)
        # 2: (1, sqrt3)
        # 3: (-1, sqrt3)
        # 4: (3, sqrt3) -> Wait, (3, sqrt3) is (2,0) + (1, sqrt3). This is a corner.
        # 5: (2, 2*sqrt3) -> Wait, (1, sqrt3) + (1, sqrt3)? No.
        # Let's use standard hex grid coordinates.
        # (0,0)
        # (2,0), (1,s3), (-1,s3), (-2,0), (-1,-s3), (1,-s3)
        # (3,s3), (0,2s3), (-3,s3), (-3,-s3), (0,-2s3), (3,-s3)
        # We have 1 center + 6 ring + 4 corners? No, 1+6+6=13. We need 11.
        # So we take 1 center, 6 ring, and 4 of the 6 outer corners.
        # Symmetric choice: (3,s3), (0,2s3), (-3,s3), (-3,-s3) -> Not symmetric under 180 rotation?
        # (3,s3) opposite is (-3,-s3).
        # (0,2s3) opposite is (0,-2s3).
        # So pairs:
        # 1: (3, s3)
        # 2: (0, 2s3)
        # 3: (2, 0) -> Wait, (2,0) is in the ring.
        # Let's stick to the ring + 4 corners.
        # Ring: (2,0), (1,s3), (-1,s3), (-2,0), (-1,-s3), (1,-s3)
        # Corners: (3,s3), (0,2s3), (-3,s3), (-3,-s3), (0,-2s3), (3,-s3)
        # We need 5 pairs.
        # Pair 1: (2,0) [Ring]
        # Pair 2: (1,s3) [Ring]
        # Pair 3: (-1,s3) [Ring]
        # Pair 4: (3,s3) [Corner]
        # Pair 5: (0,2s3) [Corner]
        # This gives:
        # (2,0), (-2,0)
        # (1,s3), (-1,-s3)
        # (-1,s3), (1,-s3)
        # (3,s3), (-3,-s3)
        # (0,2s3), (0,-2s3)
        # Total 10 + 1 center = 11.
        
        x0_sym = np.array([
            2.0, 0.0, 0.0,
            1.0, SQRT3, 0.0,
            -1.0, SQRT3, 0.0,
            3.0, SQRT3, 0.0,
            0.0, 2.0*SQRT3, 0.0,
            0.0 # outer angle
        ])
        
        best_params_sym = x0_sym.copy()
        best_val_sym = objective_symmetric(x0_sym)
        
        # Optimize symmetric case
        try:
            res_sym = minimize(objective_symmetric, x0_sym, method='Powell', 
                               options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12})
            if res_sym.fun < best_val_sym:
                best_val_sym = res_sym.fun
                best_params_sym = res_sym.x.copy()
        except Exception:
            pass
            
        # Refine with Nelder-Mead
        try:
            res_sym2 = minimize(objective_symmetric, best_params_sym, method='Nelder-Mead',
                                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-13})
            if res_sym2.fun < best_val_sym:
                best_val_sym = res_sym2.fun
                best_params_sym = res_sym2.x.copy()
        except Exception:
            pass

        # Convert symmetric solution to full 11-hex representation
        pairs = best_params_sym[:15].reshape(5, 3)
        outer_angle = best_params_sym[15]
        
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        angles[0] = 0.0
        idx = 1
        for i in range(5):
            x, y, a = pairs[i]
            positions[idx] = [x, y]
            angles[idx] = a
            positions[idx+1] = [-x, -y]
            angles[idx+1] = a + 180.0
            idx += 2
            
        inner_params_full = np.column_stack([positions, angles])
        outer_params_full = np.array([0.0, 0.0, outer_angle])
        x0_full = np.concatenate([inner_params_full.flatten(), outer_params_full])
        
        # Now run full optimization from this start to allow symmetry breaking
        def objective_full(params):
            if time.time() > deadline:
                return 1e10
            if not np.all(np.isfinite(params)):
                return 1e10
            
            inner_params = params[:33].reshape(11, 3)
            outer_params = params[33:36]
            
            inner_positions = inner_params[:, :2]
            inner_angles = inner_params[:, 2]
            outer_center = outer_params[:2]
            outer_angle = outer_params[2]
            
            s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
            
            penalty = 0.0
            verts = all_hex_vertices(inner_positions, inner_angles)
            for i in range(11):
                for j in range(i+1, 11):
                    dist = np.linalg.norm(inner_positions[i] - inner_positions[j])
                    if dist < 4.0:
                        gap = min_separation(verts[i], verts[j])
                        if gap < 1e-6:
                            penalty += 100.0 * (1e-6 - gap) ** 2 + 10.0 * (1e-6 - gap)
                                
            if s < np.sqrt(11):
                s = np.sqrt(11)
            return s + penalty

        best_params_full = x0_full.copy()
        best_val_full = objective_full(x0_full)
        
        # Multi-start full optimization
        np.random.seed(42)
        for i in range(5):
            if time.time() > deadline - 60:
                break
            if i == 0:
                x_start = x0_full.copy()
            else:
                x_start = x0_full.copy()
                # Perturb
                for k in range(11):
                    x_start[3*k] += np.random.randn() * 0.5
                    x_start[3*k+1] += np.random.randn() * 0.5
                    x_start[3*k+2] += np.random.randn() * 10
                x_start[33] += np.random.randn() * 0.5
                x_start[34] += np.random.randn() * 0.5
                x_start[35] += np.random.randn() * 10
                
            try:
                res_full = minimize(objective_full, x_start, method='Powell', 
                                    options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-12})
                if res_full.fun < best_val_full:
                    best_val_full = res_full.fun
                    best_params_full = res_full.x.copy()
            except Exception:
                pass
                
        # Final polish
        if time.time() < deadline - 10:
            try:
                res_final = minimize(objective_full, best_params_full, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res_final.fun < best_val_full:
                    best_val_full = res_final.fun
                    best_params_full = res_final.x.copy()
            except Exception:
                pass

        # Extract final result
        inner_params = best_params_full[:33].reshape(11, 3)
        outer_params = best_params_full[33:36]
        
        inner_positions = inner_params[:, :2]
        inner_angles = inner_params[:, 2]
        outer_center = outer_params[:2]
        outer_angle = outer_params[2]
        
        s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
        s = max(s, np.sqrt(11))
        
        inner_hex_data = np.column_stack([inner_positions, inner_angles])
        outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
        
        return inner_hex_data, outer_hex_data, float(s)
