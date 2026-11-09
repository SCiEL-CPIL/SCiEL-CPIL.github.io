import numpy as np
from scipy.optimize import minimize
import time

N = 11
_SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Unit hexagon vertices (flat-top, centered at origin)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    def unit_hex_verts(angle_rad=0.0):
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + angle_rad
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def hex_verts(cx, cy, angle_deg):
        angle_rad = angle_deg * np.pi / 180.0
        v = unit_hex_verts(angle_rad)
        return v + np.array([cx, cy])
    
    def hex_containment_dist(verts, outer_center, outer_side, outer_angle_deg):
        """Minimum signed distance from any vertex to outer hexagon boundary.
        Positive means inside, negative means outside."""
        outer_angle_rad = outer_angle_deg * np.pi / 180.0
        outer_verts = outer_side * unit_hex_verts(outer_angle_rad) + np.array(outer_center)
        
        min_dist = float('inf')
        for i in range(6):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            center_to_v1 = np.array(outer_center) - v1
            if np.dot(normal, center_to_v1) < 0:
                normal = -normal
            for v in verts:
                d = np.dot(v - v1, normal)
                min_dist = min(min_dist, d)
        return min_dist
    
    def fast_separation(verts1, verts2):
        """Fast SAT-based minimum separation between two convex hexagons.
        Returns positive if separated, negative if overlapping."""
        # For two convex hexagons, SAT with 12 axes
        # But we can early-terminate: if center distance > 2*inradius, they're separated
        c1 = np.mean(verts1, axis=0)
        c2 = np.mean(verts2, axis=0)
        cd = np.linalg.norm(c1 - c2)
        inradius = _SQRT3 / 2.0
        if cd >= 2.0 * inradius:
            return cd - 2.0 * inradius
        
        # Full SAT check
        all_edges = []
        for verts in (verts1, verts2):
            for i in range(6):
                edge = verts[(i + 1) % 6] - verts[i]
                all_edges.append(edge)
        
        min_sep = -float('inf')
        for edge in all_edges:
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                continue
            axis = axis / norm
            proj1 = np.dot(verts1, axis)
            proj2 = np.dot(verts2, axis)
            sep = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
            if sep > min_sep:
                min_sep = sep
        return min_sep
    
    # Initial compact arrangement: honeycomb-like
    # Place hexagons in rows with offsets
    def initial_positions():
        positions = []
        angles = []
        # Try a 3-4-4 arrangement or similar compact layout
        # For 11 hexagons, try: row of 4, row of 4, row of 3
        # Or better: use a hexagonal packing pattern
        
        # Flat-top hexagons: width = 2*sqrt(3), height = 2
        # For flat-top, adjacent in same row: dx = 2*sqrt(3), dy = 0
        # Adjacent in next row: dx = sqrt(3), dy = 1.5 (for pointy-top)
        # For flat-top: adjacent row offset: dx = 0, dy = 2, but with horizontal shift
        
        # Actually for flat-top hexagons:
        # Width (horizontal) = 2*sqrt(3) ≈ 3.464
        # Height (vertical) = 2
        # To pack, offset rows by sqrt(3) horizontally and 1.5 vertically? No...
        
        # Let me think: flat-top hexagon has vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
        # So width = 2, height = sqrt(3)
        # Wait, let me recalculate. With my unit_hex_verts:
        # angles 0,60,120,180,240,300
        # (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5,-0.866), (0.5,-0.866)
        # So width = 2 (from x=-1 to x=1), height = sqrt(3) ≈ 1.732
        
        # For packing flat-top hexagons:
        # Same row: shift by 2 in x (since width is 2, but they interlock)
        # Actually adjacent flat-top hexagons share an edge, so center-to-center distance for horizontal neighbors = 2*sqrt(3)/2 * ... 
        
        # Let me just use a grid and let the optimizer fix it
        # Start with a reasonable guess
        
        # Place in a 3x4 grid roughly
        sqrt3 = np.sqrt(3)
        # For flat-top hexagons packed in honeycomb:
        # Row 0: x = 0, 3.464, 6.928, 10.392 (4 hexagons)
        # Row 1: x = 1.732, 5.196, 8.660 (3 hexagons), y offset = 1.5
        # Row 2: x = 0, 3.464, 6.928 (3 hexagons), y offset = 3.0
        # Total: 4+3+4 = 11
        
        cols = 4
        rows = 3
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                x = c * 2 * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3  # offset every other row
                positions.append([x, y])
                angles.append(0.0)
                idx += 1
            if idx >= N:
                break
        
        # Center the arrangement
        if positions:
            px = np.array([p[0] for p in positions])
            py = np.array([p[1] for p in positions])
            cx = (px.max() + px.min()) / 2
            cy = (py.max() + py.min()) / 2
            for i in range(len(positions)):
                positions[i][0] -= cx
                positions[i][1] -= cy
        
        return positions, angles
    
    positions, angles = initial_positions()
    
    # Variables: for each hex: x, y, angle_deg, plus outer_side, outer_angle_deg
    # Total: N*3 + 2 = 35 variables
    n_vars = N * 3 + 2
    
    def unpack(x):
        pos = []
        ang = []
        for i in range(N):
            pos.append([x[3*i], x[3*i+1]])
            ang.append(x[3*i+2])
        outer_side = x[N*3]
        outer_angle = x[N*3+1]
        return pos, ang, outer_side, outer_angle
    
    def objective(x):
        pos, ang, outer_side, outer_angle = unpack(x)
        return outer_side
    
    def constraints(x):
        pos, ang, outer_side, outer_angle = unpack(x)
        
        cons = []
        
        # Containment: each inner hex must be inside outer hex
        for i in range(N):
            def containment_constr(x, i=i):
                pos, ang, os, oa = unpack(x)
                vi = hex_verts(pos[i][0], pos[i][1], ang[i])
                return hex_containment_dist(vi, (0.0, 0.0), os, oa)
            
            cons.append({'type': 'ineq', 'fun': containment_constr})
        
        # Non-overlap: pairs of inner hexagons must not overlap
        for i in range(N):
            for j in range(i+1, N):
                def nonoverlap_constr(x, i=i, j=j):
                    pos, ang, os, oa = unpack(x)
                    vi = hex_verts(pos[i][0], pos[i][1], ang[i])
                    vj = hex_verts(pos[j][0], pos[j][1], ang[j])
                    return fast_separation(vi, vj)
                
                cons.append({'type': 'ineq', 'fun': nonoverlap_constr})
        
        return cons
    
    # Bounds: outer_side > 0, angles in [-180, 180]
    bounds = []
    for i in range(N):
        bounds.append((-50, 50))  # x
        bounds.append((-50, 50))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((0.5, 50))  # outer_side
    bounds.append((-180, 180))  # outer_angle
    
    # Initial point
    x0 = []
    for i in range(N):
        x0.append(positions[i][0])
        x0.append(positions[i][1])
        x0.append(angles[i])
    # Estimate initial outer side
    max_extent = 0
    for p in positions:
        max_extent = max(max_extent, abs(p[0]), abs(p[1]))
    x0.append(max_extent + 3)  # outer_side
    x0.append(0.0)  # outer_angle
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple restarts with different initial configurations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Perturb initial positions for diversity
        if restart > 0:
            x0_perturbed = list(x0)
            for i in range(N):
                x0_perturbed[3*i] += np.random.uniform(-3, 3)
                x0_perturbed[3*i+1] += np.random.uniform(-3, 3)
                x0_perturbed[3*i+2] += np.random.uniform(-60, 60)
            x0_perturbed[N*3] = max_extent + 2 + np.random.uniform(-1, 3)
        else:
            x0_perturbed = x0
        
        try:
            # Use trf method which handles inequality constraints efficiently
            result = minimize(objective, x0_perturbed, method='trf',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 80, 'gtol': 1e-4})
            
            if result.fun < best_side:
                pos, ang, outer_side, outer_angle = unpack(result.x)
                # Verify validity
                valid = True
                for i in range(N):
                    vi = hex_verts(pos[i][0], pos[i][1], ang[i])
                    if hex_containment_dist(vi, (0.0, 0.0), outer_side, outer_angle) < -1e-5:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            vi = hex_verts(pos[i][0], pos[i][1], ang[i])
                            vj = hex_verts(pos[j][0], pos[j][1], ang[j])
                            if fast_separation(vi, vj) < -1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_result = result.x
                        best_side = outer_side
        except Exception as e:
            pass
        
        if time.time() > deadline - 10:
            break
    
    if best_result is not None:
        pos, ang, outer_side, outer_angle = unpack(best_result)
        inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
        return inner, (0.0, 0.0), outer_side, outer_angle
    else:
        # Fallback: return the initial arrangement with a safe outer side
        inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
        return inner, (0.0, 0.0), max_extent + 5.0, 0.0

# EVOLVE-BLOCK-END
