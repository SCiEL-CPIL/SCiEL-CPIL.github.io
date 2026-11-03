import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at angles 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def is_valid(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    def generate_chain_structures():
        """
        Generate non-isomorphic chain structures on the hexagonal lattice.
        We use axial coordinates (q, r).
        Moves: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        We generate paths of length 12 and normalize them to remove symmetries.
        """
        moves = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        structures = set()
        paths = []
        
        def dfs(path, visited):
            if time.time() > deadline - 60: return # Safety
            if len(path) == N:
                # Normalize: translate so min q, min r is (0,0)
                min_q = min(p[0] for p in path)
                min_r = min(p[1] for p in path)
                norm_path = tuple((p[0]-min_q, p[1]-min_r) for p in path)
                # Also check reverse
                rev_path = tuple((p[0]-min_q, p[1]-min_r) for p in reversed(path))
                key = min(norm_path, rev_path)
                if key not in structures:
                    structures.add(key)
                    paths.append(list(key))
                return
            
            q, r = path[-1]
            for move in moves:
                nq, nr = q + move[0], r + move[1]
                if (nq, nr) not in visited:
                    visited.add((nq, nr))
                    path.append((nq, nr))
                    dfs(path, visited)
                    path.pop()
                    visited.remove((nq, nr))
        
        dfs([(0,0)], {(0,0)})
        return paths

    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = math.sqrt(3) / 2 * q + math.sqrt(3) * r
        return x, y

    # Generate structures
    structures = generate_chain_structures()
    
    best_score = float('inf')
    best_params = None
    
    # Pre-calculate cartesian coords for each structure
    struct_coords = []
    for struct in structures:
        coords = [axial_to_cartesian(q, r) for q, r in struct]
        # Center the structure
        cx = np.mean([c[0] for c in coords])
        cy = np.mean([c[1] for c in coords])
        centered = [(c[0]-cx, c[1]-cy) for c in coords]
        struct_coords.append(centered)

    # Optimization parameters for each structure:
    # x, y, theta (global rotation), scale (stretch factor for flexibility), outer_angle
    # We assume inner hexagons rotate with the global theta (rigid body)
    # 'scale' allows slight expansion/contraction to break lattice rigidity if needed, 
    # but primarily we rely on global rotation and translation.
    # Actually, let's allow a small individual angle perturbation? No, keep it rigid for speed.
    # Parameters: [x, y, theta, outer_angle]
    
    def objective(params, coords):
        x, y, theta_deg, outer_angle = params
        theta_rad = math.radians(theta_deg)
        
        # Rotate coords
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        
        positions = np.zeros((N, 2))
        for i, (px, py) in enumerate(coords):
            rx = px * cos_t - py * sin_t
            ry = px * sin_t + py * cos_t
            positions[i] = [rx + x, ry + y]
        
        # Angles are all theta_deg
        angles = np.full(N, theta_deg)
        
        # Check validity (rigid body, so if it was valid in lattice, it is valid now)
        # Wait, if we rotate, the relative positions are preserved. 
        # Lattice packing is valid. So no need to check disjointness if we start from a valid lattice path.
        # But we might move them? No, the lattice structure fixes relative positions.
        # The only degree of freedom is global transform.
        # So validity is guaranteed by the structure generation (if we ensure no self-intersection in lattice).
        
        return calc_min_outer_side(positions, angles, outer_angle)

    # We need to ensure the generated lattice paths are self-disjoint.
    # In a hex lattice, a simple path (no repeated nodes) of hexagons implies they are disjoint?
    # Yes, regular hexagons on a lattice touch only if they share an edge.
    # A path visits each node once. Adjacent nodes in the path share an edge.
    # Non-adjacent nodes in the path might be close?
    # In a hex lattice, the distance between non-adjacent nodes is at least 1 (in terms of hex distance).
    # Hexagons of side 1 have distance between centers = sqrt(3) for adjacent.
    # Wait, side length 1. Distance between centers of adjacent hexagons is sqrt(3).
    # So they touch.
    # If we take a path, all consecutive pairs touch.
    # Do non-consecutive pairs overlap?
    # In a simple path, non-consecutive nodes are not adjacent in the path.
    # Could they be adjacent in the lattice?
    # Yes. E.g., a "U" shape. (0,0) -> (1,0) -> (1,-1) -> (0,-1).
    # (0,0) and (0,-1) are not consecutive in the path, but are they adjacent in the lattice?
    # (0,0) neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Yes, (0,0) and (0,-1) are neighbors.
    # So a simple path in the graph can have "shortcuts" that cause overlap.
    # We must filter out structures that are not "induced paths" (no chords).
    
    valid_struct_coords = []
    for struct, coords in zip(structures, struct_coords):
        # Check for chords
        is_valid = True
        for i in range(N):
            for j in range(i+2, N):
                if i == j-1: continue
                q1, r1 = struct[i]
                q2, r2 = struct[j]
                # Check if adjacent in lattice
                # Distance in axial coords: max(|dq|, |dr|, |dq+dr|) == 1
                dq = q1 - q2
                dr = r1 - r2
                dist = max(abs(dq), abs(dr), abs(dq+dr))
                if dist == 1:
                    is_valid = False
                    break
            if not is_valid: break
        
        if is_valid:
            valid_struct_coords.append(coords)

    # If we didn't find any valid induced paths (unlikely), fallback to full random
    if not valid_struct_coords:
        print("No valid induced paths found, using random")
        np.random.seed(42)
        for _ in range(5):
            pos = np.random.uniform(-5, 5, (N, 2))
            ang = np.random.uniform(0, 360, N)
            # Just use the original SLSQP logic as fallback? 
            # For now, just skip to ensure we return something
            pass
        # Fallback to a known valid configuration (honeycomb subset)
        sqrt3 = math.sqrt(3)
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        
        # Try to optimize this one
        def obj_fallback(params):
            x, y, theta_deg, outer_angle = params
            theta_rad = math.radians(theta_deg)
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            positions = np.zeros((N, 2))
            for i in range(N):
                px, py = init_positions[i]
                rx = px * cos_t - py * sin_t
                ry = px * sin_t + py * cos_t
                positions[i] = [rx + x, ry + y]
            return calc_min_outer_side(positions, init_angles, outer_angle)
        
        x0 = [0, 0, 0, 0]
        res = minimize(obj_fallback, x0, method='Nelder-Mead', options={'maxiter': 1000})
        if res.fun < best_score:
            best_score = res.fun
            # Reconstruct
            x, y, theta_deg, outer_angle = res.x
            theta_rad = math.radians(theta_deg)
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            final_positions = np.zeros((N, 2))
            for i in range(N):
                px, py = init_positions[i]
                rx = px * cos_t - py * sin_t
                ry = px * sin_t + py * cos_t
                final_positions[i] = [rx + x, ry + y]
            final_angles = np.full(N, theta_deg)
            best_params = (final_positions, final_angles, outer_angle)
        return [list(final_positions[i]) + [final_angles[i]] for i in range(N)], (0.0, 0.0), best_score, best_params[2]

    # Optimize each valid structure
    for coords in valid_struct_coords:
        if time.time() > deadline - 20:
            break
            
        # Initial guess: centered at 0,0, rotation 0, outer angle 0
        x0 = [0.0, 0.0, 0.0, 0.0]
        
        # Try multiple outer angles? No, optimize it.
        # Nelder-Mead is robust for this low-dimensional problem (4 vars)
        try:
            res = minimize(
                objective, 
                x0, 
                args=(coords,),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                x, y, theta_deg, outer_angle = res.x
                
                # Reconstruct final positions and angles
                theta_rad = math.radians(theta_deg)
                cos_t = math.cos(theta_rad)
                sin_t = math.sin(theta_rad)
                
                final_positions = np.zeros((N, 2))
                for i, (px, py) in enumerate(coords):
                    rx = px * cos_t - py * sin_t
                    ry = px * sin_t + py * cos_t
                    final_positions[i] = [rx + x, ry + y]
                
                final_angles = np.full(N, theta_deg)
                
                # Verify validity (should be valid by construction, but check just in case)
                if is_valid(final_positions, final_angles):
                    best_params = (final_positions, final_angles, outer_angle)
        except Exception:
            pass

    if best_params is None:
        # Fallback
        final_positions = np.zeros((N, 2))
        final_angles = np.zeros(N)
        final_outer_angle = 0.0
        best_score = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    else:
        final_positions, final_angles, final_outer_angle = best_params

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), best_score, final_outer_angle

if __name__ == "__main__":
    construct()
