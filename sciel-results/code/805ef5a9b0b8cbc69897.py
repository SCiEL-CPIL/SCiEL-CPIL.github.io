import math
import time
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import random
from itertools import combinations

N = 12
DEADLINE = 140.0  # Safety margin for 150s limit

def construct():
    start_time = time.time()
    
    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices are at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    sqrt3 = math.sqrt(3)

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
            # Normals to the sides of the outer hexagon
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # The distance from center to side is R * cos(30) = R * sqrt(3)/2
        # Side length s = 2 * R * sin(30) = R
        # Wait, for a regular hexagon, side length s = circumradius R.
        # Distance from center to side (apothem) a = R * sqrt(3)/2.
        # So R = 2 * a / sqrt(3).
        # max_proj is the maximum projection of any vertex onto the normals.
        # This max_proj represents the maximum distance from the center along the normal direction.
        # So max_proj is the apothem of the containing hexagon.
        return max_proj * 2.0 / math.sqrt(3)

    def sat_distance(poly1, poly2, axis):
        """
        Returns the minimum distance between two convex polygons along a given axis.
        Positive means separated, negative means overlapping.
        """
        proj1 = poly1 @ axis
        proj2 = poly2 @ axis
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            return min2 - max1
        elif max2 < min1:
            return min1 - max2
        else:
            return -min(max1 - min2, max2 - min1)

    def get_axes(poly1, poly2):
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm1 = np.linalg.norm(normal1)
            if norm1 > 1e-9:
                axes.append(normal1 / norm1)
            
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm2 = np.linalg.norm(normal2)
            if norm2 > 1e-9:
                axes.append(normal2 / norm2)
        return axes

    def separation_constraints(x):
        """
        Computes the separation distances for all pairs of hexagons.
        Returns an array of shape (66,) for 12 hexagons.
        """
        positions = x[:2*N].reshape(N, 2)
        angles = x[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        dists = []
        for i in range(N):
            for j in range(i+1, N):
                axes = get_axes(polys[i], polys[j])
                min_dist = float('inf')
                for axis in axes:
                    d = sat_distance(polys[i], polys[j], axis)
                    if d < min_dist:
                        min_dist = d
                dists.append(min_dist)
        return np.array(dists)

    # Define constraints for SLSQP
    # We want separation >= 0 (allowing touching)
    # To be safe and avoid numerical issues, we can require separation >= -1e-6
    # But strictly, disjoint means separation > 0 or >= 0 depending on definition.
    # The task says "disjoint", usually meaning interiors do not intersect. Touching is allowed.
    # So constraint is g(x) >= 0.
    
    def constraint_func(x):
        return separation_constraints(x)

    # Nonlinear constraint: separation >= 0
    # scipy.optimize.NonlinearConstraint expects fun(x) to be in [lb, ub]
    # We want separation >= 0.
    # So lb=0, ub=inf.
    n_constraints = N*(N-1)//2
    nlcon = NonlinearConstraint(constraint_func, 0, np.inf)

    def objective(x):
        if time.time() > DEADLINE:
            return 1e10
        positions = x[:2*N].reshape(N, 2)
        angles = x[2*N:]
        return calc_min_outer_side(positions, angles, 0.0)

    def is_valid_config(x):
        dists = separation_constraints(x)
        # Allow small numerical error
        return np.all(dists >= -1e-5)

    # --- Part 1: Generate Discrete Lattice Candidates ---
    
    def lattice_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return (x, y)
        
    def normalize_shape(coords):
        xs = [lattice_to_cartesian(q, r)[0] for q, r in coords]
        ys = [lattice_to_cartesian(q, r)[1] for q, r in coords]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(xs)
        return [(lattice_to_cartesian(q, r)[0] - cx, lattice_to_cartesian(q, r)[1] - cy) for q, r in coords]

    def generate_random_polyhex(size):
        coords = {(0,0)}
        frontier = [(0,0)]
        while len(coords) < size:
            if not frontier:
                return None
            random.shuffle(frontier)
            q, r = frontier[0]
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            random.shuffle(neighbors)
            added = False
            for nq, nr in neighbors:
                if (nq, nr) not in coords:
                    coords.add((nq, nr))
                    frontier.append((nq, nr))
                    added = True
                    break
            if not added:
                return None
        return list(coords)

    candidate_shapes = []
    
    # Add specific compact shapes
    # 2x3 block
    shape_2x3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    candidate_shapes.append(normalize_shape(shape_2x3))
    
    # 3x2 block
    shape_3x2 = [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)]
    candidate_shapes.append(normalize_shape(shape_3x2))
    
    # 1x12 line (long)
    shape_line = [(i, 0) for i in range(12)]
    candidate_shapes.append(normalize_shape(shape_line))
    
    # Hexagonal cluster (1 center + 6 ring + 5 more)
    # Center (0,0)
    # Ring: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Next layer to reach 12: add 5 more from the next ring
    hex_cluster = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
                   (2,0), (1,1), (0,2), (-2,1), (-1,-1)] 
    # Check connectivity of hex_cluster
    # (0,0) connected to all 6.
    # (2,0) connected to (1,0).
    # (1,1) connected to (0,1) and (1,0).
    # (0,2) connected to (0,1) and (1,1).
    # (-2,1) connected to (-1,1).
    # (-1,-1) connected to (0,-1) and (-1,0).
    # It is connected.
    candidate_shapes.append(normalize_shape(hex_cluster))

    # Generate random polyhexes
    num_random = 100
    for _ in range(num_random):
        if time.time() > DEADLINE - 30:
            break
        shape = generate_random_polyhex(N)
        if shape:
            candidate_shapes.append(normalize_shape(shape))
            
    # Deduplicate
    unique_shapes = []
    seen = set()
    for shape in candidate_shapes:
        key = tuple(sorted((round(x, 3), round(y, 3)) for x, y in shape))
        if key not in seen:
            seen.add(key)
            unique_shapes.append(shape)

    best_score = float('inf')
    best_params = None

    # --- Part 2: Optimize Each Candidate ---
    
    for shape in unique_shapes:
        if time.time() > DEADLINE - 10:
            break
            
        init_positions = np.array(shape)
        init_angles = np.zeros(N)
        x0 = np.concatenate([init_positions.flatten(), init_angles])
        
        # Check initial validity
        if not is_valid_config(x0):
            continue
            
        init_score = objective(x0)
        
        # Try to optimize
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                constraints=[nlcon],
                options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
            )
            
            if is_valid_config(res.x):
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    # If no valid solution found (should not happen with lattice), fallback to a simple valid config
    if best_params is None:
        # Use the 2x3 block
        shape_2x3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
        # We need 12, so duplicate the block? No, they must be disjoint.
        # Just use the 1x12 line
        shape_line = [(i, 0) for i in range(12)]
        init_positions = np.array(normalize_shape(shape_line))
        init_angles = np.zeros(N)
        best_params = np.concatenate([init_positions.flatten(), init_angles])
        best_score = objective(best_params)

    # Final refinement with multiple restarts from the best solution
    if best_params is not None:
        current_best = best_score
        current_params = best_params
        
        # Try a few random perturbations and re-optimize
        for _ in range(5):
            if time.time() > DEADLINE - 5:
                break
                
            noise_pos = np.random.normal(0, 0.05, 2*N)
            noise_ang = np.random.normal(0, 2, N) * math.pi / 180
            x_perturbed = current_params.copy()
            x_perturbed[:2*N] += noise_pos
            x_perturbed[2*N:] += noise_ang
            
            if not is_valid_config(x_perturbed):
                continue
                
            try:
                res = minimize(
                    objective, 
                    x_perturbed, 
                    method='SLSQP', 
                    constraints=[nlcon],
                    options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
                )
                
                if is_valid_config(res.x):
                    score = res.fun
                    if score < current_best:
                        current_best = score
                        current_params = res.x
            except Exception:
                continue
        
        best_score = current_best
        best_params = current_params

    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    # Simple test
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
