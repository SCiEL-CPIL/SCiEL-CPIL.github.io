import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_all_verts(positions, angles):
        """
        Vectorized computation of all hexagon vertices.
        positions: (N, 2)
        angles: (N,)
        Returns: (N, 6, 2)
        """
        n = positions.shape[0]
        angles_rad = np.radians(angles)
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)
        
        # Rotation matrices: (N, 2, 2)
        rot = np.zeros((n, 2, 2))
        rot[:, 0, 0] = cos_a
        rot[:, 0, 1] = -sin_a
        rot[:, 1, 0] = sin_a
        rot[:, 1, 1] = cos_a
        
        # rotated = np.einsum('nij,jk->nik', rot, unit_verts_0.T) # (N, 2, 6)
        # rotated = np.transpose(rotated, (0, 2, 1)) # (N, 6, 2)
        # For speed, we can do this more simply in the penalty loop if needed, 
        # but vectorized is good for the objective function.
        
        # Let's stick to a slightly faster manual loop for the critical path or 
        # ensure einsum is efficient. 
        # Actually, let's just compute vertices once per call.
        
        rotated = np.einsum('nij,jk->nik', rot, unit_verts_0.T)
        rotated = np.transpose(rotated, (0, 2, 1))
        rotated += positions[:, np.newaxis, :]
        return rotated

    def calc_min_outer_side_vectorized(positions, angles, outer_angle_deg):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle_deg)
        # Normals for the outer hexagon sides
        # Distance from center to side (apothem) is S * sqrt(3) / 2.
        # Normals are at angles theta + 30 + k*60.
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        # Get all vertices (N, 6, 2)
        verts = get_all_verts(positions, angles)
        
        # Project all vertices onto all normals
        # verts: (N, 6, 2)
        # normals: (6, 2)
        # We want max projection for each normal.
        # proj[i, k, j] = dot(verts[i, j], normals[k])
        # Using einsum: 'ijl,kl->ijk' where l is dim 2
        projs = np.einsum('ijl,kl->ijk', verts, normals) # (N, 6, 6)
        
        # Max projection per normal (across all hexagons and vertices)
        # We need max over i (hexagon) and j (vertex) for each k (normal)
        max_projs = np.max(projs, axis=(0, 1)) # (6,)
        
        # The side length S is such that apothem = S * sqrt(3) / 2
        # The projection distance from center is the apothem.
        # So S = max_proj * 2 / sqrt(3)
        return np.max(max_projs) * 2.0 / SQRT3

    def sat_gap_fast(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def calc_overlap_penalty(positions, angles):
        """
        Calculate a smooth penalty for overlaps.
        """
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap_fast(polys[i], polys[j])
                if gap < 0:
                    penalty += gap**2
        return penalty

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap_fast(polys[i], polys[j]) < -tol:
                    return False
        return True

    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:2*N+N]
        outer_angle = params[2*N+N]
        
        side = calc_min_outer_side_vectorized(positions, angles, outer_angle)
        penalty = calc_overlap_penalty(positions, angles)
        
        # Weight for penalty should be high enough to prevent overlap
        # but low enough to allow the optimizer to move through overlaps if necessary
        # to find a better topology.
        return side + 100.0 * penalty

    # --- Initialization ---
    
    # Create a reasonable initial configuration: a hexagonal lattice
    sqrt3 = math.sqrt(3)
    # Lattice vectors for hexagonal packing
    v1 = np.array([1.5, sqrt3/2])
    v2 = np.array([0, sqrt3])
    
    # Generate points in a grid
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            points.append(p)
    
    # Sort by distance from origin to pick the central ones
    points.sort(key=lambda p: np.dot(p, p))
    init_pos = np.array(points[:N])
    
    # Center the configuration
    centroid = np.mean(init_pos, axis=0)
    init_pos -= centroid
    
    init_angles = np.zeros(N)
    init_outer_angle = 0.0
    
    # Scale down slightly to ensure they start inside a reasonable bound?
    # Actually, the objective function handles the outer size dynamically.
    # But if they start too far apart, side will be large.
    # The lattice spacing is ~2.88 between centers for side 1 hexes.
    # Distance from center to farthest vertex in a 12-hex cluster is roughly 3-4 units.
    # So outer side will be ~ 3-4 * 2/sqrt(3) ~ 4-5.
    
    init_params = np.concatenate([init_pos.flatten(), init_angles, [init_outer_angle]])
    
    # --- Simulated Annealing ---
    
    np.random.seed(42)
    random.seed(42)
    
    current_params = init_params.copy()
    current_score = objective(current_params)
    
    best_params = current_params.copy()
    best_score = current_score
    
    T = 1.0  # Initial temperature
    cooling_rate = 0.9998
    step_size = 0.5
    max_iter = 50000
    
    # Track time
    start_sa = time.time()
    
    for iteration in range(max_iter):
        if time.time() > deadline - 20:  # Leave 20s for polishing
            break
            
        # Perturb a random subset of parameters
        new_params = current_params.copy()
        
        # Perturb positions and angles of 1-3 hexagons
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, N-1)
            # Position
            new_params[2*idx] += np.random.randn() * step_size
            new_params[2*idx+1] += np.random.randn() * step_size
            # Angle
            new_params[2*N + idx] += np.random.randn() * step_size * 10 # degrees
            
        # Perturb outer angle
        if random.random() < 0.1:
            new_params[2*N+N] += np.random.randn() * step_size * 5
            
        new_score = objective(new_params)
        
        # Acceptance
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            delta = new_score - current_score
            if T > 1e-6:
                acceptance_prob = math.exp(-delta / T)
                if random.random() < acceptance_prob:
                    current_params = new_params
                    current_score = new_score
                    
        T *= cooling_rate
        # Slowly reduce step size
        if iteration % 1000 == 0:
            step_size *= 0.95

    # --- Polishing Phase ---
    
    # Use the best valid configuration found
    pos = best_params[:2*N].reshape(N, 2)
    ang = best_params[2*N:2*N+N]
    outer_ang = best_params[2*N+N]
    
    # Check if the best found is valid. If not, try to repair or use a fallback.
    # If it's not valid, the penalty was non-zero. 
    # We can run a local optimization that heavily penalizes overlap to try to fix it,
    # or just accept the best score if it's close.
    # Given the high penalty weight (100), the optimizer likely found a valid or near-valid config.
    
    # Let's run a Nelder-Mead polish on the best params found by SA
    # to squeeze out the last bit of precision.
    
    def polish_objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:2*N+N]
        outer_angle = params[2*N+N]
        
        side = calc_min_outer_side_vectorized(positions, angles, outer_angle)
        penalty = calc_overlap_penalty(positions, angles)
        
        # Much higher penalty for polishing to ensure validity
        return side + 1000.0 * penalty

    try:
        res = minimize(
            polish_objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        
        # Verify the polished result
        p_pos = res.x[:2*N].reshape(N, 2)
        p_ang = res.x[2*N:2*N+N]
        p_outer = res.x[2*N+N]
        
        if is_valid(p_pos, p_ang, tol=1e-4):
            # Calculate the true side length without penalty
            true_side = calc_min_outer_side_vectorized(p_pos, p_ang, p_outer)
            
            # Compare with best_score (which included penalty, so if penalty is 0, they match)
            # res.fun includes penalty. If penalty is 0, res.fun == true_side.
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
                pos = p_pos
                ang = p_ang
                outer_ang = p_outer
    except Exception:
        pass
    
    # Final extraction
    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:2*N+N]
    final_outer_ang = best_params[2*N+N]
    
    # One last check: calculate the precise side length
    final_side = calc_min_outer_side_vectorized(final_pos, final_ang, final_outer_ang)
    
    # Construct return values
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], final_ang[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_ang

if __name__ == "__main__":
    construct()
