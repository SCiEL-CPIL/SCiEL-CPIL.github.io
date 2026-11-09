import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    sqrt3 = math.sqrt(3.0)
    r_in = sqrt3 / 2.0  # Inradius of unit hexagon

    # Unit hexagon vertices (flat-top, side length 1)
    # Angles: 0, 60, 120, 180, 240, 300
    hex_verts_base = np.array([
        [1.0, 0.0],
        [0.5, r_in],
        [-0.5, r_in],
        [-1.0, 0.0],
        [-0.5, -r_in],
        [0.5, -r_in]
    ])

    # Outer hexagon normals (pointing outward from center)
    # The outer hexagon is defined by side length L and rotation 0 (flat-top? or pointy-top?)
    # The prompt says "regular hexagon". The seed used normals at 30, 90...
    # This implies the outer hexagon has a flat edge at the top (normal 90 deg) and bottom (270 deg).
    # Vertices of such a hexagon are at 0, 60... ?
    # If normal is 90 (up), the edge is horizontal.
    # Vertices are at angles 30, 90... ? No.
    # Let's stick to the seed's definition of the outer container to be safe.
    # Seed normals: cos(pi/6 + k*pi/3), sin(pi/6 + k*pi/3).
    # k=0: 30 deg. k=1: 90 deg. k=2: 150 deg.
    # This is a hexagon with vertices at 0, 60, 120... ?
    # Distance from center to edge (inradius) is L * sqrt(3) / 2.
    # Constraint: n_k . p <= L * sqrt(3) / 2.
    outer_normals = np.array([
        [math.cos(math.pi / 6 + k * math.pi / 3), math.sin(math.pi / 6 + k * math.pi / 3)]
        for k in range(6)
    ])

    def get_hex_verts(theta):
        """
        Get vertices of a unit hexagon rotated by theta.
        """
        c = math.cos(theta)
        s = math.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        return hex_verts_base @ rot.T

    def compute_L(positions, angles):
        """
        Compute the minimum side length L of the outer hexagon centered at (0,0)
        that contains all inner hexagons.
        """
        max_s = 0.0
        for i in range(N):
            verts = get_hex_verts(angles[i])
            all_pts = positions[i] + verts
            
            # Project all points onto all 6 outer normals
            # proj[i, k] = dot(all_pts[i], outer_normals[k])
            # We need max over i and points for each k?
            # No, for each normal k, we need the max projection of ANY point from ANY hexagon.
            # Then L_k = 2 * max_proj_k / sqrt(3).
            # L = max_k L_k.
            
            # Vectorized:
            # all_pts is (N, 6, 2)
            # outer_normals is (6, 2)
            # We want (N, 6) matrix of projections? No.
            # For each hex i, for each vertex v, for each normal k:
            # proj = dot(v + pos[i], n[k])
            
            # Let's do it efficiently.
            # verts_all: (N, 6, 2)
            # pos_all: (N, 2) -> (N, 1, 2)
            # pts_all: (N, 6, 2)
            
            # Projections: (N, 6, 6) -> N hexes, 6 verts, 6 normals
            # This is 12 * 6 * 6 = 432 values.
            # max_proj_per_normal[k] = max over i, v of proj[i, v, k]
            
            verts_all = np.array([get_hex_verts(angles[i]) for i in range(N)])
            pts_all = verts_all + positions[:, np.newaxis, :]
            
            # pts_all shape: (12, 6, 2)
            # outer_normals shape: (6, 2)
            # We want to compute dot product for each pair of (point, normal).
            # Use einsum or matmul.
            # proj[i, v, k] = sum_d pts_all[i, v, d] * outer_normals[k, d]
            # This is (12, 6, 2) @ (2, 6) ? No.
            # pts_all @ outer_normals.T -> (12, 6, 6)
            projs = np.einsum('ivd,kd->ivk', pts_all, outer_normals)
            
            # Max over i and v for each k
            max_projs = np.max(projs, axis=(0, 1)) # shape (6,)
            
            # L_k = 2 * max_projs[k] / sqrt(3)
            L_candidates = 2.0 * max_projs / sqrt3
            return np.max(L_candidates)

    def min_distance_hex(i, j, pos_i, pos_j, ang_i, ang_j):
        """
        Calculate the minimum distance between two unit hexagons.
        If they overlap, returns negative value (depth of overlap).
        Uses Separating Axis Theorem.
        """
        verts_i = get_hex_verts(ang_i) + pos_i
        verts_j = get_hex_verts(ang_j) + pos_j
        
        # Get normals for both hexagons
        # Normals are perpendicular to edges.
        # For a regular hexagon, normals are at angles theta + 30, 90, ...
        # We can compute normals from vertices:
        # Edge vector e = v[k+1] - v[k]. Normal n = rotate90(e).
        
        def get_normals(verts):
            normals = []
            for k in range(6):
                v1 = verts[k]
                v2 = verts[(k + 1) % 6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                normals.append(n)
            return np.array(normals)
        
        normals_i = get_normals(verts_i)
        normals_j = get_normals(verts_j)
        
        # Total 12 axes
        all_normals = np.vstack([normals_i, normals_j])
        
        min_dist = float('inf')
        
        for n in all_normals:
            # Normalize n
            n_len = np.linalg.norm(n)
            if n_len < 1e-9:
                continue
            n = n / n_len
            
            # Project hex i
            proj_i = np.dot(verts_i, n)
            min_i = np.min(proj_i)
            max_i = np.max(proj_i)
            
            # Project hex j
            proj_j = np.dot(verts_j, n)
            min_j = np.min(proj_j)
            max_j = np.max(proj_j)
            
            # Check separation
            # If max_i < min_j: separated by min_j - max_i
            # If max_j < min_i: separated by min_i - max_j
            # Otherwise: overlap on this axis.
            
            if max_i < min_j:
                dist = min_j - max_i
                if dist < min_dist:
                    min_dist = dist
            elif max_j < min_i:
                dist = min_i - max_j
                if dist < min_dist:
                    min_dist = dist
            else:
                # Overlap on this axis
                # The minimum distance is 0? No, if they overlap on ALL axes, they intersect.
                # If they don't intersect, there is at least one separating axis.
                # If we find an axis where they overlap, it doesn't mean they intersect.
                # We need to check if they intersect.
                # If they intersect, the minimum distance is 0 (or negative if we define overlap depth).
                # For the penalty function, we want a measure of "how much they overlap".
                # If they are separated, dist > 0.
                # If they intersect, we can estimate overlap depth.
                # A simple metric: 0.
                # But for gradient descent, 0 is not helpful.
                # We can use the maximum penetration depth?
                # For simplicity, if they overlap on this axis, we return 0?
                # No, if they overlap on *all* axes, they intersect.
                # If they are separated on *any* axis, they don't intersect.
                
                # If we are here, it means they overlap on THIS axis.
                # We cannot conclude intersection yet.
                # We must continue checking other axes.
                # If we finish all axes and they overlap on ALL, they intersect.
                # In that case, what is the "distance"?
                # We can return a small negative number to indicate overlap.
                # Or we can calculate the actual minimum distance between edges.
                
                # For efficiency, if they overlap on all axes, we return -0.1 (arbitrary penalty).
                # But this makes the function non-smooth.
                
                # Better: If they overlap on this axis, we don't update min_dist (which is for separation).
                # We set a flag "intersecting".
                pass
                
        # If we found a separating axis, min_dist is the distance.
        # If we did not (min_dist is inf), they intersect.
        if min_dist == float('inf'):
            # They intersect.
            # Estimate overlap depth.
            # We can use the maximum over all axes of the overlap length?
            # Or just return -0.1.
            return -0.1
        else:
            return min_dist

    def penalty_objective(x):
        """
        x: (36,) -> 12 positions (2 each), 12 angles.
        Returns L_outer + lambda * sum(overlap_penalty)
        """
        positions = x[:24].reshape(12, 2)
        angles = x[24:36]
        
        L = compute_L(positions, angles)
        
        overlap_penalty = 0.0
        # Check pairs
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_distance_hex(i, j, positions[i], positions[j], angles[i], angles[j])
                if dist < 0:
                    # Overlapping
                    # Penalty proportional to square of overlap depth
                    overlap_penalty += (dist ** 2)
        
        # Weight for penalty
        # If L is around 5, penalty should be significant if overlap is 0.1.
        # 0.1^2 = 0.01.
        # Let's use a weight that scales with L? Or fixed.
        # Fixed weight 10.0 seems reasonable.
        return L + 10.0 * overlap_penalty

    # Bounds
    # Positions: within [-5, 5]
    # Angles: [0, 2*pi]
    bounds = []
    for _ in range(24):
        bounds.append((-5.0, 5.0))
    for _ in range(12):
        bounds.append((0.0, 2.0 * math.pi))

    best_L = float('inf')
    best_x = None
    
    # Use Differential Evolution for global search
    # This is robust to non-smoothness.
    
    if time.time() < deadline - 20:
        try:
            # DE can be slow, limit maxiter and popsize
            result_de = differential_evolution(
                penalty_objective, 
                bounds=bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-5,
                polish=False # We will polish manually
            )
            
            if result_de.success:
                x_de = result_de.x
                L_de = compute_L(x_de[:24].reshape(12, 2), x_de[24:36])
                if L_de < best_L:
                    best_L = L_de
                    best_x = x_de
        except Exception:
            pass

    # Local refinement with L-BFGS-B
    # Start from best DE result or a good heuristic
    
    def get_initial_guess():
        positions = []
        angles = []
        # Hexagonal grid
        positions.append([0.0, 0.0])
        angles.append(0.0)
        for k in range(6):
            angle = k * math.pi / 3.0
            x = sqrt3 * math.cos(angle)
            y = sqrt3 * math.sin(angle)
            positions.append([x, y])
            angles.append(0.0)
        # Ring 2
        coords = [(2, 0), (2, -1), (1, 1), (-1, 2), (-2, 1)]
        for q, r in coords:
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            positions.append([x, y])
            angles.append(0.0)
        
        positions = np.array(positions)
        angles = np.array(angles)
        # Scale down to ensure disjointness?
        # The grid is touching. Scale by 1.05 to separate.
        positions *= 1.05
        return np.concatenate([positions.flatten(), angles])

    x0 = best_x if best_x is not None else get_initial_guess()
    
    # Polish with L-BFGS-B
    # We need to be careful with the non-smooth penalty.
    # L-BFGS-B might get stuck.
    # Try a few random restarts of local polishing if time permits.
    
    if time.time() < deadline - 5:
        try:
            res = minimize(
                penalty_objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if res.success:
                L_res = compute_L(res.x[:24].reshape(12, 2), res.x[24:36])
                # Check for overlaps
                pos_res = res.x[:24].reshape(12, 2)
                ang_res = res.x[24:36]
                has_overlap = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = min_distance_hex(i, j, pos_res[i], pos_res[j], ang_res[i], ang_res[j])
                        if d < 0:
                            has_overlap = True
                            break
                    if has_overlap:
                        break
                
                # Only accept if no overlaps (or very small)
                if not has_overlap and L_res < best_L:
                    best_L = L_res
                    best_x = res.x
        except Exception:
            pass

    # Fallback: If best_x is still None or has overlaps, use the initial guess and ensure no overlaps
    # by scaling positions up slightly? No, that increases L.
    # We must ensure the returned solution is valid.
    
    if best_x is None:
        best_x = get_initial_guess()
        best_L = compute_L(best_x[:24].reshape(12, 2), best_x[24:36])

    positions = best_x[:24].reshape(12, 2)
    angles = best_x[24:36]
    
    # Final Safety Check: Ensure no overlaps.
    # If there are overlaps, we must fix them.
    # Simple fix: Move overlapping pairs apart.
    for _ in range(10): # Iterations
        fixed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = min_distance_hex(i, j, positions[i], positions[j], angles[i], angles[j])
                if d < 0:
                    # Overlap
                    # Move them apart along the vector connecting centers
                    vec = positions[j] - positions[i]
                    norm = np.linalg.norm(vec)
                    if norm < 1e-6:
                        vec = np.array([1.0, 0.0])
                        norm = 1.0
                    else:
                        vec = vec / norm
                    
                    # Push apart by -d + epsilon
                    push = -d + 0.01
                    positions[i] -= vec * push / 2.0
                    positions[j] += vec * push / 2.0
                    fixed = True
        if not fixed:
            break

    # Recalculate L
    L_final = compute_L(positions, angles)
    
    # Add a tiny margin to L to account for numerical errors in the evaluator
    L_final *= 1.001

    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i] * 180.0 / math.pi)])
    
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, float(L_final), angle

# EVOLVE-BLOCK-END
