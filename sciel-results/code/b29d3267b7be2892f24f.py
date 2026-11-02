import numpy as np
import scipy.optimize
import time

N = 14
D = 3

def _regular_icosahedron():
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    return vertices

def _face_centers(icosahedron):
    """Find centers of triangular faces of the icosahedron."""
    # For a regular icosahedron, face centers can be computed from triples of vertices
    # that form equilateral triangles on the unit sphere
    n = len(icosahedron)
    centers = []
    used = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check if these three form a face (all pairwise distances equal)
                d_ij = np.linalg.norm(icosahedron[i] - icosahedron[j])
                d_jk = np.linalg.norm(icosahedron[j] - icosahedron[k])
                d_ik = np.linalg.norm(icosahedron[i] - icosahedron[k])
                
                if abs(d_ij - d_jk) < 1e-10 and abs(d_jk - d_ik) < 1e-10:
                    # This is a face
                    center = (icosahedron[i] + icosahedron[j] + icosahedron[k]) / 3.0
                    center_norm = np.linalg.norm(center)
                    if center_norm > 1e-10:
                        center = center / center_norm
                        # Check if we've already seen this face (or its opposite)
                        key = tuple(sorted([i, j, k]))
                        opp_key = tuple(sorted([i, j, k]))  # Will handle opposites separately
                        if key not in used:
                            used.add(key)
                            centers.append(center)
    
    # Also add opposite face centers (negated)
    all_centers = list(centers)
    for c in centers:
        opp = -c
        # Check if already present
        if not any(np.linalg.norm(opp - existing) < 1e-10 for existing in all_centers):
            all_centers.append(opp)
    
    return np.array(all_centers)

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    icosahedron = _regular_icosahedron()
    face_centers = _face_centers(icosahedron)
    
    # Select two opposite face centers
    # Find pairs of opposite faces
    n_faces = len(face_centers)
    opposite_pairs = []
    for i in range(n_faces):
        for j in range(i + 1, n_faces):
            if np.linalg.norm(face_centers[i] + face_centers[j]) < 1e-10:
                opposite_pairs.append((i, j))
    
    if len(opposite_pairs) == 0:
        # Fallback: just use first two face centers
        fc1 = face_centers[0] if n_faces > 0 else np.array([1.0, 0, 0])
        fc2 = face_centers[1] if n_faces > 1 else -fc1
    else:
        fc1 = face_centers[opposite_pairs[0][0]]
        fc2 = face_centers[opposite_pairs[0][1]]
    
    def build_points(params):
        """
        Build 14 points from parameters:
        params[0]: radial scale for point 13 (face center 1)
        params[1]: radial scale for point 14 (face center 2)
        params[2]: rotation angle around z-axis
        params[3:]: optional additional symmetry breaking (unused for now)
        """
        r1 = params[0]
        r2 = params[1]
        theta = params[2]
        
        # Rotate icosahedron vertices
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])
        
        ico_rotated = icosahedron @ R.T
        
        # Place two additional points at scaled face centers
        p13 = r1 * fc1
        p14 = r2 * fc2
        
        # Combine all 14 points
        points = np.vstack([ico_rotated, p13.reshape(1, -1), p14.reshape(1, -1)])
        return points
    
    def raw_score(points):
        """Compute the squared ratio of max to min pairwise distance."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dist2[mask])
        max_d = np.max(dists)
        min_d = np.min(dists)
        if min_d < 1e-10:
            return 1e10
        return (max_d / min_d) ** 2
    
    def objective(params):
        points = build_points(params)
        return raw_score(points)
    
    best_score = float('inf')
    best_params = None
    
    # Multi-start Nelder-Mead optimization
    np.random.seed(42)
    
    # Initial parameter ranges
    # r1, r2: radial scales for the two extra points (typically around 0.5-1.0)
    # theta: rotation angle [0, 2*pi)
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Random initial parameters
        r1_init = np.random.uniform(0.3, 1.2)
        r2_init = np.random.uniform(0.3, 1.2)
        theta_init = np.random.uniform(0, 2 * np.pi)
        
        params_init = np.array([r1_init, r2_init, theta_init])
        
        try:
            res = scipy.optimize.minimize(
                objective,
                params_init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # If we found good parameters, use them
    if best_params is not None and best_score < 1e9:
        points = build_points(best_params)
        # Center the points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        return points
    
    # Fallback to the original random optimization approach
    def smooth_objective(params, beta):
        P = params.reshape(N, D)
        centroid = np.mean(P, axis=0)
        P = P - centroid
        norm_factor = np.sqrt(np.sum(np.sum(P**2, axis=1)) / N)
        if norm_factor < 1e-10:
            return 1e10, None
        P = P / norm_factor
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        
        mask = ~np.eye(N, dtype=bool)
        dist2_masked = dist2[mask]
        
        max_d2 = np.max(dist2_masked)
        min_d2 = np.min(dist2_masked)
        
        log_sum_exp_pos = max_d2 + np.log(np.sum(np.exp(beta * (dist2_masked - max_d2))))
        M = log_sum_exp_pos / beta
        
        log_sum_exp_neg_val = -beta * min_d2 + np.log(np.sum(np.exp(-beta * (dist2_masked - min_d2))))
        m = -log_sum_exp_neg_val / beta
        
        if m < 1e-10:
            return 1e10, None
            
        val = (M / m) ** 2
        return val, None

    def objective_wrapper(params):
        return smooth_objective(params, current_beta[0])[0]

    current_beta = [1.0]
    
    best_score_rand = float('inf')
    best_params_rand = None
    
    num_restarts_rand = 10
    for i in range(num_restarts_rand):
        if time.time() - start_time > timeout:
            break
            
        P_init = np.random.randn(N, D)
        params_init = P_init.flatten()
        
        betas = [1.0, 10.0, 100.0, 1000.0]
        
        current_params = params_init.copy()
        
        for beta in betas:
            if time.time() - start_time > timeout:
                break
                
            current_beta[0] = beta
            
            try:
                res = scipy.optimize.minimize(
                    objective_wrapper, 
                    current_params, 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_params = res.x
            except Exception:
                pass
        
        P_final = current_params.reshape(N, D)
        centroid = np.mean(P_final, axis=0)
        P_final = P_final - centroid
        norm_factor = np.sqrt(np.sum(np.sum(P_final**2, axis=1)) / N)
        if norm_factor > 1e-10:
            P_final = P_final / norm_factor
            
        diff = P_final[:, np.newaxis, :] - P_final[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dist2[mask])
        
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        if min_d > 1e-10:
            score = (max_d / min_d) ** 2
            if score < best_score_rand:
                best_score_rand = score
                best_params_rand = P_final.copy()
    
    if best_params_rand is not None:
        return best_params_rand
    
    # Final fallback
    side = int(np.ceil(N ** (1 / D)))
    import itertools
    grid = list(itertools.product(range(side), repeat=D))[:N]
    return np.array(grid, dtype=float)

# EVOLVE-BLOCK-START
# The code above is the new design.
# I will place it inside the block.
# EVOLVE-BLOCK-END
