# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Uses a physics-based relaxation (L-BFGS-B) to minimize the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    # Constants
    sqrt3 = math.sqrt(3)
    N = 12  # number of inner hexagons
    
    # Precompute unit hexagon vertices at angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([[math.cos(math.radians(60*i)), math.sin(math.radians(60*i))] for i in range(6)])
    
    # Precompute outward normals for unit hexagon at angle 0
    # For a CCW polygon, outward normal of edge (v_i to v_{i+1}) is right normal: (dy, -dx)
    unit_normals = []
    for i in range(6):
        v1 = unit_verts[i]
        v2 = unit_verts[(i+1)%6]
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        # Ensure outward
        mid = (v1 + v2) / 2.0
        if np.dot(n, mid) < 0:
            n = -n
        unit_normals.append(n)
    unit_normals = np.array(unit_normals)  # (6, 2)
    
    # Outer hexagon normals at angle 0: 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([[math.cos(math.radians(30 + 60*i)), math.sin(math.radians(30 + 60*i))] for i in range(6)])
    
    def rotation_matrix(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]])
    
    def get_hex_vertices(center, angle_rad):
        """Get vertices of unit hexagon at given center and angle."""
        R = rotation_matrix(angle_rad)
        return center + (unit_verts @ R.T)
    
    def get_hex_normals(angle_rad):
        """Get outward normals of unit hexagon at given angle."""
        R = rotation_matrix(angle_rad)
        return unit_normals @ R.T
    
    def compute_s(centers, angles_rad):
        """
        Compute minimum outer hex side length for given centers and angles.
        centers: (12, 2)
        angles_rad: (12,)
        Returns s (float)
        """
        # Compute all vertices
        all_verts = []
        for i in range(N):
            verts = get_hex_vertices(centers[i], angles_rad[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)  # (72, 2)
        
        # Project onto outer normals
        projs = all_verts @ outer_normals.T  # (72, 6)
        max_proj = projs.max(axis=0)  # (6,)
        
        # s * sqrt(3)/2 >= max_proj
        # s >= max_proj * 2 / sqrt(3)
        s = np.max(max_proj) * 2.0 / sqrt3
        return s
    
    def collision_penalty(centers, angles_rad, penalty_scale=100.0):
        """
        Compute collision penalty using SAT.
        Returns total penalty (0 if no collisions).
        """
        total_penalty = 0.0
        
        for i in range(N):
            for j in range(i+1, N):
                # Get axes for both hexagons
                axes_i = get_hex_normals(angles_rad[i])  # (6, 2)
                axes_j = get_hex_normals(angles_rad[j])  # (6, 2)
                
                # Combine axes
                axes = np.vstack([axes_i, axes_j])  # (12, 2)
                
                # Get vertices
                verts_i = get_hex_vertices(centers[i], angles_rad[i])  # (6, 2)
                verts_j = get_hex_vertices(centers[j], angles_rad[j])  # (6, 2)
                
                # Project onto axes
                proj_i = verts_i @ axes.T  # (6, 12)
                proj_j = verts_j @ axes.T  # (6, 12)
                
                min_i = proj_i.min(axis=0)
                max_i = proj_i.max(axis=0)
                min_j = proj_j.min(axis=0)
                max_j = proj_j.max(axis=0)
                
                # Check separation for each axis
                # If max_i < min_j or max_j < min_i, separated on that axis
                sep_i = max_i < min_j
                sep_j = max_j < min_i
                
                # If any axis shows separation, no collision
                if np.any(sep_i) or np.any(sep_j):
                    continue
                
                # Compute minimum overlap
                overlaps = np.minimum(max_i - min_j, max_j - min_i)
                min_overlap = np.min(overlaps)
                
                if min_overlap > 0:
                    total_penalty += penalty_scale * min_overlap**2
        
        return total_penalty
    
    def containment_penalty(centers, angles_rad, s, penalty_scale=100.0):
        """
        Compute containment penalty: vertices outside outer hexagon.
        """
        total_penalty = 0.0
        
        # Outer hex vertices
        outer_verts = s * unit_verts  # at origin, angle 0
        
        # For each outer edge, check if inner vertices are inside
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1)%6]
            edge = p2 - p1
            
            # For all inner vertices
            all_inner_verts = []
            for k in range(N):
                verts = get_hex_vertices(centers[k], angles_rad[k])
                all_inner_verts.append(verts)
            all_inner_verts = np.vstack(all_inner_verts)  # (72, 2)
            
            vecs = all_inner_verts - p1  # (72, 2)
            crosses = edge[0]*vecs[:,1] - edge[1]*vecs[:,0]
            
            # If cross < 0, vertex is outside
            violations = np.maximum(0, -crosses)
            total_penalty += penalty_scale * np.sum(violations**2)
        
        return total_penalty
    
    def objective(params):
        """
        Objective function for L-BFGS-B.
        params: (36,) = [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
        """
        if time.time() > deadline:
            return 1e10
        
        centers = params[0:36:3].reshape(12, 2)
        angles_rad = params[1:36:3]
        
        # Compute s
        s = compute_s(centers, angles_rad)
        
        # Add penalties
        penalty = collision_penalty(centers, angles_rad)
        penalty += containment_penalty(centers, angles_rad, s)
        
        # Objective: minimize s + penalty
        return s + penalty
    
    def gradient(params):
        """
        Numerical gradient for L-BFGS-B.
        """
        if time.time() > deadline:
            return np.zeros(36)
        
        eps = 1e-6
        grad = np.zeros(36)
        f0 = objective(params)
        
        for i in range(36):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            grad[i] = (f_plus - f0) / eps
        
        return grad
    
    def run_lbfgsb(centers_init, angles_init, max_iter=200):
        """
        Run L-BFGS-B optimization.
        """
        params = np.concatenate([centers_init.flatten(), angles_init])
        
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success:
            params_opt = result.x
            centers_opt = params_opt[0:36:3].reshape(12, 2)
            angles_opt = params_opt[1:36:3]
            
            # Center the configuration
            centroid = np.mean(centers_opt, axis=0)
            centers_opt = centers_opt - centroid
            
            s_opt = compute_s(centers_opt, angles_opt)
            
            # Validate
            penalty = collision_penalty(centers_opt, angles_opt)
            penalty += containment_penalty(centers_opt, angles_opt, s_opt)
            
            if penalty < 1e-4:
                return s_opt, centers_opt, angles_opt, result.fun
            else:
                return None, None, None, result.fun
        else:
            return None, None, None, result.fun
    
    def validate_and_score(centers, angles_rad):
        """
        Validate configuration and compute score.
        """
        # Center
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        
        s = compute_s(centers, angles_rad)
        
        # Check collisions
        penalty_col = collision_penalty(centers, angles_rad, penalty_scale=1.0)
        
        # Check containment
        penalty_cont = containment_penalty(centers, angles_rad, s, penalty_scale=1.0)
        
        total_penalty = penalty_col + penalty_cont
        
        if total_penalty > 1e-6:
            return -1, None, None
        
        return 1.0/s, centers, s
    
    # Initial configuration: hexagonal lattice
    def generate_initial_lattice():
        """Generate initial hexagonal lattice configuration."""
        dx = 2.5
        dy = sqrt3
        
        centers = []
        angles = np.zeros(12)
        
        # 3 rows of 4
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2.0
                centers.append([x, y])
        
        return np.array(centers), angles
    
    best_val = 0.0
    best_centers = None
    best_angles = None
    best_s = float('inf')
    
    # Run multiple restarts with different initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Hexagonal lattice
            init_centers, init_angles = generate_initial_lattice()
        elif restart == 1:
            # Slightly perturbed lattice
            init_centers, init_angles = generate_initial_lattice()
            init_centers += np.random.normal(0, 0.5, init_centers.shape)
        elif restart == 2:
            # 4-4-4 grid
            init_centers = []
            init_angles = np.zeros(12)
            dx = 2.5
            dy = sqrt3
            for r in range(4):
                for c in range(3):
                    x = c * dx
                    y = r * dy
                    if r % 2 == 1:
                        x += dx / 2.0
                    init_centers.append([x, y])
            init_centers = np.array(init_centers)
        elif restart == 3:
            # Random initial
            init_centers = np.random.uniform(-4, 4, (12, 2))
            init_angles = np.zeros(12)
        else:
            # Random with some structure
            init_centers = np.random.uniform(-3, 3, (12, 2))
            init_angles = np.random.uniform(0, math.pi/3, 12)
        
        # Run L-BFGS-B
        s_opt, centers_opt, angles_opt, _ = run_lbfgsb(init_centers, init_angles, max_iter=100)
        
        if s_opt is not None:
            val, centers_valid, s_valid = validate_and_score(centers_opt, angles_opt)
            if val > 0 and val > best_val:
                best_val = val
                best_centers = centers_valid
                best_angles = angles_opt
                best_s = s_valid
    
    # If we haven't found a good solution, try more restarts
    if best_val < 1.5:
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Random initial with perturbation
            init_centers = np.random.uniform(-4, 4, (12, 2))
            init_angles = np.zeros(12)
            
            # Add some local structure
            for i in range(12):
                for j in range(i+1, 12):
                    dist = np.linalg.norm(init_centers[i] - init_centers[j])
                    if dist < 2.0:
                        # Push apart
                        diff = init_centers[i] - init_centers[j]
                        if np.linalg.norm(diff) > 1e-10:
                            diff /= np.linalg.norm(diff)
                            init_centers[i] += diff * (2.0 - dist) * 0.5
                            init_centers[j] -= diff * (2.0 - dist) * 0.5
            
            s_opt, centers_opt, angles_opt, _ = run_lbfgsb(init_centers, init_angles, max_iter=80)
            
            if s_opt is not None:
                val, centers_valid, s_valid = validate_and_score(centers_opt, angles_opt)
                if val > 0 and val > best_val:
                    best_val = val
                    best_centers = centers_valid
                    best_angles = angles_opt
                    best_s = s_valid
    
    # Fallback: use initial lattice if nothing better found
    if best_centers is None:
        best_centers, best_angles = generate_initial_lattice()
        centroid = np.mean(best_centers, axis=0)
        best_centers = best_centers - centroid
        best_s = compute_s(best_centers, best_angles)
        best_val = 1.0 / best_s
    
    # Prepare return
    inner_ret = np.column_stack((best_centers, best_angles * 180.0 / math.pi))
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s

# EVOLVE-BLOCK-END
