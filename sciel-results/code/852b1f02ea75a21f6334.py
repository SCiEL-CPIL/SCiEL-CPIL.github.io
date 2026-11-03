import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Vertices of a unit regular hexagon (aligned)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([k * np.pi / 3 for k in range(6)])
    base_vertices = np.vstack([np.cos(angles), np.sin(angles)]).T
    
    # Normals for the outer hexagon containment check
    # The apothem direction (normal to side) is offset by 30 degrees (pi/6) from vertices
    normal_angles_base = np.array([np.pi/6 + k * np.pi / 3 for k in range(6)])
    
    def get_outer_normals(theta):
        # Rotate the base normal angles by theta
        current_angles = normal_angles_base + theta
        return np.vstack([np.cos(current_angles), np.sin(current_angles)]).T

    def compute_S_and_penalty(positions, theta):
        """
        positions: (N, 2) array of centers
        theta: rotation of outer hexagon
        Returns: (S, overlap_penalty)
        """
        # Compute vertices for all inner hexagons
        # positions shape (N, 2), base_vertices shape (6, 2)
        # vertices[i, j] = positions[i] + base_vertices[j]
        vertices = positions[:, np.newaxis, :] + base_vertices[np.newaxis, :, :]
        
        normals = get_outer_normals(theta)
        
        # Projection: vertices (N, 6, 2) dot normals (6, 2)
        # We need max over j (vertex index) and k (normal index) of v_ij . n_k
        # v_ij . n_k = sum(v_ij * n_k)
        
        # Efficient calculation:
        # For each normal k, compute dot product with all vertices
        # projections[i, j, k] = vertices[i, j] . normals[k]
        
        # Use broadcasting
        # vertices: (N, 6, 2)
        # normals: (6, 2) -> (1, 1, 6, 2) ? No.
        
        # Let's compute for each normal k
        max_proj = -np.inf
        for k in range(6):
            n = normals[k]
            # Dot product of all vertices with n
            # vertices shape (N, 6, 2), n shape (2,)
            dots = np.sum(vertices * n, axis=2) # Shape (N, 6)
            # Max over vertices j for this normal k
            max_dot_k = np.max(dots)
            if max_dot_k > max_proj:
                max_proj = max_dot_k
        
        # S is determined by max_proj
        # max_proj <= S * sqrt(3) / 2
        S = max_proj * 2.0 / np.sqrt(3.0)
        
        # Overlap penalty
        # Distance between all pairs
        # positions shape (N, 2)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # We only care about i < j
        # Create mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dists[mask]
        
        min_dist = np.sqrt(3.0)
        violations = np.maximum(0.0, min_dist - pair_dists)
        penalty = np.sum(violations ** 2)
        
        return S, penalty

    def objective(vars):
        """
        vars: flattened array of length 2*N + 1
        First 2*N elements are x0, y0, x1, y1, ...
        Last element is theta
        """
        positions = vars[:2*N].reshape(N, 2)
        theta = vars[2*N]
        
        S, penalty = compute_S_and_penalty(positions, theta)
        
        # We want to minimize S, but ensure no overlap.
        # If penalty > 0, we are in an invalid state.
        # Use a large weight for penalty to drive it to 0.
        return S + 100.0 * penalty

    best_score = np.inf
    best_vars = None
    
    # Random restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initial guess: Random positions in a circle of radius 4
        # Or a structured guess
        if r == 0:
            # Structured guess: Hexagonal packing
            # Center at 0,0
            # 6 neighbors at distance sqrt(3)
            # 5 more...
            positions = np.zeros((N, 2))
            # 1 center
            # 6 neighbors
            for k in range(6):
                angle = k * np.pi / 3
                positions[k+1] = [np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)]
            # 5 more: place them further out?
            # Just randomize the rest
            for k in range(7, N):
                angle = random.uniform(0, 2*np.pi)
                dist = random.uniform(2.5, 3.5)
                positions[k] = [dist*np.cos(angle), dist*np.sin(angle)]
            theta = random.uniform(0, np.pi/3)
        else:
            positions = np.random.uniform(-3, 3, (N, 2))
            theta = random.uniform(0, np.pi/3)
        
        vars_init = np.concatenate([positions.flatten(), [theta]])
        
        # Optimize
        try:
            result = scipy.optimize.minimize(
                objective, 
                vars_init, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if result.success or result.fun < best_score:
                # Check if valid (penalty ~ 0)
                pos_opt = result.x[:2*N].reshape(N, 2)
                theta_opt = result.x[2*N]
                S_opt, pen_opt = compute_S_and_penalty(pos_opt, theta_opt)
                if pen_opt < 1e-3:
                    if S_opt < best_score:
                        best_score = S_opt
                        best_vars = result.x
        except Exception as e:
            pass
            
    if best_vars is None:
        # Fallback: Return a very large side length to indicate failure or use a simple valid packing
        # Simple valid packing: Line up in a row?
        # Distance between centers must be >= sqrt(3).
        # Total length approx 11 * sqrt(3) + 2 * 1 (radius) ~ 21
        inner = [[k * np.sqrt(3), 0, 0] for k in range(N)]
        return inner, (0.0, 0.0), 30.0, 0.0

    # Extract best solution
    positions = best_vars[:2*N].reshape(N, 2)
    theta = best_vars[2*N]
    
    # Refine with higher precision if time permits
    if time.time() - start_time < timeout - 10:
        try:
            result_refine = scipy.optimize.minimize(
                objective, 
                best_vars, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result_refine.fun < best_score:
                pos_final = result_refine.x[:2*N].reshape(N, 2)
                theta_final = result_refine.x[2*N]
                S_final, pen_final = compute_S_and_penalty(pos_final, theta_final)
                if pen_final < 1e-3:
                    best_score = S_final
                    positions = pos_final
                    theta = theta_final
        except:
            pass

    # Format output
    inner = []
    for i in range(N):
        x, y = positions[i]
        # Angle is 0 for all inner hexagons in this aligned strategy
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), best_score, theta
# EVOLVE-BLOCK-END
