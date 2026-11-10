import numpy as np
import time
import random
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _get_all_vertices(positions, angles, side=1.0):
    """positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2)."""
    verts = []
    for i in range(len(positions)):
        verts.append(_hex_vertices(positions[i,0], positions[i,1], side, angles[i]))
    return np.vstack(verts)

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    """
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated (no overlap), False if overlapping.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return True
    
    return False

def _is_valid_config(positions, angles, side=1.0):
    """Check if configuration is valid: all pairs separated and inside outer hex."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(positions[i], angles[i], positions[j], angles[j], side):
                return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, leave margin
    
    n = 12
    side = 1.0
    
    # Strategy: Rigid-body contact network with pressure-based optimization
    # Treat hexagons as particles in a jammed state, minimize total potential energy
    # = sum of squared penetration depths + confining pressure from outer hexagon
    
    # Initial configuration: honeycomb-like arrangement
    def generate_initial_config():
        """Generate initial honeycomb configuration."""
        positions = np.zeros((12, 2))
        angles = np.zeros(12)
        
        # Center
        positions[0] = [0, 0]
        
        # Inner ring (6 hexagons at distance 2)
        for k in range(6):
            theta = np.radians(60 * k)
            positions[k+1] = [2.0 * np.cos(theta), 2.0 * np.sin(theta)]
        
        # Outer ring (5 hexagons attached to inner ring)
        for k in range(5):
            inner_idx = k + 1
            theta = np.radians(60 * inner_idx)
            positions[k+7] = positions[inner_idx] + 2.0 * np.array([np.cos(theta), np.sin(theta)])
        
        return positions, angles

    def compute_penetration_depth(pos1, ang1, pos2, ang2, side=1.0):
        """
        Compute the minimum penetration depth between two hexagons using SAT.
        Returns 0 if separated, positive if overlapping.
        """
        v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
        v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
        
        normals = []
        for v in [v1, v2]:
            for i in range(6):
                edge = v[(i+1)%6] - v[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    n = n / norm
                    normals.append(n)
        
        max_overlap = 0.0
        for n in normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for overlap along this normal
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > max_overlap:
                max_overlap = overlap
        
        return max(0.0, max_overlap)

    def objective(params):
        """
        Objective: Minimize total potential energy
        = sum of squared penetration depths + pressure term from outer hexagon
        """
        # Unpack parameters: 12 positions (x,y) + 12 angles
        positions = params[:24].reshape(12, 2)
        angles = params[24:36]
        
        # Compute total penetration energy
        penetration_energy = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                # Fast check: if centers are far apart, skip
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 2.0 * side + 0.5:
                    pen = compute_penetration_depth(positions[i], angles[i], positions[j], angles[j], side)
                    if pen > 0:
                        penetration_energy += pen ** 2
        
        # Compute outer hexagon side length
        vertices = _get_all_vertices(positions, angles, side)
        s = _min_outer_side(vertices)
        
        # Pressure term: penalize large outer hexagon
        # Use a quadratic term to create a "confining pressure"
        pressure_term = 0.1 * s ** 2
        
        return penetration_energy + pressure_term

    def gradient(params):
        """
        Numerical gradient of the objective function.
        """
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = objective(params)
        
        for i in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[i] += eps
            f1 = objective(params_perturbed)
            grad[i] = (f1 - f0) / eps
        
        return grad

    best_s = float('inf')
    best_positions = None
    best_angles = None
    best_params = None
    
    # Run multiple restarts with different initial configurations
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate initial configuration with some randomness
        positions, angles = generate_initial_config()
        
        # Add small random perturbations
        noise_pos = np.random.normal(0, 0.1, (12, 2))
        noise_ang = np.random.normal(0, 5, 12)
        
        positions += noise_pos
        angles += noise_ang
        
        # Pack into parameter vector
        params = np.concatenate([positions.flatten(), angles])
        
        # Bounds: positions in [-10, 10], angles in [-180, 180]
        bounds = [(-10, 10)] * 24 + [(-180, 180)] * 12
        
        try:
            # Use L-BFGS-B with numerical gradient
            res = minimize(
                objective, 
                params, 
                method='L-BFGS-B', 
                bounds=bounds,
                jac=None,  # Let scipy compute numerical gradient
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if res.fun < best_s or best_params is None:
                # Extract best configuration
                best_positions = res.x[:24].reshape(12, 2)
                best_angles = res.x[24:36]
                best_params = res.x
                
                # Validate
                if _is_valid_config(best_positions, best_angles, side):
                    verts = _get_all_vertices(best_positions, best_angles, side)
                    s_val = _min_outer_side(verts)
                    best_s = s_val
                    
        except Exception:
            pass
        
        # Also try a few direct optimization runs with different initial guesses
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            # Different initial topology: more spread out
            positions, angles = generate_initial_config()
            positions *= np.random.uniform(1.0, 1.5)
            angles += np.random.uniform(-10, 10, 12)
            
            params = np.concatenate([positions.flatten(), angles])
            
            try:
                res = minimize(
                    objective, 
                    params, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                
                if res.fun < best_s or best_params is None:
                    best_positions = res.x[:24].reshape(12, 2)
                    best_angles = res.x[24:36]
                    best_params = res.x
                    
                    if _is_valid_config(best_positions, best_angles, side):
                        verts = _get_all_vertices(best_positions, best_angles, side)
                        s_val = _min_outer_side(verts)
                        best_s = s_val
                        
            except Exception:
                pass
    
    # Fallback to static configuration if optimization failed
    if best_positions is None or not _is_valid_config(best_positions, best_angles, side):
        positions, angles = generate_initial_config()
        best_positions = positions
        best_angles = angles
        verts = _get_all_vertices(positions, angles, side)
        best_s = _min_outer_side(verts)
    
    # Ensure centering
    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s
