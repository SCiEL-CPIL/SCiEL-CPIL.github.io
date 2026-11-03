import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0  # Increased margin to be safe

    # Hexagon geometry
    # Inner hexagon side length 1
    # Vertices of a unit hexagon with rotation 0
    # Angle 0: vertex at (1, 0)
    base_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Core centers: 1 center + 6 neighbors
    # Distance between centers in honeycomb is SQRT3
    d = SQRT3
    core_centers = np.array([[0.0, 0.0]])
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        core_centers = np.vstack([core_centers, [[x, y]]])
        
    # Core rotations: all 0
    core_rots = np.zeros(7)
    
    # Ring 2 centers
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (SQRT3 / 2.0) * r
                ring2_centers.append([x, y])
    ring2_centers = np.array(ring2_centers)
    
    def get_max_projection(centers, rots, theta):
        # centers: (11, 2)
        # rots: (11,)
        # theta: scalar
        
        all_verts = []
        for i in range(11):
            cx, cy = centers[i]
            phi = rots[i]
            c, s = math.cos(phi), math.sin(phi)
            vx = base_verts[:, 0]
            vy = base_verts[:, 1]
            new_vx = c * vx - s * vy
            new_vy = s * vx + c * vy
            verts = np.column_stack([cx + new_vx, cy + new_vy])
            all_verts.append(verts)
            
        all_verts = np.vstack(all_verts) # (66, 2)
        
        # Container normals
        k_angles = np.arange(6) * math.pi / 3
        normal_angles = theta + k_angles
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # (6, 2)
        
        dots = all_verts @ normals.T
        abs_dots = np.abs(dots)
        
        max_proj = np.max(abs_dots)
        
        # S = 2/sqrt(3) * max_proj
        return (2.0 / SQRT3) * max_proj

    def project_valid(pos):
        """
        Takes positions of 4 outer hexagons (4, 2).
        Pushes them apart from core and each other until valid.
        Returns the projected positions.
        """
        pos = pos.copy()
        min_dist = SQRT3
        # We do a few iterations of gradient descent on the overlap constraint
        # Or simply a simple repulsion force
        for _ in range(50):
            moved = False
            # Check against core
            for i in range(4):
                for j in range(7):
                    dx = pos[i, 0] - core_centers[j, 0]
                    dy = pos[i, 1] - core_centers[j, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = math.sqrt(dist_sq)
                    if dist < min_dist:
                        # Push away
                        if dist < 1e-6:
                            # Random direction
                            ang = np.random.uniform(0, 2*math.pi)
                            pos[i, 0] += 0.1 * math.cos(ang)
                            pos[i, 1] += 0.1 * math.sin(ang)
                        else:
                            # Move along the vector
                            force = (min_dist - dist) * 0.5
                            pos[i, 0] += force * (dx / dist)
                            pos[i, 1] += force * (dy / dist)
                        moved = True
            
            # Check against each other
            for i in range(4):
                for j in range(i+1, 4):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = math.sqrt(dist_sq)
                    if dist < min_dist:
                        if dist < 1e-6:
                            ang = np.random.uniform(0, 2*math.pi)
                            pos[i, 0] += 0.1 * math.cos(ang)
                            pos[i, 1] += 0.1 * math.sin(ang)
                            pos[j, 0] -= 0.1 * math.cos(ang)
                            pos[j, 1] -= 0.1 * math.sin(ang)
                        else:
                            force = (min_dist - dist) * 0.25
                            pos[i, 0] += force * (dx / dist)
                            pos[i, 1] += force * (dy / dist)
                            pos[j, 0] -= force * (dx / dist)
                            pos[j, 1] -= force * (dy / dist)
                        moved = True
            
            if not moved:
                break
        return pos

    def objective(vars):
        # Check time
        if time.time() > deadline - safety_margin:
            return 1e9
            
        # Unpack
        pos = vars[:8].reshape(4, 2)
        rots_outer = vars[8:12]
        theta = vars[12]
        
        # Project to valid space
        pos_valid = project_valid(pos)
        
        # Construct full centers and rots
        centers = np.vstack([core_centers, pos_valid])
        rots = np.concatenate([core_rots, rots_outer])
        
        # Calculate S
        S = get_max_projection(centers, rots, theta)
        return S

    # Initializations
    best_S = float('inf')
    best_vars = None
    
    # Generate a list of initial guesses
    # Using specific indices from ring2_centers
    # ring2_centers has 12 points. Indices 0-11.
    # We pick sets of 4.
    init_sets = [
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 2, 4, 6],
        [0, 1, 5, 6],
        [0, 3, 6, 9],
        [0, 1, 2, 8],
        [0, 2, 5, 7],
        [0, 1, 4, 5],
        [0, 2, 3, 4]
    ]
    
    for idx_set in init_sets:
        if time.time() > deadline - safety_margin:
            break
            
        # Initial positions
        init_pos = ring2_centers[idx_set].copy()
        # Add small noise
        noise = np.random.normal(0, 0.1, init_pos.shape)
        init_pos = init_pos + noise
        
        # Project initial guess to be valid
        init_pos = project_valid(init_pos)
        
        # Initial rotations: 0
        init_rots = np.zeros(4)
        
        # Initial theta: 0
        init_theta = 0.0
        
        init_vars = np.concatenate([init_pos.flatten(), init_rots, [init_theta]])
        
        # Run optimizer
        try:
            res = minimize(objective, init_vars, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_S and res.fun < 1e8:
                best_S = res.fun
                best_vars = res.x
        except Exception:
            pass

    # Fallback if optimization fails
    if best_vars is None:
        # Use the first init set projected
        pos = project_valid(ring2_centers[[0, 1, 2, 3]])
        centers = np.vstack([core_centers, pos])
        rots = np.concatenate([core_rots, np.zeros(4)])
        S = get_max_projection(centers, rots, 0.0)
        inner = []
        for i in range(11):
            inner.append([centers[i, 0], centers[i, 1], 0.0])
        return inner, (0.0, 0.0), S, 0.0

    # Extract best result
    pos = best_vars[:8].reshape(4, 2)
    rots_outer = best_vars[8:12]
    theta = best_vars[12]
    
    # Final projection to ensure validity
    pos = project_valid(pos)
    
    centers = np.vstack([core_centers, pos])
    rots = np.concatenate([core_rots, rots_outer])
    
    # Final S calculation
    final_S = get_max_projection(centers, rots, theta)
    
    inner = []
    for i in range(11):
        inner.append([centers[i, 0], centers[i, 1], math.degrees(rots[i])])
        
    return inner, (0.0, 0.0), final_S, math.degrees(theta)
