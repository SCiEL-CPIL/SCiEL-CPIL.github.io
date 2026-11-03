import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 115.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial to cartesian for unit hexagons
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate diverse initial configurations
    def gen_initial_config(seed=0):
        np.random.seed(seed)
        # Start with a hexagonal arrangement
        positions = []
        # Center
        positions.append([0.0, 0.0])
        # Ring 1 (6 hexagons)
        for k in range(6):
            ang = k * math.pi / 3
            x = math.sqrt(3) * math.cos(ang)
            y = math.sqrt(3) * math.sin(ang)
            positions.append([x, y])
        # Ring 2 (partial - 4 more for total 11)
        ring2_positions = []
        for k in range(12):
            ang = k * math.pi / 6
            if k % 2 == 0:
                x = 2 * math.sqrt(3) * math.cos(ang)
                y = 2 * math.sqrt(3) * math.sin(ang)
            else:
                x = 3 * math.sqrt(3) / 2 * math.cos(ang + math.pi/6)
                y = 3 * math.sqrt(3) / 2 * math.sin(ang + math.pi/6)
            ring2_positions.append([x, y])
        # Pick 4 from ring 2
        chosen = np.random.choice(len(ring2_positions), 4, replace=False)
        for idx in chosen:
            positions.append(ring2_positions[idx])
        
        return np.array(positions)

    def compute_min_enclosing_hexagon(pts):
        """Compute minimal enclosing regular hexagon side length and center."""
        n = len(pts)
        
        # Objective: minimize the hexagon side length
        # Parameters: cx, cy, beta (rotation angle)
        # The hexagon has apothem D, so side = (D + APOTHEM) / APOTHEM
        
        def hexagon_objective(params):
            cx, cy, beta = params
            # Normalize beta to [0, pi/6)
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            
            # 6 normals of the hexagon
            max_proj = 0.0
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                nx = math.cos(ang)
                ny = math.sin(ang)
                # Project all points
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                proj_max = np.max(projs)
                proj_min = np.min(projs)
                # Need both sides to be within D
                d_required = max(abs(proj_max), abs(proj_min))
                if d_required > max_proj:
                    max_proj = d_required
            
            # Side length of enclosing hexagon
            D = max_proj
            S = (D + APOTHEM) / APOTHEM
            return S
        
        # Multi-start optimization
        best_S = float('inf')
        best_params = None
        
        # Initial guesses
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        
        start_betas = [0.0, math.pi/12, math.pi/6, math.pi/24, 2*math.pi/24, 3*math.pi/24]
        
        for beta0 in start_betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            for cx_off in [-0.5, 0.0, 0.5]:
                for cy_off in [-0.5, 0.0, 0.5]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    init = [cx0 + cx_off, cy0 + cy_off, beta0]
                    try:
                        res = minimize(hexagon_objective, init, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_params = res.x
                    except Exception:
                        pass
        
        if best_params is None:
            # Fallback
            cx, cy = cx0, cy0
            beta = 0.0
            max_proj = 0.0
            for k in range(6):
                ang = math.pi / 6.0 + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            best_S = (max_proj + APOTHEM) / APOTHEM
            best_params = np.array([cx, cy, beta])
        
        return best_S, best_params

    def force_relaxation(positions, num_iterations=5000, dt=0.01):
        """Force-directed relaxation to maximize local packing density."""
        n = len(positions)
        pos = positions.copy()
        velocities = np.zeros_like(pos)
        
        # Repulsion radius (hexagons should not overlap)
        min_dist = 2.0 * APOTHEM + 0.01  # Slightly more than touching
        
        for it in range(num_iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            forces = np.zeros_like(pos)
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    
                    if dist < 1e-10:
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        dist = math.sqrt(dx * dx + dy * dy)
                    
                    if dist < min_dist:
                        # Strong repulsion when overlapping
                        overlap = min_dist - dist
                        force_mag = 10.0 * overlap + 0.1 * overlap * overlap
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                    else:
                        # Mild attraction for packing density (within a certain range)
                        if dist < min_dist * 1.5:
                            force_mag = 0.01 * (dist - min_dist)
                            fx = force_mag * dx / dist
                            fy = force_mag * dy / dist
                            forces[i, 0] += fx
                            forces[i, 1] += fy
                            forces[j, 0] -= fx
                            forces[j, 1] -= fy
            
            # Update velocities with damping
            velocities = 0.9 * velocities + dt * forces
            pos += velocities
            
            # Center the configuration
            pos -= np.mean(pos, axis=0)
        
        return pos

    # Try multiple initial configurations and relaxations
    best_global_S = float('inf')
    best_global_pts = None
    best_global_params = None
    
    num_configs = 15
    for seed in range(num_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Generate initial configuration
        if seed == 0:
            # Perfect hexagonal packing
            positions = gen_initial_config(seed)
        elif seed == 1:
            # Linear arrangement
            positions = np.array([[i * math.sqrt(3), 0.0] for i in range(N)])
        elif seed == 2:
            # 2-row arrangement
            positions = []
            for i in range(6):
                positions.append([i * math.sqrt(3), 0.0])
            for i in range(5):
                positions.append([(i + 0.5) * math.sqrt(3), 1.5])
            positions = np.array(positions)
        elif seed == 3:
            # 3-row arrangement
            positions = []
            for i in range(4):
                positions.append([i * math.sqrt(3), 0.0])
            for i in range(4):
                positions.append([(i + 0.5) * math.sqrt(3), 1.5])
            for i in range(3):
                positions.append([i * math.sqrt(3), 3.0])
            positions = np.array(positions)
        elif seed == 4:
            # Diamond shape
            positions = []
            positions.append([0, 0])
            for d in [1, 2]:
                for k in range(4 * d):
                    ang = k * math.pi / 2 + math.pi / 4
                    x = d * math.sqrt(3) * math.cos(ang)
                    y = d * math.sqrt(3) * math.sin(ang)
                    positions.append([x, y])
            positions = np.array(positions[:N])
        else:
            # Random with some structure
            positions = gen_initial_config(seed)
            # Add some random perturbation
            positions += np.random.normal(0, 0.5, positions.shape)
        
        # Force relaxation
        relaxed = force_relaxation(positions, num_iterations=3000, dt=0.005)
        
        # Compute minimal enclosing hexagon
        S, params = compute_min_enclosing_hexagon(relaxed)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = relaxed
            best_global_params = params
    
    # If no valid configuration found, use a default
    if best_global_pts is None or best_global_S == float('inf'):
        positions = gen_initial_config(0)
        best_global_pts = positions
        best_global_S, best_global_params = compute_min_enclosing_hexagon(positions)
    
    cx, cy, beta = best_global_params
    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
    angle_deg = math.degrees(beta)
    
    # Verify and adjust side length with safety margin
    max_proj = 0.0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (best_global_pts[:, 0] - cx) * nx + (best_global_pts[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D = max_proj
    S_final = (D + APOTHEM) / APOTHEM
    
    # Add tiny safety margin
    S_final *= (1.0 + 1e-10)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
