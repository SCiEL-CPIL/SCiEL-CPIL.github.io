import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    """Generate vertices of a unit hexagon centered at (x,y) with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    # Initial guess: 7-hex cluster + 4 more
    # Centers for a tight 7-hex cluster (side 1 hexes)
    # Distance between centers is sqrt(3)
    d = np.sqrt(3)
    init_centers = [
        (0, 0),
        (d, 0), (d/2, 1.5), (-d/2, 1.5),
        (-d, 0), (-d/2, -1.5), (d/2, -1.5)
    ]
    # Add 4 more, place them roughly in a ring or extend
    # Let's just scatter them initially
    init_params = []
    for i in range(N):
        if i < 7:
            x, y = init_centers[i]
        else:
            # Place new ones further out
            x = (i - 6) * 2.0
            y = 3.0
        ang = 0.0
        init_params.extend([x, y, ang])
    
    init_params = np.array(init_params)

    def calc_penalty_and_side(params):
        # Reshape to N x 3
        pts = params.reshape(N, 3)
        xs = pts[:, 0]
        ys = pts[:, 1]
        thetas = pts[:, 2]
        
        # 1. Calculate required outer hexagon side
        # Normals of outer hexagon (centered at 0,0, rotation 0)
        # Faces are perpendicular to vectors at angles 0, 60, ...
        normals = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        max_proj = 0.0
        for k in range(N):
            verts = hex_vertices(xs[k], ys[k], thetas[k])
            # Project all vertices onto all normals
            # verts is 6x2, normals is 6x2
            # We want max dot product
            projs = verts @ normals.T # 6x6
            max_proj = max(max_proj, np.max(projs))
        
        # Apothem A = max_proj
        # Side S = 2 * A / sqrt(3)
        side = 2.0 * max_proj / np.sqrt(3)
        
        # 2. Calculate overlap penalty
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        return side, penalty

    def objective(params):
        side, penalty = calc_penalty_and_side(params)
        return side + 100.0 * penalty

    best_side = float('inf')
    best_params = init_params.copy()
    
    # Use Nelder-Mead for local search
    # We will do a few random restarts
    np.random.seed(42)
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Perturb initial guess
        if r == 0:
            current_params = init_params.copy()
        else:
            # Random scatter within a box
            current_params = init_params + np.random.normal(0, 1.0, N*3)
            # Ensure angles are reasonable
            current_params[2::3] = np.random.uniform(-10, 10, N)
            
        # Run optimization
        try:
            res = scipy.optimize.minimize(
                objective, 
                current_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if res.success or res.fun < best_side + 100*calc_penalty_and_side(res.x)[1]:
                side, pen = calc_penalty_and_side(res.x)
                if pen < 1e-4: # Valid packing
                    if side < best_side:
                        best_side = side
                        best_params = res.x.copy()
        except Exception:
            pass

    # If no valid packing found, try a simpler grid or return seed
    if best_side == float('inf'):
        # Fallback: return seed-like structure
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct output
    inner = []
    for i in range(N):
        x, y, ang = best_params[i*3], best_params[i*3+1], best_params[i*3+2]
        inner.append([float(x), float(y), float(ang)])
        
    return inner, (0.0, 0.0), float(best_side), 0.0
