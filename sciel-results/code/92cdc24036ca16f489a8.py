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

def point_in_hexagon(p, center, angle_deg):
    """Check if point p is inside a unit hexagon centered at 'center' with rotation 'angle_deg'."""
    # Transform p to hexagon's local coordinate system
    p_local = p - center
    angle_rad = -np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    p_local = p_local @ R.T
    
    # Unit hexagon inequalities:
    # |x| <= 1
    # |x/2 + sqrt(3)y/2| <= sqrt(3)/2  =>  |x + sqrt(3)y| <= sqrt(3)
    # Wait, the vertices are (1,0), (0.5, sqrt(3)/2), etc.
    # Faces are:
    # 1. x = 1 (normal 1,0) -> x <= 1
    # 2. x/2 + sqrt(3)y/2 = sqrt(3)/2 (normal 0.5, sqrt(3)/2) -> x/2 + sqrt(3)y/2 <= sqrt(3)/2
    # 3. -x/2 + sqrt(3)y/2 = sqrt(3)/2 (normal -0.5, sqrt(3)/2) -> -x/2 + sqrt(3)y/2 <= sqrt(3)/2
    # 4. x = -1
    # 5. -x/2 - sqrt(3)y/2 <= sqrt(3)/2
    # 6. x/2 - sqrt(3)y/2 <= sqrt(3)/2
    
    # Let's just project onto the 6 normals and check <= apothem
    # Apothem = sqrt(3)/2
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    projs = p_local @ normals.T
    apothem = np.sqrt(3)/2
    return np.all(projs <= apothem + 1e-9)

def calc_side_and_penalty(params):
    """Calculate outer hexagon side length and overlap penalty."""
    # 1. Calculate required outer hexagon side
    # Outer hexagon centered at (0,0), rotation 0
    # Normals are at 0, 60, ...
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    max_apothem = 0.0
    for k in range(N):
        x, y, ang = params[k*3], params[k*3+1], params[k*3+2]
        verts = hex_vertices(x, y, ang)
        # Project all vertices onto all normals
        for v in verts:
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_apothem:
                    max_apothem = proj
        
    side = 2.0 * max_apothem / np.sqrt(3)
    
    # 2. Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        x_i, y_i, ang_i = params[i*3], params[i*3+1], params[i*3+2]
        verts_i = hex_vertices(x_i, y_i, ang_i)
        for j in range(i+1, N):
            x_j, y_j, ang_j = params[j*3], params[j*3+1], params[j*3+2]
            verts_j = hex_vertices(x_j, y_j, ang_j)
            
            # Check vertices of i inside j
            for v in verts_i:
                if point_in_hexagon(v, np.array([x_j, y_j]), ang_j):
                    # Calculate penetration depth
                    # Distance from v to boundary of j
                    # We can approximate penalty by distance to center or just a fixed value
                    # Better: calculate min distance to boundary
                    # For simplicity, use a large penalty if inside
                    penalty += 10.0
            # Check vertices of j inside i
            for v in verts_j:
                if point_in_hexagon(v, np.array([x_i, y_i]), ang_i):
                    penalty += 10.0
    return side, penalty

def objective(params):
    side, penalty = calc_side_and_penalty(params)
    return side + penalty

def construct():
    start_time = time.time()
    deadline = 145.0

    # Generate initial configurations
    d = np.sqrt(3)
    
    def make_config_1():
        """Hexagonal cluster of 7 + 4 more"""
        centers = [
            (0, 0),
            (d, 0), (d/2, 1.5), (-d/2, 1.5),
            (-d, 0), (-d/2, -1.5), (d/2, -1.5),
            (0, 3.0), (2.6, 1.5), (2.6, -1.5), (0, -3.0)
        ]
        params = []
        for x, y in centers:
            params.extend([x, y, 0.0])
        return np.array(params)

    def make_config_2():
        """Linear arrangement"""
        params = []
        for i in range(N):
            params.extend([i * d, 0.0, 0.0])
        return np.array(params)

    def make_config_3():
        """Two rows"""
        params = []
        for i in range(N):
            if i < 6:
                params.extend([i * d, 0.0, 0.0])
            else:
                params.extend([(i-6) * d + d/2, 1.5, 0.0])
        return np.array(params)

    def make_config_4():
        """Random scatter in a box"""
        np.random.seed(42)
        params = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            ang = np.random.uniform(-15, 15)
            params.extend([x, y, ang])
        return np.array(params)

    def make_config_5():
        """Tighter hexagonal arrangement"""
        centers = [
            (0, 0),
            (d, 0), (d/2, 1.5), (-d/2, 1.5),
            (-d, 0), (-d/2, -1.5), (d/2, -1.5),
            (0, 3.0), (-2.6, 1.5), (2.6, 1.5), (0, -3.0)
        ]
        params = []
        for x, y in centers:
            params.extend([x, y, 0.0])
        return np.array(params)

    initial_configs = [make_config_1(), make_config_2(), make_config_3(), make_config_4(), make_config_5()]

    best_side = float('inf')
    best_params = None
    best_penalty = float('inf')

    np.random.seed(42)

    # Run Nelder-Mead from each initial configuration
    for config_idx, init_config in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break

        try:
            res = scipy.optimize.minimize(
                objective,
                init_config,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            side, penalty = calc_side_and_penalty(res.x)
            
            # We accept if penalty is small enough (no significant overlap)
            if penalty < 1e-3 and side < best_side:
                best_side = side
                best_params = res.x.copy()
                best_penalty = penalty
        except Exception:
            pass

    # If still no valid packing, return a safe fallback
    if best_side == float('inf') or best_params is None:
        # Return a widely spaced configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct output
    inner = []
    for i in range(N):
        x, y, ang = best_params[i*3], best_params[i*3+1], best_params[i*3+2]
        inner.append([float(x), float(y), float(ang)])

    return inner, (0.0, 0.0), float(best_side), 0.0
