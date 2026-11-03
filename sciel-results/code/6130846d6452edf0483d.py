import numpy as np

N = 11

def get_hex_vertices(rotation_deg=0.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + rotation_deg)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def calculate_min_s_and_center(inner_centers, inner_rotations, outer_rotation_deg):
    base_v = get_hex_vertices(0.0)
    theta_rad = np.deg2rad(outer_rotation_deg)
    normals = [theta_rad + np.deg2rad(30 + 60*k) for k in range(6)]
    
    min_s = 0.0
    intervals = []
    
    for n in normals:
        n_vec = np.array([np.cos(n), np.sin(n)])
        max_proj = -np.inf
        min_proj = np.inf
        
        for i, center in enumerate(inner_centers):
            rot_i = np.deg2rad(inner_rotations[i])
            c, s = np.cos(rot_i), np.sin(rot_i)
            R = np.array([[c, -s], [s, c]])
            verts = base_v @ R.T + center
            projs = verts @ n_vec
            max_proj = max(max_proj, np.max(projs))
            min_proj = min(min_proj, np.min(projs))
            
        width = max_proj - min_proj
        s_required = width / np.sqrt(3)
        min_s = max(min_s, s_required)
        
        # Store interval for C dot n
        # C dot n <= min_proj + S * sqrt(3) / 2
        # C dot n >= max_proj - S * sqrt(3) / 2
        # We will update intervals after finding min_s
        intervals.append((min_proj, max_proj, n_vec))
        
    # Now calculate center
    # S is min_s
    S = min_s
    # For each interval, the feasible range for C dot n is:
    # [max_proj - S*sqrt(3)/2, min_proj + S*sqrt(3)/2]
    # Note: min_proj is the min of the inner set, max_proj is the max.
    # Wait, my previous logic:
    # Outer extends [C.n - S*sqrt(3)/2, C.n + S*sqrt(3)/2]
    # Inner extends [min_proj, max_proj]
    # Condition:
    # C.n - S*sqrt(3)/2 <= min_proj  => C.n <= min_proj + S*sqrt(3)/2
    # C.n + S*sqrt(3)/2 >= max_proj  => C.n >= max_proj - S*sqrt(3)/2
    
    # Let's pick the midpoint of the feasible range for the first two normals
    # to determine Cx, Cy
    
    def get_feasible_midpoint(min_p, max_p, n_vec, S):
        lower = max_p - S * np.sqrt(3) / 2
        upper = min_p + S * np.sqrt(3) / 2
        # If lower > upper, it's infeasible, but S was calculated from width, so it should be feasible
        # Width = max_p - min_p. S = Width / sqrt(3).
        # upper - lower = (min_p + S*sqrt(3)/2) - (max_p - S*sqrt(3)/2) = S*sqrt(3) - (max_p - min_p) = 0.
        # So the range is a single point!
        return (lower + upper) / 2
        
    # The range is exactly one point because S is minimal.
    # So C dot n_k is fixed.
    # We can use any two non-parallel normals to solve for C.
    
    # Use normals 0 and 1
    n0 = intervals[0][2]
    n1 = intervals[1][2]
    
    v0 = get_feasible_midpoint(intervals[0][0], intervals[0][1], n0, S)
    v1 = get_feasible_midpoint(intervals[1][0], intervals[1][1], n1, S)
    
    # Solve:
    # Cx * n0x + Cy * n0y = v0
    # Cx * n1x + Cy * n1y = v1
    
    A = np.array([n0, n1]).T
    b = np.array([v0, v1])
    
    try:
        C = np.linalg.solve(A, b)
        cx, cy = C
    except np.linalg.LinAlgError:
        cx, cy = 0.0, 0.0
        
    return S, (cx, cy)

def construct():
    candidates = []
    
    # Candidate 1: Line
    centers = []
    for i in range(N):
        x = (i - (N-1)/2.0) * np.sqrt(3)
        centers.append((x, 0.0))
    rots = [0.0] * N
    candidates.append((centers, rots))
    
    # Candidate 2: Flower + 4 (0, 60, 120, 180)
    centers = [(0.0, 0.0)]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        centers.append((np.sqrt(3) * np.cos(ang), np.sqrt(3) * np.sin(ang)))
    ext_angs = [0, 60, 120, 180]
    for ang_deg in ext_angs:
        ang = np.deg2rad(ang_deg)
        centers.append((2 * np.sqrt(3) * np.cos(ang), 2 * np.sqrt(3) * np.sin(ang)))
    rots = [0.0] * N
    candidates.append((centers, rots))
    
    # Candidate 3: Axial Cluster
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return (x, y)
        
    axial_points = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2)
    ]
    centers = [axial_to_cartesian(q,r) for q,r in axial_points]
    rots = [0.0] * N
    candidates.append((centers, rots))
    
    best_s = np.inf
    best_config = None
    best_center = (0.0, 0.0)
    best_theta = 0.0
    
    for centers, rots in candidates:
        min_s_for_config = np.inf
        best_theta_for_config = 0.0
        best_center_for_config = (0.0, 0.0)
        
        for theta_deg in np.arange(0, 60.01, 0.1):
            s, center = calculate_min_s_and_center(centers, rots, theta_deg)
            if s < min_s_for_config:
                min_s_for_config = s
                best_theta_for_config = theta_deg
                best_center_for_config = center
                
        if min_s_for_config < best_s:
            best_s = min_s_for_config
            best_config = (centers, rots)
            best_center = best_center_for_config
            best_theta = best_theta_for_config
            
    inner_list = []
    for c, r in zip(best_config[0], best_config[1]):
        inner_list.append([c[0], c[1], r])
        
    return inner_list, best_center, best_s, best_theta

