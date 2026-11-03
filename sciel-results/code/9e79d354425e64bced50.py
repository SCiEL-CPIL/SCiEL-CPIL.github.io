<<<<<<< SEARCH
def _find_min_enclosing_hex_side(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices.
    Returns (center_x, center_y, side_length).
    """
    # Normals of the outer hexagon
    # The sides are perpendicular to these normals.
    # If the hexagon has a vertex at angle outer_angle_deg, 
    # the normals to the sides are at outer_angle_deg + 30 + 60*k.
    # Wait, standard definition:
    # If vertices are at angles theta + 60k, the edges are between theta and theta+60.
    # The normal to the edge between theta and theta+60 is theta+30.
    # So normals are at outer_angle_deg + 30 + 60*k.
    
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    # LP Variables: [cx, cy, R]
    # Minimize R
    # Constraints:
    # For each vertex v and normal n:
    # (v - c) . n <= R  =>  c . n - v . n >= -R  =>  -c . n - R <= -v . n
    # (c - v) . n <= R  =>  c . n - v . n <= R   =>   c . n - R <=  v . n
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # Constraint 1: -nx*cx - ny*cy - R <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
            # Constraint 2: nx*cx + ny*cy - R <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None
=======
def _find_min_enclosing_hex_side(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices.
    Returns (center_x, center_y, side_length) or None.
    vertices: list of (x,y) tuples or (n,2) array.
    """
    if isinstance(vertices, np.ndarray):
        verts = vertices
    else:
        verts = np.array(vertices)
    
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    n_verts = len(verts)
    # Build A and b efficiently
    A = np.zeros((2 * n_verts + 1, 3))
    b = np.zeros(2 * n_verts + 1)
    
    for i in range(n_verts):
        vx, vy = verts[i]
        for j in range(6):
            nx, ny = normals[j]
            v_dot_n = vx * nx + vy * ny
            row = 2 * i + 2 * j
            A[row] = [-nx, -ny, -1.0]
            b[row] = -v_dot_n
            A[row + 1] = [nx, ny, -1.0]
            b[row + 1] = v_dot_n
    
    A[-1] = [0.0, 0.0, -1.0]
    b[-1] = 0.0
    
    c = np.array([0.0, 0.0, 1.0])
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _min_enclosing_side_for_angles(positions, inner_angle_deg, outer_angle_deg):
    """Compute min outer side length given positions, inner angle, outer angle."""
    verts = _get_hex_vertices_array(positions, inner_angle_deg)
    res = _find_min_enclosing_hex_side(verts, outer_angle_deg)
    if res is None:
        return float('inf')
    return res[2]

def _polish_solution(positions, inner_angle_deg, cx0, cy0, outer_angle0):
    """Polish the solution using Nelder-Mead on (cx, cy, outer_angle)."""
    def objective(params):
        cx, cy, oa = params
        # Compute the required R for this center and angle
        verts = _get_hex_vertices_array(positions, inner_angle_deg)
        normals = []
        for k in range(6):
            a = math.radians(oa + 30 + 60 * k)
            normals.append((math.cos(a), math.sin(a)))
        
        # R must satisfy: for all v, n: |dot(v - c, n)| <= R
        max_R = 0.0
        for nx, ny in normals:
            dots = (verts[:, 0] - cx) * nx + (verts[:, 1] - cy) * ny
            max_R = max(max_R, np.max(np.abs(dots)))
        
        side = max_R / (math.sqrt(3) / 2)
        return side
    
    x0 = np.array([cx0, cy0, outer_angle0])
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.success or res.fun < objective(x0):
            cx, cy, oa = res.x
            return cx, cy, oa, res.fun
        else:
            return cx0, cy0, outer_angle0, objective(x0)
    except:
        return cx0, cy0, outer_angle0, objective(x0)
>>>>>>> REPLACE

