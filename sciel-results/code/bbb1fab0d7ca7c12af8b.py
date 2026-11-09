# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    # Precompute unit hexagon vertices for flat-top orientation
    unit_verts = np.array([
        [math.cos(math.radians(i*60)), math.sin(math.radians(i*60))] 
        for i in range(6)
    ])
    
    # Precompute edge normals for the outer hexagon (flat-top)
    edge_normals = np.array([
        [math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] 
        for i in range(6)
    ])
    
    # Precompute base max projections for a unit hexagon at origin
    # base_max_proj[k] = max_{v in unit_verts} dot(v, edge_normals[k])
    base_max_proj = np.array([
        (unit_verts @ edge_normals.T)[:, k].max() for k in range(6)
    ])
    
    # Angles to try
    # We can afford many angles now because the subset selection is O(19 log 19) or O(19)
    # instead of O(50000).
    # Let's try 60 angles (every 1 degree) or even 120.
    angles_deg = list(range(0, 60, 1))
    
    best_L = float('inf')
    best_subset_indices = None
    best_angle = 0.0
    
    for angle_deg in angles_deg:
        if time.time() > deadline:
            break
            
        theta = math.radians(angle_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        
        # Rotate all 19 centers
        M_vals = []
        for x, y in all_centers:
            rx = x * c - y * s
            ry = x * s + y * c
            
            c_vec = np.array([rx, ry])
            c_projs = c_vec @ edge_normals.T
            total_projs = base_max_proj + c_projs
            M_vals.append(total_projs.max())
            
        # Find the 12 centers with smallest M_vals
        # argpartition returns indices of the 12 smallest
        idxs = np.argpartition(M_vals, 11)[:12]
        
        # The cost is the max of these 12
        current_L = 2.0 * np.max([M_vals[i] for i in idxs]) / sqrt3
        
        if current_L < best_L:
            best_L = current_L
            best_subset_indices = idxs
            best_angle = angle_deg

    if best_subset_indices is None:
        # Fallback
        best_subset_indices = np.array(range(12))
        best_L = 3.0 # Rough estimate
        best_angle = 0.0

    # Construct inner hexagons
    theta = math.radians(best_angle)
    c = math.cos(theta)
    s = math.sin(theta)
    
    S = 1.0001
    inner = []
    for i in best_subset_indices:
        x, y = all_centers[i]
        rx = x * c - y * s
        ry = x * s + y * c
        inner.append([rx * S, ry * S, 0.0])
        
    final_L = best_L * S
    
    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
