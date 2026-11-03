    # 3. Structured: Vertices + Edge midpoints + Center
    struct_pts = []
    struct_pts.append(A + np.array([0.05, 0.01]))
    struct_pts.append(B + np.array([-0.05, 0.01]))
    struct_pts.append(C + np.array([0.0, -0.05]))
    # Midpoints
    struct_pts.append((A+B)/2 + np.array([0, 0.02]))
    struct_pts.append((B+C)/2 + np.array([-0.02, 0]))
    struct_pts.append((C+A)/2 + np.array([0.02, 0]))
    # Center
    center = (A+B+C)/3
    struct_pts.append(center)
    # Extra points
    struct_pts.append(center + np.array([0.1, 0.05]))
    struct_pts.append(center + np.array([-0.1, 0.05]))
    struct_pts.append(center + np.array([0, -0.1]))
    # Ensure inside
    inits.append(np.array([_strict_project_to_triangle(p) for p in struct_pts]))
