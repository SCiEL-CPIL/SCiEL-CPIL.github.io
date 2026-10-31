def generate_greedy_init():
    """
    Generate points greedily to maximize the minimum triangle area.
    """
    inits = []
    
    # Variant 1: Start with vertices
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2]
    ]
    
    n_grid = 25
    u_vals = np.linspace(0, 1, n_grid)
    v_vals = np.linspace(0, 1, n_grid)
    
    for _ in range(8):
        best_val = -1
        best_pt = None
        
        for u in u_vals:
            for v in v_vals:
                if u + v > 1:
                    continue
                p = [u, v]
                min_area = 1.0
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        p1 = pts[i]
                        p2 = pts[j]
                        u1, v1, w1 = p1[0], p1[1], 1-p1[0]-p1[1]
                        u2, v2, w2 = p2[0], p2[1], 1-p2[0]-p2[1]
                        up, vp, wp = u, v, 1-u-v
                        det = up*(v1*w2 - w1*v2) - vp*(u1*w2 - w1*u2) + wp*(u1*v2 - w1*v1)
                        area = abs(det)
                        if area < min_area:
                            min_area = area
                if min_area > best_val:
                    best_val = min_area
                    best_pt = p
        if best_pt is not None:
            pts.append(best_pt)
            
    bary = np.array(pts)
    bary = clamp_bary(bary)
    inits.append(bary)
    
    # Variant 2: Start with random 3 points
    rng = np.random.default_rng(99)
    for _ in range(3):
        pts = []
        for _ in range(3):
            u = rng.random()
            v = rng.random() * (1 - u)
            pts.append([u, v])
            
        for _ in range(8):
            best_val = -1
            best_pt = None
            for u in u_vals:
                for v in v_vals:
                    if u + v > 1:
                        continue
                    p = [u, v]
                    min_area = 1.0
                    for i in range(len(pts)):
                        for j in range(i+1, len(pts)):
                            p1 = pts[i]
                            p2 = pts[j]
                            u1, v1, w1 = p1[0], p1[1], 1-p1[0]-p1[1]
                            u2, v2, w2 = p2[0], p2[1], 1-p2[0]-p2[1]
                            up, vp, wp = u, v, 1-u-v
                            det = up*(v1*w2 - w1*v2) - vp*(u1*w2 - w1*u2) + wp*(u1*v2 - w1*v1)
                            area = abs(det)
                            if area < min_area:
                                min_area = area
                    if min_area > best_val:
                        best_val = min_area
                        best_pt = p
            if best_pt is not None:
                pts.append(best_pt)
                
        bary = np.array(pts)
        bary = clamp_bary(bary)
        inits.append(bary)
        
    return inits
