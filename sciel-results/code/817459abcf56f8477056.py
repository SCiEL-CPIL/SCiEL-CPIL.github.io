def kick_move(pts, n_kick, max_angle):
    # Pick random points to kick
    idx = np.random.choice(len(pts), n_kick, replace=False)
    # Random axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    # Random angle
    theta = np.random.uniform(-max_angle, max_angle)
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Rodrigues for batch of points
    v = pts[idx]
    cross = np.cross(axis, v) # (n_kick, 3)
    dot = np.dot(axis, v) # (n_kick,)
    
    v_new = v * c + cross * s + np.outer(axis, dot) * (1.0 - c)
    
    return idx, v, v_new
