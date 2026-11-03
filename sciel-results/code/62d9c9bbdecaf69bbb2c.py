<<<<<<< SEARCH
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
=======
    inner = []
    for x, y in pts_scaled:
        # Ensure finite values
        if not (np.isfinite(x) and np.isfinite(y)):
            x, y = 0.0, 0.0
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
>>>>>>> REPLACE
