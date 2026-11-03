def _staggered_seed():
    """Two-row staggered arrangement."""
    s3 = math.sqrt(3)
    params = np.zeros(3 * N)
    # Row 0: 6 hexes, spacing 2.0
    # Row 1: 5 hexes, spacing 2.0, offset by 1.0 in x, s3 in y?
    # Distance between (x, 0) and (x+1, s3): sqrt(1 + 3) = 2.
    # Distance between (x, 0) and (x-1, s3): sqrt(1 + 3) = 2.
    # So offset 1.0 and height s3 works for touching.
    
    x0 = -5.0
    y0 = 0.0
    x1 = -4.0
    y1 = s3
    
    for i in range(6):
        params[3*i] = x0 + 2.0 * i
        params[3*i+1] = y0
        params[3*i+2] = 0.0
    for i in range(5):
        idx = 6 + i
        params[3*idx] = x1 + 2.0 * i
        params[3*idx+1] = y1
        params[3*idx+2] = 0.0
        
    # Center
    cx = np.mean(params[0::3])
    cy = np.mean(params[1::3])
    params[0::3] -= cx
    params[1::3] -= cy
    return params
