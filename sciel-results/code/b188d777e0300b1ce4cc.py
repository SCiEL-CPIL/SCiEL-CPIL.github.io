<<<<<<< SEARCH
    # Initial guess: hexagonal packing
    # Place 1 center, 6 around, 4 more
    init_params = []
    positions = [(0, 0)]
    for i in range(6):
        angle = i * 60
        r = np.sqrt(3)
        positions.append((r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))))
    # 4 more in second ring
    for i in range(4):
        angle = i * 90 + 30
        r = 2 * np.sqrt(3)
        positions.append((r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))))
    
    for (px, py) in positions:
        init_params.extend([px, py, 0.0])
    init_params.extend([0.0, 0.0, 10.0, 0.0])
    
    best_params = np.array(init_params)
    best_S = _compute_objective(best_params)
=======
    # Initial guess: hexagonal packing
    # Center at (0,0), 6 neighbors at distance sqrt(3) (touching), 9 in second ring at distance 2*sqrt(3)
    # Total 1 + 6 + 4 = 11? No, 1+6=7. Next ring has 12. We need 11.
    # Let's use the first 11 spots in a hex grid.
    init_params = []
    positions = [(0, 0)]
    # Ring 1
    for i in range(6):
        angle = i * 60
        r = np.sqrt(3)
        positions.append((r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))))
    # Ring 2 (first 4)
    for i in range(4):
        angle = i * 60 + 30
        r = 2 * np.sqrt(3)
        positions.append((r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))))
    
    for (px, py) in positions:
        init_params.extend([px, py, 0.0])
    
    # Estimate outer size
    # The furthest points are at distance 2*sqrt(3) + 1 (radius of hex)
    # Hexagon with side S has inradius S*sqrt(3)/2 and circumradius S.
    # If center is 0,0, and farthest point is D, we need S such that D <= S*sqrt(3)/2 (if aligned) or S (if vertex aligned).
    # Let's start with S = 6.0
    init_params.extend([0.0, 0.0, 6.0, 0.0])
    
    best_params = np.array(init_params)
    best_S = _compute_objective(best_params)
>>>>>>> REPLACE

