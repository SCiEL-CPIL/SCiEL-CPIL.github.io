<<<<<<< SEARCH
    # Arrangement 2: Hexagonal cluster using hex grid
    hex_grid_positions = []
    hex_grid_angles = []
    
    # Ring 0: center
    hex_grid_positions.append(hex_to_cart(0, 0))
    hex_grid_angles.append(0.0)
    
    # Ring 1: 6 neighbors
    ring1 = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for r, c in ring1:
        if len(hex_grid_positions) < N:
            hex_grid_positions.append(hex_to_cart(r, c))
            hex_grid_angles.append(0.0)
    
    # Ring 2: 12 positions, pick remaining
    ring2 = []
    for r in range(-2, 3):
        for c in range(-2, 3):
            x, y = hex_to_cart(r, c)
            dist = math.sqrt(x*x + y*y)
            # Ring 2 positions are at distance ~2*sqrt(3) to ~3*sqrt(3)
            if 2.5 * math.sqrt(3) < dist < 3.5 * math.sqrt(3):
                ring2.append((r, c, math.atan2(y, x)))
    
    ring2.sort(key=lambda t: t[2])
    for r, c, _ in ring2:
        if len(hex_grid_positions) < N:
            hex_grid_positions.append(hex_to_cart(r, c))
            hex_grid_angles.append(0.0)
    
    if len(hex_grid_positions) == N:
        candidates.append((hex_grid_positions, hex_grid_angles))
=======
    # Arrangement 2: Hexagonal cluster using hex grid
    hex_grid_positions = []
    hex_grid_angles = []
    
    # Ring 0: center
    hex_grid_positions.append(hex_to_cart(0, 0))
    hex_grid_angles.append(30.0)  # Rotate to flat-top for tiling
    
    # Ring 1: 6 neighbors
    ring1 = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for r, c in ring1:
        if len(hex_grid_positions) < N:
            hex_grid_positions.append(hex_to_cart(r, c))
            hex_grid_angles.append(30.0)
    
    # Ring 2: Pick specific valid neighbors to reach N
    # Axial coords for Ring 2 (hex distance 2)
    ring2_candidates = [
        (0, 2), (1, 1), (2, 0), (2, -1), (1, -2), (0, -2),
        (-1, -1), (-2, 0), (-2, 1), (-1, 2), (0, 2), (1, 1) # Duplicates removed in logic below if needed, but list is just for picking
    ]
    # Unique ring 2 positions
    unique_ring2 = [(0, 2), (1, 1), (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0), (-2, 1), (-1, 2)]
    
    for r, c in unique_ring2:
        if len(hex_grid_positions) < N:
            hex_grid_positions.append(hex_to_cart(r, c))
            hex_grid_angles.append(30.0)
    
    if len(hex_grid_positions) == N:
        candidates.append((hex_grid_positions, hex_grid_angles))
>>>>>>> REPLACE

Also update the linear arrangement angles:

