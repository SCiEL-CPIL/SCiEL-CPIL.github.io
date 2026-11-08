<<<<<<< SEARCH
    sqrt3 = math.sqrt(3)
    
    # Place hexagons in a hexagonal lattice
    # Using axial coordinates (q, r), convert to pixel coordinates
    # For flat-top hexagons with side length 1:
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    
    inner = []
    for q, r_coord in positions:
        x = 1.5 * q
        y = sqrt3 * (r_coord + 0.5 * q)
        inner.append([x, y, 0.0])
=======
    sqrt3 = math.sqrt(3)
    
    # Place hexagons in a hexagonal lattice
    # Using axial coordinates (q, r), convert to pixel coordinates
    # For flat-top hexagons with side length 1:
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    
    # Scale factor to ensure strict non-overlap (avoid floating point boundary issues)
    scale = 1.0 + 1e-6
    
    inner = []
    for q, r_coord in positions:
        x = 1.5 * q * scale
        y = sqrt3 * (r_coord + 0.5 * q) * scale
        inner.append([x, y, 0.0])
>>>>>>> REPLACE
