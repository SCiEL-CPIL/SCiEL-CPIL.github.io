def design(problem):
    """
    Designs a 32x32 pixel unit cell for a 2D square-lattice photonic crystal.
    Uses a 4-rod configuration (one rod per quadrant) to exploit sub-periodicity
    and maximize the TM band gap between bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    
    # Optimal radius for dielectric rods (epsilon=12) in air square lattice.
    # Lattice constant for the 4-rod arrangement is 16 pixels.
    # Filling factor ~0.5 is typically optimal for TM gaps.
    # Area = pi * R^2. Total Area = 4 * pi * R^2. Cell Area = 32*32 = 1024.
    # 4 * pi * R^2 / 1024 = 0.5 => R^2 = 1024 / (4*pi) = 81.48 => R ~ 9.0?
    # Wait.
    # If L=16, Area of primitive cell = 256.
    # Filling factor f = pi * R^2 / 256.
    # If f = 0.5, R^2 = 256 / (2*pi) = 40.7 => R ~ 6.4.
    # So R=6.5 is consistent with 50% filling factor in the 16x16 sub-cell.
    R = 6.5
    
    centers = [
        (7.5, 7.5),
        (7.5, 23.5),
        (23.5, 7.5),
        (23.5, 23.5)
    ]
    
    cell = [[0 for _ in range(size)] for _ in range(size)]
    R_sq = R * R
    
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < R_sq:
                    cell[i][j] = 1
                    break
            # else remains 0
            
    return cell
