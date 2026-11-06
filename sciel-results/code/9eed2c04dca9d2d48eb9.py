"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a square lattice photonic crystal unit cell with air holes in a dielectric background.
    Uses a central diamond hole and four corner diamond holes to maximize the TM band gap.
    """
    size = problem["pixels"]
    cell = []
    
    # Parameters for the diamond holes
    # Diamond defined by |x| + |y| < r
    r_center = 0.30
    r_corner = 0.30
    
    for i in range(size):
        row = []
        # y coordinate of the center of pixel i
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            # x coordinate of the center of pixel j
            x = (j + 0.5) / size - 0.5
            
            is_air = False
            
            # Check central diamond
            if abs(x) + abs(y) < r_center:
                is_air = True
            
            # Check corner diamonds
            # Corners are at (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)
            # Due to symmetry, we can just check distance to the nearest corner
            # But explicitly checking all 4 is safer and fast enough for 32x32
            
            # Corner 1: (0.5, 0.5)
            if abs(x - 0.5) + abs(y - 0.5) < r_corner:
                is_air = True
            # Corner 2: (-0.5, 0.5)
            if abs(x + 0.5) + abs(y - 0.5) < r_corner:
                is_air = True
            # Corner 3: (0.5, -0.5)
            if abs(x - 0.5) + abs(y + 0.5) < r_corner:
                is_air = True
            # Corner 4: (-0.5, -0.5)
            if abs(x + 0.5) + abs(y + 0.5) < r_corner:
                is_air = True
            
            # 1 = dielectric, 0 = air
            row.append(0 if is_air else 1)
        cell.append(row)
        
    return cell
# EVOLVE-BLOCK-END
