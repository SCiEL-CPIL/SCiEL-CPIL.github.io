"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """A four-rod lattice pattern centered at (±0.25, ±0.25) with radius 0.15."""
    size = problem["pixels"]
    cell = [[0] * size for _ in range(size)]
    
    # Define the centers of the four rods
    # In a 32x32 grid, center is at (15.5, 15.5)
    # Offset 0.25 * 32 = 8 pixels
    # So centers are at 15.5 ± 8 => 7.5 and 23.5
    # Pixel coordinates are i+0.5, so we check distance to (7.5, 7.5), etc.
    
    centers = [
        (7.5, 7.5),
        (7.5, 23.5),
        (23.5, 7.5),
        (23.5, 23.5)
    ]
    
    radius = 0.15 * size # 0.15 * 32 = 4.8 pixels
    
    for i in range(size):
        for j in range(size):
            # Pixel center coordinates
            x = i + 0.5
            y = j + 0.5
            
            is_dielectric = False
            for cx, cy in centers:
                dist_sq = (x - cx)**2 + (y - cy)**2
                if dist_sq < radius**2:
                    is_dielectric = True
                    break
            
            if is_dielectric:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
                
    return cell
# EVOLVE-BLOCK-END
