"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell with 4 air holes in a dielectric background.
    This 'diamond' or 'rotated square' hole pattern is known to open large
    TM band gaps in square photonic crystals.
    """
    size = problem["pixels"]
    # Ensure size is 32 as per contract, but handle generically
    if size != 32:
        # Fallback to center rod if size is not 32, though contract says 32
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell

    # Centers of the 4 air holes
    # Offset from the center of the cell (15.5, 15.5)
    offset = 6.0
    center_x = size / 2.0 - 0.5 # 15.5
    center_y = size / 2.0 - 0.5 # 15.5
    
    c1 = (center_x - offset, center_y - offset)
    c2 = (center_x + offset, center_y - offset)
    c3 = (center_x - offset, center_y + offset)
    c4 = (center_x + offset, center_y + offset)
    
    hole_radius = 4.0
    
    cell = []
    for i in range(size):
        row = []
        # Pixel center coordinates
        py = (i + 0.5) - 0.5 # i is 0..31, so y is 0..31. 
        # Wait, coordinate system.
        # Let's use standard image coordinates where (0,0) is top-left.
        # The "center" of the cell is at (15.5, 15.5).
        # Pixel (i,j) has center at (j+0.5, i+0.5) if x is column, y is row.
        # Let's stick to x = j+0.5, y = i+0.5.
        
        py = i + 0.5
        for j in range(size):
            px = j + 0.5
            
            # Check distance to each hole center
            d1 = (px - c1[0])**2 + (py - c1[1])**2
            d2 = (px - c2[0])**2 + (py - c2[1])**2
            d3 = (px - c3[0])**2 + (py - c3[1])**2
            d4 = (px - c4[0])**2 + (py - c4[1])**2
            
            min_d = min(d1, d2, d3, d4)
            
            if min_d < hole_radius**2:
                row.append(0) # Air hole
            else:
                row.append(1) # Dielectric
        
        cell.append(row)
        
    return cell
# EVOLVE-BLOCK-END
