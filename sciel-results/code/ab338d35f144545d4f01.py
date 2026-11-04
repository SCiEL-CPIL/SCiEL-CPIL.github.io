"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Hexagonal packing: 6 rows with 4, 3, 4, 3, 4, 3 circles respectively
    # This gives 4+3+4+3+4+3 = 21 circles
    # Centers are placed in a hexagonal lattice pattern.
    
    # Row counts
    row_counts = [4, 3, 4, 3, 4, 3]
    
    # Calculate the radius r such that the bounding rectangle has perimeter at most 4
    # For this hexagonal packing:
    # - Width W = 8r (from x=r to x=7r for rows with 4 circles, plus r on each side)
    #   Actually, let's recalculate carefully.
    
    # Let r be the radius of each circle.
    # Rows with 4 circles: centers at x = r, 3r, 5r, 7r (spacing 2r)
    # Rows with 3 circles: centers at x = 2r, 4r, 6r (spacing 2r, offset by r)
    # Vertical spacing between rows: sqrt(3)*r
    
    # Y coordinates of row centers (bottom to top):
    # Row 1: y = r
    # Row 2: y = r + sqrt(3)*r
    # Row 3: y = r + 2*sqrt(3)*r
    # Row 4: y = r + 3*sqrt(3)*r
    # Row 5: y = r + 4*sqrt(3)*r
    # Row 6: y = r + 5*sqrt(3)*r
    
    # So min y = r, max y = r + 5*sqrt(3)*r
    # Height H = 5*sqrt(3)*r + r = r*(5*sqrt(3) + 1)
    # Wait, max y - min y = (r + 5*sqrt(3)*r) - r = 5*sqrt(3)*r
    # But the circles extend from y_center - r to y_center + r
    # So min y of any circle = r - r = 0 (for row 1)
    # Max y of any circle = (r + 5*sqrt(3)*r) + r = r*(2 + 5*sqrt(3))
    # Height H = r*(2 + 5*sqrt(3))
    
    # Min x of any circle: for rows with 4 circles, min x_center = r, so min x = r - r = 0
    # Max x of any circle: for rows with 4 circles, max x_center = 7r, so max x = 7r + r = 8r
    # Width W = 8r
    
    # Perimeter = 2*(W + H) = 2*(8r + r*(2 + 5*sqrt(3))) = 2*r*(8 + 2 + 5*sqrt(3)) = 2*r*(10 + 5*sqrt(3))
    # We want perimeter <= 4, so 2*r*(10 + 5*sqrt(3)) <= 4
    # r <= 2 / (10 + 5*sqrt(3))
    
    r = 2.0 / (10 + 5 * np.sqrt(3))
    
    # Build the list of circle centers and radii
    circles = []
    y_coords = [r + i * np.sqrt(3) * r for i in range(6)]
    
    for i, count in enumerate(row_counts):
        y = y_coords[i]
        if count == 4:
            # Centers at x = r, 3r, 5r, 7r
            x_coords = [r + 2 * j * r for j in range(4)]
        else:  # count == 3
            # Centers at x = 2r, 4r, 6r
            x_coords = [2 * r + 2 * j * r for j in range(3)]
        
        for x in x_coords:
            circles.append([x, y, r])
    
    return np.array(circles[:N])
# EVOLVE-BLOCK-END
