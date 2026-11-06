import time
import math

def design(problem):
    """
    Design a symmetric 32x32 pixel unit cell to maximize the TM band gap 
    between the 4th and 5th bands.
    
    Strategy:
    1. Start with a square pillar, which is known to produce larger gaps 
       than circular rods for higher-order TM bands.
    2. Apply a 'chamfer' or 'rounding' to the corners of the square to 
       optimize the Fourier coefficients, which can further widen the gap.
    3. Ensure the cell is invariant under the square symmetry group (C4v).
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Initialize cell with air (0)
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Define the core square pillar
    # A square pillar of size approx 0.5 * lattice constant is a good starting point.
    # In a 32x32 grid, center is at 16.
    # Let's try a square of half-width w.
    # Based on literature and the hint about square pillars, a half-width of 8-10 pixels (25-30% of width) 
    # often works well for the 4-5 gap.
    
    # We will use a "squircle" or chamfered square.
    # Condition for being inside:
    # x^p + y^p <= R^p
    # For p=2 it's a circle, p=4 is a squircle, p->inf is a square.
    # The hint says "square pillars" improved the score. So we want p large.
    # However, sharp corners can cause issues. A "chamfered" square is a good compromise.
    
    # Let's define the boundary using a superellipse equation:
    # |x|^n + |y|^n = 1
    # We want to tune n and the radius R.
    
    # From the knowledge: "Replacing circular rods with square pillars... Measured: 2.732 -> 3.511."
    # This suggests a square (n->inf) is better than a circle (n=2).
    # Let's try a high-n superellipse, effectively a square with slightly rounded corners.
    
    n = 6.0  # High exponent approximates a square but with smooth corners
    # Radius in pixels. Center is at (15.5, 15.5) for 0-indexed 32x32 if we consider pixel centers.
    # Actually, pixel (i,j) center is at (i+0.5, j+0.5).
    # Cell center is at (16, 16) in continuous coordinates if we map [0,32]x[0,32].
    # Let's use coordinates centered at 0.
    # x = (i + 0.5) / 32 - 0.5
    
    # Let's try a radius such that the square side is roughly 14-16 pixels wide.
    # If side is 14, half-side is 7.
    # Let's iterate through a few candidate radii and exponents if time permits, 
    # but for robustness, we'll pick a strong static candidate.
    
    # Candidate: Superellipse n=4 (squircle) or n=8 (near square).
    # Let's try n=8 and a radius R.
    # If R=1.0 in normalized coords, the max extent is 1.0.
    # Pixel i=15 (center-ish) has x = 15.5/32 - 0.5 = -0.0156
    # Pixel i=10 has x = 10.5/32 - 0.5 = -0.1718
    # Pixel i=5 has x = 5.5/32 - 0.5 = -0.3281
    # Pixel i=2 has x = 2.5/32 - 0.5 = -0.4218
    
    # Let's aim for a "square" that is about 10x10 pixels in the center.
    # Half-width 5 pixels.
    # x_max = 5.5/32 - 0.5? No.
    # If we want pixels i=11..20 to be filled (10 pixels), 
    # i=11 -> x = 11.5/32 - 0.5 = -0.1406
    # i=20 -> x = 20.5/32 - 0.5 = 0.1406
    # So |x| < 0.14.
    # For a square |x|<=a, |y|<=b.
    # Let's just use a square mask with chamfered corners.
    
    # Square half-width in pixels
    half_w = 5.5 # Covers pixels 10 to 21 approx?
    # Let's be precise.
    # Center is 16.
    # Pixels 16-5.5 = 10.5 to 16+5.5 = 21.5.
    # So pixels 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21?
    # Let's just use the superellipse formula which handles this smoothly.
    
    R = 0.16 # Normalized radius. 0.5 is half the cell. 0.16 is about 10% of cell width? 
             # 0.16 * 32 = 5.12 pixels. So radius ~5 pixels. Side ~10 pixels.
    n_exp = 10.0 # Very close to a square.
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Superellipse: |x|^n + |y|^n <= R^n
            # To avoid pow issues with 0, we can just compute
            val = abs(x)**n_exp + abs(y)**n_exp
            if val <= R**n_exp:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
                
    # The superellipse with high n approximates a square.
    # To ensure we are in the "square pillar" family which showed improvement,
    # this is a good choice.
    
    # Symmetry check:
    # The function is symmetric under x->-x, y->-y, and x<->y.
    # So cell == cell[::-1] and cell == cell.T should hold.
    
    return cell
