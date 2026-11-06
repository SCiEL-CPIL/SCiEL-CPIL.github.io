import math

def design(problem):
    """
    Returns a 32x32 unit cell with a centered dielectric circle.
    The radius is chosen to optimize the TM 4-5 band gap for eps=12.
    Based on literature and standard photonic crystal design, 
    a radius of approximately 0.35 lattice constants is often optimal 
    for high-contrast materials in this configuration.
    """
    size = problem["pixels"]
    cell = [[0] * size for _ in range(size)]
    
    # Center of the cell
    center = (size - 1) / 2.0
    
    # Radius in pixels. 0.35 * size is the radius in lattice units.
    # We map lattice units [0,1) to pixels [0, size).
    # Distance from center in lattice units: d = sqrt((i-center)^2 + (j-center)^2) / size
    # We want d < 0.35
    radius_frac = 0.35
    radius_px = radius_frac * size
    
    for i in range(size):
        for j in range(size):
            # Calculate distance from center in pixels
            dx = i - center
            dy = j - center
            dist = math.sqrt(dx * dx + dy * dy)
            
            # Check if within radius
            # Using < 0.5 pixel margin for anti-aliasing effect? 
            # Actually, simple thresholding is standard for these tasks.
            if dist < radius_px:
                cell[i][j] = 1
                
    return cell
