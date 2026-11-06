# EVOLVE-BLOCK-START
import time

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    
    # For TM polarization in a square lattice, the standard configuration is
    # dielectric rods in an air background.
    # The optimal radius for epsilon=12 is approximately 0.30-0.35 of the lattice constant.
    # Lattice constant a = 32 pixels.
    # Target radius ~ 9.6 to 11.2 pixels.
    
    # We construct a circular rod centered at the cell center.
    # To ensure strict C4v symmetry for an even-sized grid (32x32),
    # we use a distance metric that is symmetric with respect to the center.
    # Pixel i is at distance min(i, size-1-i) + 0.5 from the center line.
    
    def generate_rod(radius):
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Symmetric distance from center
                # Center is between 15 and 16.
                # Dist for i=15 is 0.5, i=16 is 0.5.
                # Dist for i=0 is 15.5, i=31 is 15.5.
                di = min(i, size - 1 - i) + 0.5
                dj = min(j, size - 1 - j) + 0.5
                
                # Check if pixel is inside the circle of given radius
                # Using squared distance to avoid sqrt
                if di * di + dj * dj < radius * radius:
                    cell[i][j] = 1
        return cell

    # Try a few radii around the optimal theoretical value.
    # We pick R=10 as a strong candidate.
    # We could also try R=9, 11, but R=10 is a safe middle ground.
    
    # Let's create candidates
    candidates = []
    for r in [9, 10, 11]:
        candidates.append(generate_rod(r))
    
    # Verify symmetry of the chosen candidate (R=10)
    cell = candidates[1] # R=10
    
    # Double check symmetry just in case
    if not _check_symmetry(cell, size):
        # Fallback to a simple symmetric pattern if something went wrong
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                di = min(i, size - 1 - i) + 0.5
                dj = min(j, size - 1 - j) + 0.5
                if di * di + dj * dj < 100: # R=10
                    cell[i][j] = 1
        # If even this fails (unlikely), return a cross
        if not _check_symmetry(cell, size):
             cell = [[0] * size for _ in range(size)]
             for i in range(size):
                 for j in range(size):
                     if min(i, size-1-i) < 4 or min(j, size-1-j) < 4:
                         cell[i][j] = 1

    return cell

def _check_symmetry(cell, size):
    """Check if cell has square symmetry: 180-degree rotation, transpose, and reflections."""
    # Check 180-degree rotation symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][size - 1 - j]:
                return False
    
    # Check transpose symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                return False
    
    # Check horizontal reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][j]:
                return False
    
    # Check vertical reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size - 1 - j]:
                return False
    
    return True
# EVOLVE-BLOCK-END
