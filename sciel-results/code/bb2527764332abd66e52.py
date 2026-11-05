import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    
    def make_symmetric_cell(fill_func):
        """Generate a 32x32 cell with full square symmetry."""
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                v = 1 if fill_func(x, y) else 0
                # Enforce all 8 symmetry operations
                cell[i][j] = v
                cell[size-1-i][j] = v
                cell[i][size-1-j] = v
                cell[size-1-i][size-1-j] = v
                cell[j][i] = v
                cell[j][size-1-i] = v
                cell[size-1-j][i] = v
                cell[size-1-j][size-1-j] = v
        return cell
    
    # Based on photonic crystal literature for 2D square lattice,
    # TM polarization, dielectric rods in air (eps=12):
    # Optimal radius is approximately 0.38 of lattice constant
    # This gives a significant gap between bands 4 and 5
    
    r = 0.38
    
    def fill(x, y):
        return (x*x + y*y) < r*r
    
    return make_symmetric_cell(fill)

# EVOLVE-BLOCK-END
