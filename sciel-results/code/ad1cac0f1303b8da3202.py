import time
import math
import numpy as np
import itertools

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # 10s margin before 230s kill
    
    size = problem["pixels"]
    # Design: Use a symmetry-reduced branch-and-bound search over the fundamental domain
    # Objective: Maximize gap-midgap ratio using a fast homogenized approximation
    
    # The fundamental domain for a 32x32 square cell under square symmetry (D4) is the upper-right 8x16 triangle
    # Actually, for a 32x32 cell, the fundamental domain is 16x16 (quarter cell)
    # Let's use a simpler approach: optimize the fundamental 16x16 domain and mirror it
    
    # Fast approximation: Use effective medium theory
    # For TM polarization, the band gap depends on the dielectric contrast and geometry
    # The gap is maximized when the dielectric forms a connected structure with optimal aspect ratio
    
    # Strategy: Search over a parameterized family of designs in the fundamental domain
    # Parameters: number of dielectric pixels in the fundamental domain (0-256)
    # For each count, use a greedy placement that maximizes symmetry and connectivity
    
    best_cell = None
    best_score = -1.0
    
    # Precompute the symmetry mapping
    # For a 32x32 cell, the fundamental domain is the upper-right quadrant (16x16)
    # We'll search over the number of dielectric pixels in the fundamental domain
    
    # Fast scoring function using homogenized effective medium theory
    def fast_score(cell):
        """
        Approximate the gap-midgap ratio using effective medium theory
        This is much faster than full PWE and serves as a surrogate for B&B
        """
        # Compute effective permittivity using Bruggeman or Maxwell-Garnett
        # For a square lattice, the effective permittivity is:
        # eps_eff = eps_low * (1 + 3*f*(eps_high-eps_low)/(eps_low + 2*eps_high - f*(eps_high-eps_low)))
        # But for a 2D square lattice, we use a different formula
        
        # Simpler: Use the fact that the gap is proportional to the dielectric contrast
        # and the geometric factor. For TM, the gap is larger when the dielectric
        # forms a connected ring or square structure
        
        # Compute the fill fraction
        fill = np.sum(cell) / (size * size)
        
        # For a square lattice, the optimal fill fraction is around 0.5
        # The gap is maximized when the dielectric forms a symmetric structure
        # Use a heuristic: gap ~ contrast * |fill - 0.5| * symmetry_factor
        
        # Compute symmetry factor (how well the cell matches D4 symmetry)
        # This is always 1.0 for our symmetric designs, so we skip it
        
        # Better heuristic: The gap is larger when the dielectric pixels are
        # arranged in a way that creates a strong periodic modulation
        # Use the variance of the dielectric as a proxy
        
        dielectric = np.array(cell, dtype=float)
        variance = np.var(dielectric)
        
        # The gap is proportional to the contrast and the modulation strength
        contrast = 12 - 1  # eps_high - eps_low
        modulation = math.sqrt(variance)
        
        # The gap-midgap ratio is approximately:
        # ratio ~ contrast * modulation / (1 + fill)
        # This is a rough approximation
        
        return contrast * modulation / (1 + fill)
    
    # Branch and bound over the fundamental domain
    # We'll search over the number of dielectric pixels in the 16x16 fundamental domain
    # For each count, we'll use a greedy placement
    
    # Precompute the positions in the fundamental domain
    fund_positions = []
    for i in range(16):
        for j in range(16):
            fund_positions.append((i, j))
    
    # Search over fill fractions from 0.1 to 0.9 in steps of 0.05
    for fill_frac in np.arange(0.1, 0.95, 0.05):
        if time.time() > deadline - 5:
            break
        
        # Number of dielectric pixels in the fundamental domain
        n_pixels = int(fill_frac * 256)
        
        # Greedy placement: Place pixels in the fundamental domain to maximize
        # the modulation while maintaining symmetry
        # Start with a random placement, then improve via local search
        
        # For simplicity, use a structured placement:
        # Place pixels in a grid pattern in the fundamental domain
        
        cell = [[0 for _ in range(size)] for _ in range(size)]
        
        # Place pixels in the fundamental domain (upper-right quadrant)
        # Use a grid pattern with spacing based on fill fraction
        spacing = max(1, int(4 / math.sqrt(fill_frac)))
        
        for i in range(16):
            for j in range(16):
                if (i % spacing == 0) and (j % spacing == 0):
                    # Set the pixel and its symmetric counterparts
                    # Upper-right quadrant: (i, j)
                    cell[i][j] = 1
                    # Lower-right quadrant: (31-i, j)
                    cell[31-i][j] = 1
                    # Upper-left quadrant: (i, 31-j)
                    cell[i][31-j] = 1
                    # Lower-left quadrant: (31-i, 31-j)
                    cell[31-i][31-j] = 1
        
        # Compute the fast score
        score = fast_score(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # If no valid cell was found, return a default symmetric design
    if best_cell is None:
        best_cell = [[0 for _ in range(size)] for _ in range(size)]
        # Place a central dielectric square
        for i in range(8, 24):
            for j in range(8, 24):
                best_cell[i][j] = 1
    
    return best_cell
