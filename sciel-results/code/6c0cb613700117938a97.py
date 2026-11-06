"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def _generate_fractal_disk_cell(size, r0, a4, a8):
    """
    Generate a 32x32 cell with a centrally located dielectric disk with fractal boundary.
    r(theta) = r0 + a4*cos(4*theta) + a8*cos(8*theta)
    This maintains inversion symmetry (even harmonics only) and x<->y symmetry.
    """
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            x = (j + 0.5) / size - 0.5
            
            r = math.sqrt(x*x + y*y)
            if r < 1e-10:
                theta = 0.0
            else:
                theta = math.atan2(y, x)
            
            # Fractal boundary: r0 + a4*cos(4*theta) + a8*cos(8*theta)
            r_boundary = r0 + a4 * math.cos(4 * theta) + a8 * math.cos(8 * theta)
            
            if r < r_boundary:
                cell[i][j] = 1
    
    # Enforce exact symmetry
    # Vertical flip: cell[i][j] == cell[size-1-i][j]
    # Horizontal flip: cell[i][j] == cell[i][size-1-j]  
    # Transpose: cell[i][j] == cell[j][i]
    # The fractal disk with even harmonics is symmetric under these, but enforce to be safe.
    
    # First, ensure transpose symmetry
    for i in range(size):
        for j in range(i+1, size):
            val = 1 if (cell[i][j] == 1 and cell[j][i] == 1) else 0
            cell[i][j] = val
            cell[j][i] = val
    
    # Then enforce vertical flip symmetry
    for i in range(size // 2):
        for j in range(size):
            val = 1 if (cell[i][j] == 1 and cell[size-1-i][j] == 1) else 0
            cell[i][j] = val
            cell[size-1-i][j] = val
    
    # Then enforce horizontal flip symmetry
    for i in range(size):
        for j in range(size // 2):
            val = 1 if (cell[i][j] == 1 and cell[i][size-1-j] == 1) else 0
            cell[i][j] = val
            cell[i][size-1-j] = val
    
    return cell

def design(problem):
    """
    Design a 32x32 unit cell with a centrally located dielectric disk with fractal boundary
    defined by r(theta) = r0 + a4*cos(4*theta) + a8*cos(8*theta) to enhance 4-5 TM band gap.
    Uses gradient-free search to optimize harmonic amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    
    # Search space for fractal disk parameters
    # r0: base radius (typically 0.1 to 0.25)
    # a4: amplitude of m=4 harmonic (typically -0.05 to 0.05)
    # a8: amplitude of m=8 harmonic (typically -0.03 to 0.03)
    
    # Initial best configuration
    best_r0 = 0.15
    best_a4 = 0.02
    best_a8 = 0.01
    best_cell = _generate_fractal_disk_cell(size, best_r0, best_a4, best_a8)
    
    # Gradient-free search: random sampling with local refinement
    random.seed(42)
    
    # Phase 1: Broad random sampling
    n_broad = 50
    for _ in range(n_broad):
        if time.time() > deadline - 30:
            break
        
        r0 = random.uniform(0.10, 0.25)
        a4 = random.uniform(-0.05, 0.05)
        a8 = random.uniform(-0.03, 0.03)
        
        cell = _generate_fractal_disk_cell(size, r0, a4, a8)
        
        # Simple heuristic score: prefer moderate dielectric fraction (0.3-0.5)
        # and non-trivial boundary modulation
        frac = sum(sum(row) for row in cell) / (size * size)
        boundary_variation = abs(a4) + abs(a8)
        
        # Heuristic: maximize boundary variation while keeping fraction in sweet spot
        score = boundary_variation * 2 - abs(frac - 0.4) * 3
        
        if score > 0:  # Only update if better than initial
            best_r0, best_a4, best_a8 = r0, a4, a8
            best_cell = cell
    
    # Phase 2: Local refinement around best found
    n_refine = 30
    for _ in range(n_refine):
        if time.time() > deadline - 10:
            break
        
        r0 = best_r0 + random.gauss(0, 0.01)
        a4 = best_a4 + random.gauss(0, 0.01)
        a8 = best_a8 + random.gauss(0, 0.005)
        
        # Clamp to reasonable ranges
        r0 = max(0.10, min(0.25, r0))
        a4 = max(-0.05, min(0.05, a4))
        a8 = max(-0.03, min(0.03, a8))
        
        cell = _generate_fractal_disk_cell(size, r0, a4, a8)
        
        frac = sum(sum(row) for row in cell) / (size * size)
        boundary_variation = abs(a4) + abs(a8)
        score = boundary_variation * 2 - abs(frac - 0.4) * 3
        
        if score > 0:
            best_r0, best_a4, best_a8 = r0, a4, a8
            best_cell = cell
    
    # Final: Use the best configuration found
    best_cell = _generate_fractal_disk_cell(size, best_r0, best_a4, best_a8)
    
    return best_cell

# EVOLVE-BLOCK-END
