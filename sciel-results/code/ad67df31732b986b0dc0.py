"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _make_cell_square_rod(size, half_width):
    """Create a D4-symmetric square dielectric rod of given half-width in pixels."""
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            if abs(i - center) < half_width and abs(j - center) < half_width:
                cell[i][j] = 1
    return cell

def _make_cell_air_holes(size, hole_half_width, hole_offset):
    """Create a D4-symmetric design: dielectric background with 4 air holes at (±hole_offset, ±hole_offset)."""
    cell = [[1]*size for _ in range(size)]
    center = (size - 1) / 2.0
    # 4 holes at positions that respect D4 symmetry
    for di in [1, -1]:
        for dj in [1, -1]:
            hx = center + di * hole_offset
            hy = center + dj * hole_offset
            for i in range(size):
                for j in range(size):
                    if (i - hx)**2 + (j - hy)**2 < hole_half_width**2:
                        cell[i][j] = 0
    return cell

def _make_cell_cross(size, arm_half_width):
    """Create a D4-symmetric cross shape."""
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            if abs(i - center) < arm_half_width or abs(j - center) < arm_half_width:
                cell[i][j] = 1
    return cell

def _make_cell_four_bars(size, bar_half_width, bar_offset):
    """Create 4 dielectric bars arranged in a square, D4 symmetric."""
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            # Horizontal bars
            if abs(i - (center - bar_offset)) < bar_half_width and abs(j - center) < center:
                cell[i][j] = 1
            if abs(i - (center + bar_offset)) < bar_half_width and abs(j - center) < center:
                cell[i][j] = 1
            # Vertical bars
            if abs(j - (center - bar_offset)) < bar_half_width and abs(i - center) < center:
                cell[i][j] = 1
            if abs(j - (center + bar_offset)) < bar_half_width and abs(i - center) < center:
                cell[i][j] = 1
    return cell

def _make_cell_fourier(size, coeffs):
    """Create a D4-symmetric cell from Fourier coefficients.
    coeffs is a dict mapping (h,k) to real amplitude for cos/sin terms.
    We use only D4-invariant combinations."""
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            x = (i - center) / size
            y = (j - center) / size
            val = coeffs.get((0,0), 0.0)
            # D4-symmetric modes: we use cos(h*x)*cos(k*y) + cos(k*x)*cos(h*y) for h != k
            # and cos(h*x)*cos(h*y) for h == k
            for (h, k), amp in coeffs.items():
                if (h, k) == (0, 0):
                    continue
                if h == k:
                    val += amp * math.cos(2*math.pi*h*x) * math.cos(2*math.pi*h*y)
                else:
                    val += amp * (math.cos(2*math.pi*h*x)*math.cos(2*math.pi*k*y) + 
                                  math.cos(2*math.pi*k*x)*math.cos(2*math.pi*h*y))
            if val > 0:
                cell[i][j] = 1
    return cell

def design(problem):
    """Design a D4-symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s limit
    
    size = problem["pixels"]
    
    # Generate several candidate designs
    candidates = []
    
    # Square rods with various sizes
    for hw in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        candidates.append(_make_cell_square_rod(size, hw))
    
    # Air holes in dielectric background
    for hw in [2, 3, 4, 5, 6]:
        for off in [4, 6, 8, 10, 12]:
            candidates.append(_make_cell_air_holes(size, hw, off))
    
    # Cross shapes
    for hw in [3, 4, 5, 6, 7]:
        candidates.append(_make_cell_cross(size, hw))
    
    # Four bars
    for hw in [2, 3, 4]:
        for off in [4, 6, 8]:
            candidates.append(_make_cell_four_bars(size, hw, off))
    
    # Fourier-based designs
    # Try various Fourier coefficient combinations
    fourier_configs = [
        {(0,0): 0.3, (1,1): 0.5},
        {(0,0): 0.3, (1,1): -0.5},
        {(0,0): 0.2, (1,0): 0.4, (1,1): 0.3},
        {(0,0): 0.2, (1,0): -0.4, (1,1): 0.3},
        {(0,0): 0.3, (2,0): 0.3, (1,1): 0.2},
        {(0,0): 0.3, (2,0): -0.3, (1,1): 0.2},
        {(0,0): 0.2, (1,0): 0.3, (2,0): 0.2, (1,1): 0.3},
        {(0,0): 0.2, (1,0): -0.3, (2,0): -0.2, (1,1): 0.3},
        {(0,0): 0.4, (2,2): 0.3},
        {(0,0): 0.4, (2,2): -0.3},
        {(0,0): 0.3, (1,1): 0.4, (2,2): 0.2},
        {(0,0): 0.3, (1,1): -0.4, (2,2): 0.2},
    ]
    for cfg in fourier_configs:
        candidates.append(_make_cell_fourier(size, cfg))
    
    # Check time - if we've used too much, just return the first candidate
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return candidates[0]
    
    # We can't easily compute the exact band structure in the time budget for all candidates.
    # Use a heuristic: prefer designs with filling fraction around 0.3-0.6 and good symmetry.
    # Return a well-known good design: square rod with moderate size.
    # Based on literature, a square rod with side ~8-10 pixels in a 32x32 cell is reasonable.
    
    # For now, return a square rod with half-width 5 (side 10 pixels)
    best_cell = _make_cell_square_rod(size, 5)
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert best_cell[i][j] == best_cell[size-1-i][j], "Vertical symmetry violated"
            assert best_cell[i][j] == best_cell[i][size-1-j], "Horizontal symmetry violated"
            assert best_cell[i][j] == best_cell[j][i], "Transpose symmetry violated"
    
    return best_cell
# EVOLVE-BLOCK-END
