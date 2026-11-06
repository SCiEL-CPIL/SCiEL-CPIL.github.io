"""Rewritten design: 2x2 rod array with D4 symmetry, tuned via a fast surrogate."""
import math
import time

def _make_cell_2x2(r, cx):
    """Create a 32x32 D4-symmetric cell with a 2x2 array of rods.
    r: rod radius (in lattice units, i.e. fraction of cell)
    cx: center offset of each rod from cell center (fraction of cell)
    Returns 32x32 list of 0/1.
    """
    size = 32
    cell = [[0]*size for _ in range(size)]
    # The 2x2 array has rods at (+/-cx, +cx), (-cx, +cx), (+cx, -cx), (-cx, -cx)
    # But due to D4 symmetry about cell center, we need to be careful.
    # Actually for a square lattice PC, the unit cell contains one rod typically.
    # A 2x2 array means 4 rods per cell, which is a supercell concept but here
    # we're in a single 32x32 cell. Let me reconsider.
    # 
    # For a square lattice photonic crystal with one rod per unit cell, the rod is
    # at the center. The "2x2 array" insight likely means placing 4 smaller rods
    # in the cell, or it could mean the effective Bragg scattering is enhanced.
    #
    # Actually, re-reading: "Switching from a single central rod to a 2x2 array of rods"
    # This likely means 4 rods arranged in a 2x2 pattern within the unit cell.
    # For D4 symmetry about the cell center, the 4 rods should be at positions
    # that are symmetric under 90-degree rotation and reflections.
    #
    # A 2x2 array centered at the cell center would have rods at:
    # (+d, +d), (-d, +d), (+d, -d), (-d, -d) where d is the offset.
    # This is D4 symmetric.
    
    d = cx  # offset from center for each rod
    
    for i in range(size):
        for j in range(size):
            # Pixel center in cell coordinates (0 to 1)
            px = (i + 0.5) / size
            py = (j + 0.5) / size
            # Convert to centered coordinates (-0.5 to 0.5)
            x = px - 0.5
            y = py - 0.5
            
            # Check if this pixel is inside any of the 4 rods
            inside = False
            for rx, ry in [(d, d), (-d, d), (d, -d), (-d, -d)]:
                dist_sq = (x - rx)**2 + (y - ry)**2
                if dist_sq < r*r:
                    inside = True
                    break
            
            cell[i][j] = 1 if inside else 0
    
    return cell

def _quick_band_estimate(cell, size=32):
    """Very rough estimate of band structure using a simplified model.
    This is a surrogate to guide the search, not the actual evaluator.
    Uses a simple effective medium + Bragg scattering heuristic.
    """
    # Count dielectric fraction
    total = 0
    for row in cell:
        total += sum(row)
    f = total / (size * size)
    
    # For TM bands in a square lattice PC, the band gap between 4th and 5th bands
    # depends on the dielectric contrast and the Fourier components of the permittivity.
    # The key reciprocal lattice vector for the 4-5 gap is G = (1,1) in lattice units.
    # 
    # A rough heuristic: the gap is larger when the 4th-order Fourier coefficient
    # of 1/epsilon (for TM) is large. For a binary structure, this depends on
    # the spatial arrangement.
    
    # Compute the (1,1) Fourier coefficient of the binary mask
    # This is a very rough surrogate
    fx = 0.0
    fy = 0.0
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            val = cell[i][j]
            # Fourier coefficient for G=(1,1): exp(-i*2*pi*(x+y))
            fx += val * math.cos(2*math.pi*(x+y))
            fy -= val * math.sin(2*math.pi*(x+y))
    
    fourier_11 = math.sqrt(fx*fx + fy*fy) / (size*size)
    
    # Heuristic gap estimate: proportional to fourier_11 * (f - 0.5)
    # This is extremely rough but should correlate somewhat with the actual gap
    gap_est = fourier_11 * abs(f - 0.5) * 4.0
    
    return gap_est

def design(problem):
    """Design a 32x32 D4-symmetric unit cell for maximum TM 4-5 band gap."""
    start_time = time.time()
    deadline = start_time + 220  # 10s safety margin before 230s kill
    
    size = problem["pixels"]
    
    # Search over rod radius and offset for the 2x2 array
    # Parameter space: r in [0.05, 0.25], d in [0.05, 0.3]
    
    best_score = -1
    best_cell = None
    
    # Coarse grid search
    r_values = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
    d_values = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
    
    # First pass: coarse grid
    for d in d_values:
        if time.time() > deadline - 30:
            break
        for r in r_values:
            if time.time() > deadline - 30:
                break
            cell = _make_cell_2x2(r, d)
            score = _quick_band_estimate(cell, size)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # If we have time, do a finer search around the best point
    if best_cell is not None and time.time() < deadline - 20:
        # Find the best r and d from the coarse search
        best_r = None
        best_d = None
        best_score_fine = -1
        
        for d in d_values:
            for r in r_values:
                cell = _make_cell_2x2(r, d)
                score = _quick_band_estimate(cell, size)
                if score > best_score_fine:
                    best_score_fine = score
                    best_r = r
                    best_d = d
        
        if best_r is not None:
            # Fine search around best
            fine_r = [best_r + dr for dr in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]]
            fine_d = [best_d + dd for dd in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]]
            fine_r = [r for r in fine_r if 0.03 < r < 0.30]
            fine_d = [d for d in fine_d if 0.03 < d < 0.35]
            
            for d in fine_d:
                if time.time() > deadline - 10:
                    break
                for r in fine_r:
                    if time.time() > deadline - 10:
                        break
                    cell = _make_cell_2x2(r, d)
                    score = _quick_band_estimate(cell, size)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
    
    # If we still have time, try a few alternative configurations
    if time.time() < deadline - 15:
        # Try a ring/donut shape
        for r_inner in [0.05, 0.08, 0.10]:
            for r_outer in [0.15, 0.18, 0.20, 0.22]:
                if time.time() > deadline - 10:
                    break
                cell = [[0]*size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5)/size - 0.5
                        y = (j + 0.5)/size - 0.5
                        dist = math.sqrt(x*x + y*y)
                        if r_inner < dist < r_outer:
                            cell[i][j] = 1
                score = _quick_band_estimate(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    if best_cell is None:
        # Fallback: single central rod
        best_cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = (i + 0.5)/size - 0.5
                y = (j + 0.5)/size - 0.5
                row.append(1 if x*x + y*y < 0.15**2 else 0)
            best_cell.append(row)
    
    return best_cell
