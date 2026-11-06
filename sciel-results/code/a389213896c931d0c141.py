"""Design using D4-symmetric cellular automaton (Game of Life) for complex microstructure."""
import math
import time

def _make_ca_cell(seed_pattern, n_steps, size=32):
    """Create a 32x32 D4-symmetric cell by evolving a CA from a symmetric seed.
    
    seed_pattern: a small D4-symmetric pattern (e.g., 3x3 or 5x5)
    n_steps: number of CA steps to evolve
    Returns 32x32 list of 0/1.
    """
    # Initialize a larger grid for CA evolution
    ca_size = 64  # Work on 64x64 for better resolution, then downsample
    ca = [[0]*ca_size for _ in range(ca_size)]
    
    # Place seed pattern at center with D4 symmetry
    cx, cy = ca_size // 2, ca_size // 2
    seed_size = len(seed_pattern)
    seed_half = seed_size // 2
    
    for i in range(seed_size):
        for j in range(seed_size):
            if seed_pattern[i][j] == 1:
                # Place with D4 symmetry (8 positions)
                positions = [
                    (cx + (i - seed_half), cy + (j - seed_half)),
                    (cx - (i - seed_half), cy + (j - seed_half)),
                    (cx + (j - seed_half), cy + (i - seed_half)),
                    (cx - (j - seed_half), cy + (i - seed_half)),
                    (cx + (i - seed_half), cy - (j - seed_half)),
                    (cx - (i - seed_half), cy - (j - seed_half)),
                    (cx + (j - seed_half), cy - (i - seed_half)),
                    (cx - (j - seed_half), cy - (i - seed_half)),
                ]
                for px, py in positions:
                    if 0 <= px < ca_size and 0 <= py < ca_size:
                        ca[px][py] = 1
    
    # Evolve CA for n_steps
    for step in range(n_steps):
        new_ca = [[0]*ca_size for _ in range(ca_size)]
        for i in range(ca_size):
            for j in range(ca_size):
                # Count live neighbors (toroidal boundary)
                neighbors = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni = (i + di) % ca_size
                        nj = (j + dj) % ca_size
                        neighbors += ca[ni][nj]
                
                # Game of Life rules
                if ca[i][j] == 1:
                    if neighbors in [2, 3]:
                        new_ca[i][j] = 1
                else:
                    if neighbors == 3:
                        new_ca[i][j] = 1
        ca = new_ca
    
    # Downsample to 32x32 with D4 symmetry enforcement
    cell = [[0]*size for _ in range(size)]
    scale = ca_size // size
    
    for i in range(size):
        for j in range(size):
            # Sample the CA grid at this position
            ca_i = i * scale + scale // 2
            ca_j = j * scale + scale // 2
            cell[i][j] = ca[ca_i][ca_j]
    
    # Enforce D4 symmetry about cell center
    # cell[i][j] == cell[size-1-i][j] == cell[i][size-1-j] == cell[size-1-i][size-1-j]
    # cell[i][j] == cell[j][i] (transpose symmetry)
    for i in range(size):
        for j in range(size):
            # Average over the 8 symmetric positions and threshold
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i),
            ]
            val = sum(cell[pi][pj] for pi, pj in positions)
            cell[i][j] = 1 if val >= 4 else 0
    
    return cell

def _count_dielectric(cell, size=32):
    """Count the fraction of dielectric pixels."""
    total = sum(sum(row) for row in cell)
    return total / (size * size)

def _fourier_coeff_11(cell, size=32):
    """Compute the magnitude of the (1,1) Fourier coefficient of the binary mask."""
    fx = 0.0
    fy = 0.0
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            val = cell[i][j]
            fx += val * math.cos(2*math.pi*(x+y))
            fy -= val * math.sin(2*math.pi*(x+y))
    return math.sqrt(fx*fx + fy*fy) / (size*size)

def _estimate_gap(cell, size=32):
    """Rough estimate of the 4-5 band gap using Fourier analysis."""
    f = _count_dielectric(cell, size)
    fourier_11 = _fourier_coeff_11(cell, size)
    
    # The gap is enhanced by high-frequency content and moderate dielectric fraction
    # For TM polarization, the relevant quantity is 1/epsilon modulation
    # A rough heuristic: gap ~ |f - 0.5| * fourier_11 * contrast_factor
    contrast = 1 - 1/12  # epsilon contrast factor
    gap_est = abs(f - 0.5) * fourier_11 * contrast * 10.0
    
    return gap_est

def design(problem):
    """Design a 32x32 D4-symmetric unit cell for maximum TM 4-5 band gap using CA."""
    start_time = time.time()
    deadline = start_time + 220  # 10s safety margin before 230s kill
    
    size = problem["pixels"]
    
    # Define several D4-symmetric seed patterns
    seed_patterns = [
        # Single pixel
        [[1]],
        # 3x3 cross
        [[0,1,0],
         [1,1,1],
         [0,1,0]],
        # 3x3 block
        [[1,1,1],
         [1,1,1],
         [1,1,1]],
        # 5x5 ring
        [[0,0,1,0,0],
         [0,1,1,1,0],
         [1,1,0,1,1],
         [0,1,1,1,0],
         [0,0,1,0,0]],
        # 5x5 plus
        [[0,0,1,0,0],
         [0,0,1,0,0],
         [1,1,1,1,1],
         [0,0,1,0,0],
         [0,0,1,0,0]],
        # 5x5 X
        [[1,0,0,0,1],
         [0,1,0,1,0],
         [0,0,1,0,0],
         [0,1,0,1,0],
         [1,0,0,0,1]],
    ]
    
    best_score = -1
    best_cell = None
    best_params = None
    
    # Search over seed patterns and CA steps
    for seed_idx, seed in enumerate(seed_patterns):
        for n_steps in range(0, 8):
            if time.time() > deadline - 30:
                break
            
            cell = _make_ca_cell(seed, n_steps, size)
            f = _count_dielectric(cell, size)
            
            # Skip if dielectric fraction is too extreme
            if f < 0.15 or f > 0.85:
                continue
            
            score = _estimate_gap(cell, size)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_params = (seed_idx, n_steps)
    
    # If we have time, try to refine by trying more CA steps or different thresholds
    if best_cell is not None and time.time() < deadline - 15:
        # Try additional CA steps around the best
        best_seed_idx, best_steps = best_params
        for extra_steps in range(1, 5):
            if time.time() > deadline - 10:
                break
            cell = _make_ca_cell(seed_patterns[best_seed_idx], best_steps + extra_steps, size)
            f = _count_dielectric(cell, size)
            if 0.15 < f < 0.85:
                score = _estimate_gap(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Fallback if CA didn't produce good results
    if best_cell is None or best_score <= 0:
        # Use a simple square pillar as fallback
        best_cell = [[0]*size for _ in range(size)]
        pillar_size = 12  # 12x12 square pillar in center
        offset = (size - pillar_size) // 2
        for i in range(offset, offset + pillar_size):
            for j in range(offset, offset + pillar_size):
                best_cell[i][j] = 1
    
    return best_cell
