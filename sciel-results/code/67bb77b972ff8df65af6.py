<<<<<<< SEARCH
def _initial_design_square(size=32, frac=0.35):
    """Create an initial square pillar design."""
    cell = np.zeros((size, size), dtype=np.float64)
    # Square pillar centered
    half = int(frac * size / 2)
    start = size // 2 - half
    end = size // 2 + half
    cell[start:end, start:end] = 1.0
    return _enforce_symmetry(cell)


def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell to maximize the TM band-4/5 gap.
    Uses a square pillar initial guess with local optimization.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Start with a square pillar design
    best_cell = _initial_design_square(size, frac=0.35)
    best_score = -np.inf
    
    # Try a few different initial designs
    initial_fracs = [0.25, 0.30, 0.35, 0.40, 0.45]
    
    for frac in initial_fracs:
        if time.time() > _DEADLINE - 30:
            break
        
        cell = _initial_design_square(size, frac=frac)
        
        # Evaluate this design
        try:
            score = _compute_score(cell, n_k=30, n_pw=17)
        except Exception:
            score = -1.0
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Try to improve with a simple local search on the square size
    if time.time() < _DEADLINE - 60:
        # Binary search / grid search on the square fraction
        for frac in np.arange(0.20, 0.55, 0.025):
            if time.time() > _DEADLINE - 30:
                break
            
            cell = _initial_design_square(size, frac=frac)
            try:
                score = _compute_score(cell, n_k=30, n_pw=17)
            except Exception:
                score = -1.0
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
    
    # Try a ring/square-ring design (hollow square)
    if time.time() < _DEADLINE - 40:
        for outer_frac in [0.45, 0.50, 0.55]:
            for inner_frac in [0.15, 0.20, 0.25, 0.30]:
                if time.time() > _DEADLINE - 30:
                    break
                
                cell = np.zeros((size, size), dtype=np.float64)
                outer_half = int(outer_frac * size / 2)
                inner_half = int(inner_frac * size / 2)
                o_start = size // 2 - outer_half
                o_end = size // 2 + outer_half
                i_start = size // 2 - inner_half
                i_end = size // 2 + inner_half
                
                cell[o_start:o_end, o_start:o_end] = 1.0
                cell[i_start:i_end, i_start:i_end] = 0.0
                
                cell = _enforce_symmetry(cell)
                
                try:
                    score = _compute_score(cell, n_k=30, n_pw=17)
                except Exception:
                    score = -1.0
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
    
    # Convert to 0/1 integer list
    best_cell_binary = (best_cell > 0.5).astype(int)
    
    # Ensure exact symmetry
    best_cell_binary = _enforce_symmetry(best_cell_binary).astype(int)
    best_cell_binary = (best_cell_binary > 0.5).astype(int)
    
    # Return as nested list
    return best_cell_binary.tolist()
=======
def _generate_superlattice_cell(size, rod_size, offset):
    """
    Generate a 2x2 superlattice of square rods with D4 symmetry.
    The unit cell contains 4 rods. Due to D4 symmetry of the whole 32x32 cell,
    we define the pattern in the top-left quadrant (16x16) and reflect it.
    However, a 2x2 superlattice usually implies periodicity over 2 cells.
    To fit in a single 32x32 cell with D4 symmetry, we place 4 rods centered
    at (8,8), (24,8), (8,24), (24,24) approximately, or more generally
    at positions that maintain the symmetry.
    
    Let's place rods centered at (size/4, size/4), (3*size/4, size/4),
    (size/4, 3*size/4), (3*size/4, 3*size/4).
    """
    cell = np.zeros((size, size), dtype=np.float64)
    centers = [
        (size // 4, size // 4),
        (3 * size // 4, size // 4),
        (size // 4, 3 * size // 4),
        (3 * size // 4, 3 * size // 4)
    ]
    half_rod = rod_size // 2
    for cx, cy in centers:
        start_r = max(0, cy - half_rod)
        end_r = min(size, cy + half_rod)
        start_c = max(0, cx - half_rod)
        end_c = min(size, cx + half_rod)
        cell[start_r:end_r, start_c:end_c] = 1.0
    return _enforce_symmetry(cell)

def _generate_ca_cell(size, seed, steps):
    """
    Generate a D4-symmetric cell using Game of Life evolution.
    """
    # Initialize a 16x16 quadrant with random or simple seed
    q = np.zeros((size//2, size//2), dtype=int)
    np.random.seed(seed)
    # Simple seed: a few random pixels
    for _ in range(5):
        q[np.random.randint(0, size//2), np.random.randint(0, size//2)] = 1
    
    # Evolve Game of Life in the quadrant, reflecting boundaries? 
    # For simplicity, we evolve a 32x32 grid but enforce symmetry after each step?
    # Or just evolve the full grid and then enforce symmetry.
    grid = np.zeros((size, size), dtype=int)
    # Map quadrant to full grid with symmetry
    grid[:size//2, :size//2] = q
    grid[:size//2, size//2:] = q[:, ::-1]
    grid[size//2:, :size//2] = q[::-1, :]
    grid[size//2:, size//2:] = q[::-1, ::-1]
    
    for _ in range(steps):
        # Count neighbors
        neighbors = np.zeros_like(grid)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
        
        # Apply rules
        new_grid = np.zeros_like(grid)
        new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
        new_grid[(grid == 0) & (neighbors == 3)] = 1
        
        # Enforce D4 symmetry
        grid = new_grid
        grid = _enforce_symmetry(grid).astype(int)
        
    return grid

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell to maximize the TM band-4/5 gap.
    Uses a 2x2 superlattice of rods as the primary strategy.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    best_cell = np.zeros((size, size), dtype=np.float64)
    best_score = -np.inf
    
    # Strategy 1: 2x2 Superlattice of Rods (Proven effective)
    # Search over rod sizes
    rod_sizes = range(4, 12, 1)
    for rs in rod_sizes:
        if time.time() > _DEADLINE - 20:
            break
        cell = _generate_superlattice_cell(size, rs, 0)
        try:
            score = _compute_score(cell, n_k=30, n_pw=17)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    # Strategy 2: Cellular Automata (Game of Life)
    # Try a few seeds and steps
    for seed in [1, 2, 3, 4, 5]:
        for steps in [2, 3, 4]:
            if time.time() > _DEADLINE - 20:
                break
            cell = _generate_ca_cell(size, seed, steps)
            try:
                score = _compute_score(cell, n_k=30, n_pw=17)
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                
    # Strategy 3: Single Central Rod (Baseline)
    for frac in [0.3, 0.4, 0.5]:
        if time.time() > _DEADLINE - 20:
            break
        half = int(frac * size / 2)
        start = size // 2 - half
        end = size // 2 + half
        cell = np.zeros((size, size), dtype=np.float64)
        cell[start:end, start:end] = 1.0
        cell = _enforce_symmetry(cell)
        try:
            score = _compute_score(cell, n_k=30, n_pw=17)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    # Convert to 0/1 integer list
    best_cell_binary = (best_cell > 0.5).astype(int)
    
    # Ensure exact symmetry
    best_cell_binary = _enforce_symmetry(best_cell_binary).astype(int)
    best_cell_binary = (best_cell_binary > 0.5).astype(int)
    
    # Return as nested list
    return best_cell_binary.tolist()
>>>>>>> REPLACE
