"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit
    
    # Strategy: Use concentric square rings with alternating dielectric/air assignments
    # The cell is invariant under square symmetry group, so we can parameterize by
    # radial segments from center to edge.
    
    # For a 32x32 cell, the maximum Chebyshev distance from center is 15.5
    # We'll use rings defined by |x| + |y| or max(|x|, |y|) thresholds
    
    best_cell = None
    best_score = -1.0
    
    # Generate candidates using concentric square rings
    # Parameter: list of ring assignments (1=dielectric, 0=air) for each ring
    # Ring k covers pixels where max(|i-15.5|, |j-15.5|) is in [k, k+1)
    
    # Try various patterns of ring assignments
    # The key insight: for TM band gaps in square lattice photonic crystals,
    # we want periodic modulation of dielectric constant
    
    # Pattern 1: Single central dielectric square
    for half_size in range(4, 14):
        cell = [[0] * size for _ in range(size)]
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                if abs(i + 0.5 - center) < half_size and abs(j + 0.5 - center) < half_size:
                    cell[i][j] = 1
        if _check_symmetry(cell, size):
            # Heuristic: moderate sized squares tend to give better gaps
            score = 0.7 if 6 <= half_size <= 10 else 0.5
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Pattern 2: Concentric square rings with alternating pattern
    # Try different numbers of rings and patterns
    for num_rings in [3, 4, 5, 6]:
        for pattern in itertools.product([0, 1], repeat=num_rings):
            if time.time() > deadline - 5:
                break
            cell = [[0] * size for _ in range(size)]
            center = size / 2.0
            for i in range(size):
                for j in range(size):
                    # Chebyshev distance from center
                    dist = max(abs(i + 0.5 - center), abs(j + 0.5 - center))
                    ring_idx = int(dist)
                    if ring_idx < num_rings:
                        cell[i][j] = pattern[ring_idx]
            if _check_symmetry(cell, size):
                # Heuristic score based on pattern
                ones_count = sum(sum(row) for row in cell)
                # Moderate fill fraction tends to work well
                fill_frac = ones_count / (size * size)
                score = 0.75 if 0.3 < fill_frac < 0.7 else 0.5
                if score > best_score:
                    best_score = score
                    best_cell = cell
        if time.time() > deadline - 5:
            break
    
    # Pattern 3: Dielectric background with air holes arranged symmetrically
    for hole_half in range(2, 8):
        cell = [[1] * size for _ in range(size)]
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                if abs(i + 0.5 - center) < hole_half and abs(j + 0.5 - center) < hole_half:
                    cell[i][j] = 0
        if _check_symmetry(cell, size):
            score = 0.65
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Pattern 4: Checkerboard-like pattern with larger blocks
    for block_size in [4, 6, 8]:
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Block checkerboard
                block_i = i // block_size
                block_j = j // block_size
                if (block_i + block_j) % 2 == 0:
                    cell[i][j] = 1
        if _check_symmetry(cell, size):
            score = 0.6
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Pattern 5: Cross pattern (dielectric cross in air)
    for arm_width in range(2, 6):
        cell = [[0] * size for _ in range(size)]
        center = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - center + 0.5) < arm_width or abs(j - center + 0.5) < arm_width:
                    cell[i][j] = 1
        if _check_symmetry(cell, size):
            score = 0.55
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # If no valid cell found, create a simple symmetric pattern
    if best_cell is None:
        best_cell = [[0] * size for _ in range(size)]
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                if max(abs(i + 0.5 - center), abs(j + 0.5 - center)) < 8:
                    best_cell[i][j] = 1
    
    return best_cell

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
