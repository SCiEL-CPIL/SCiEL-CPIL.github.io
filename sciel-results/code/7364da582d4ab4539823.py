"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_cell(size, rod_half, bg=0):
    """Build a symmetric cell with a central square rod of half-size rod_half pixels."""
    cell = np.full((size, size), bg, dtype=np.int8)
    center = size / 2.0
    for i in range(size):
        for j in range(size):
            # Distance from center in pixel units
            dx = abs(i + 0.5 - center)
            dy = abs(j + 0.5 - center)
            # Square rod: max(dx, dy) <= rod_half
            if max(dx, dy) < rod_half:
                cell[i, j] = 1
    return cell.tolist()

def _build_cell_circular(size, radius, bg=0):
    """Build a symmetric cell with a central circular rod of given radius (in lattice constants)."""
    cell = np.full((size, size), bg, dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < radius * radius:
                cell[i, j] = 1
    return cell.tolist()

def design(problem):
    """
    Design a symmetric 32x32 unit cell for TM band gap optimization.
    
    The seed uses a circular rod of radius 0.2. We try several geometric
    configurations that are known to open larger band gaps in 2D square-lattice
    photonic crystals for TM polarization:
    
    1. Circular rods of various radii
    2. Square rods of various sizes
    3. Inverted structures (dielectric background with air holes)
    
    Since we cannot run the band structure solver here, we rely on known
    physics results. For TM polarization in 2D square lattices, the best
    band gaps are typically achieved with:
    - Circular holes in a dielectric background (inverted structure)
    - Or specific rod geometries
    
    The classic result for TM band gaps in square lattices is that the
    optimal structure is often a dielectric background with circular air
    holes, or a specific rod geometry.
    
    Given the constraints and the need for a deterministic, fast solution,
    we'll try a few candidate structures and return one that is known to
    perform well. The circular rod of radius ~0.2-0.3 is the textbook case,
    but inverted structures (dielectric with air holes) often give larger gaps.
    
    Let's try a dielectric background with a central circular air hole.
    """
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try several candidate structures
    # For TM polarization, inverted structures (dielectric bg, air holes) often give larger gaps
    # But the seed uses dielectric rods in air. Let's try both.
    
    # Candidate 1: Original circular rod (radius 0.2)
    candidates = []
    
    # Try circular rods with different radii
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        cell = _build_cell_circular(size, r, bg=0)
        candidates.append(cell)
    
    # Try square rods
    for half in [3, 4, 5, 6, 7, 8]:
        cell = _build_cell(size, half, bg=0)
        candidates.append(cell)
    
    # Try inverted: dielectric background with central air hole
    # This is often better for TM gaps
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        cell = _build_cell_circular(size, r, bg=1)
        # Invert: where the hole is, set to 0
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < r * r:
                    cell[i][j] = 0
        candidates.append(cell)
    
    # Try inverted square holes
    for half in [3, 4, 5, 6, 7, 8]:
        cell = _build_cell(size, half, bg=1)
        # Invert the central square to air
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if max(dx, dy) < half:
                    cell[i][j] = 0
        candidates.append(cell)
    
    # Without being able to evaluate the band gap, we need to pick the most
    # likely candidate. Based on photonic crystal literature:
    # - For TM polarization in 2D square lattices, the 4th-5th band gap
    #   is typically optimized with structures that have moderate filling fractions.
    # - The circular rod of radius ~0.2 (seed) gives a baseline.
    # - Inverted structures often give larger gaps for TM.
    
    # Since we can't evaluate, let's return the seed's structure as a safe choice,
    # but slightly optimized. Actually, let's try a slightly larger circular rod
    # which may open a larger gap.
    
    # Based on typical results, a circular rod with radius around 0.25-0.3
    # in a square lattice often gives good TM band gaps.
    
    # Let's return a circular rod of radius 0.25
    cell = _build_cell_circular(size, 0.25, bg=0)
    
    return cell
# EVOLVE-BLOCK-END
